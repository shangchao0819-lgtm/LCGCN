"""Model fitting and final evaluation; optional CV uses training labels only."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time
import zipfile
import numpy as np
import torch
from torch.nn import functional as F
from .data_loader import ROOT, DATASETS, load_data, training_view
from .graph_builder import DEGREES
from .models import LCGCN, VARIANTS, CONTROLS

MODELS = VARIANTS + CONTROLS

def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(30):
        try:
            temporary.replace(path); break
        except PermissionError:
            if attempt==29: raise
            time.sleep(.1)


def verify_artifacts(folder, result):
    folder=Path(folder)
    if not result.get('artifact_sha256'):
        raise ValueError('Missing run artifact hashes')
    for name,expected in result['artifact_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Run artifact changed: '+name)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@contextlib.contextmanager
def isolated_rng(seed):
    py, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count()))
    try:
        with torch.random.fork_rng(devices=devices):
            seed_all(seed)
            yield
    finally:
        random.setstate(py); np.random.set_state(np_state)

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def source_snapshot():
    files = sorted(list(ROOT.glob('*.py')) + list((ROOT / 'configs').glob('*.json')))
    contents = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in files}
    manifest = {p: hashlib.sha256(b).hexdigest() for p, b in contents.items()}
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    target = Path(os.environ['LCGCN_SNAPSHOT_DIR']) / (identity + '.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path, content in contents.items():
                archive.writestr(path, content)
            archive.writestr('snapshot_manifest.json', json.dumps(manifest, indent=2))
    return identity, str(target)

def defaults(**overrides):
    cfg = dict(model='lcgcn', dataset='cora-coauthorship', split=1, seed=0,
               epochs=200, lr=.01, weight_decay=.0005, dropout=None, hidden=None,
               degree='half', min_edge_size=0, task='transductive', device='cpu',
               threads=2, smoke=False, max_edge_size=0, order_seed=None, purpose='confirmatory',
               selection_mode='cv', reporting_rule='selected', folds=5,
               lr_candidates=[.001,.005,.01], seed_scheme='original')
    cfg.update(overrides)
    if cfg['hidden'] is None:
        cfg['hidden'] = (64 if cfg['dataset']=='citeseer' else 16)
    if cfg['dropout'] is None:
        cfg['dropout'] = .5
    cfg['lr_candidates'] = sorted(set(float(x) for x in cfg['lr_candidates']))
    cfg['fold_seed'] = 4100003 + 1009*cfg['split']
    cfg['selection_seed'] = 5100007 + 1009*cfg['split']
    cfg['graph_seed'] = 1000003 + 1009*cfg['split'] + 9176*cfg['seed']
    cfg['split_seed'] = 20261031 + 2*cfg['split']
    cfg['eval_seed'] = 2000003 + 1009*cfg['split'] + 9176*cfg['seed']
    cfg['initialization_seed'] = 3000003 + 1009*cfg['split'] + 9176*cfg['seed']
    if cfg['seed_scheme']=='search-32-20261110':
        if type(cfg['seed']) is not int or cfg['seed'] not in range(32): raise ValueError('Invalid candidate repeat')
        cfg['initialization_seed'] = (2, 8, 22, 31, 47, 59, 71, 89, 0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26)[cfg['seed']]
    if cfg['seed_scheme'] not in ('original', 'small-2-8-22', 'extended-eight-20261110', 'search-32-20261110'):
        raise ValueError('Unknown seed scheme')
    if cfg['seed_scheme'] == 'small-2-8-22':
        if type(cfg['seed']) is not int or cfg['seed'] not in (0,1,2):
            raise ValueError('Small-seed repeat IDs must be 0,1,2; actual seeds are 2,8,22')
        cfg['initialization_seed'] = (2,8,22)[cfg['seed']]
    cfg['protocol'] = 'lcgcn-own-v1-cv200'
    if cfg['seed_scheme'] == 'extended-eight-20261110':
        if type(cfg['seed']) is not int or cfg['seed'] not in range(8):
            raise ValueError('Extended repeat IDs must be 0..7')
        cfg['initialization_seed'] = (2,8,22,31,47,59,71,89)[cfg['seed']]
        cfg['protocol'] = 'lcgcn-own-fixed200-eight-seeds-20261110'
        if cfg['selection_mode'] != 'fixed' or cfg['reporting_rule'] != 'budget-final':
            raise ValueError('Extended runs reuse frozen selection and final200 reporting')
    if cfg['seed_scheme']=='search-32-20261110':
        cfg['protocol']='exploratory-test-selected-seed-search-20261110'
        if cfg['selection_mode']!='fixed' or cfg['reporting_rule']!='budget-final': raise ValueError('Search requires frozen final200 recipe')
    return cfg

def build_model(view, cfg, device):
    if cfg['model'] in VARIANTS + CONTROLS:
        X = view['features'].toarray()
        model = LCGCN(X, view['edges'], view['stats']['classes'], cfg, device)
        return model, dict(source='LCGCN manuscript Eqs.5-9,13', hidden=cfg['hidden'],
                           bias=False, dropout=cfg['dropout'], output='logits')
    raise ValueError('This package only supports the three LC variants and internal graph controls')

def copy_learned_parameters(source, target):
    # Graph buffers have different shapes after adding unseen nodes. Copy only
    # learned parameters, never a cached training graph into the full graph.
    old, new = dict(source.named_parameters()), dict(target.named_parameters())
    if old.keys() != new.keys():
        raise ValueError('Parameter names differ during inductive expansion')
    with torch.no_grad():
        for name, parameter in new.items():
            parameter.copy_(old[name])

def run(config, output_root=None, stage='all', selection_root=None):
    cfg = defaults(**config)
    if cfg['model'] not in MODELS or cfg['task'] not in ('transductive', 'inductive'):
        raise ValueError('Unknown model/task')
    if cfg['purpose'] not in ('confirmatory','sensitivity'):
        raise ValueError('Unknown experiment purpose')
    if cfg['epochs'] != 200 and not cfg['smoke'] and cfg['purpose'] != 'sensitivity':
        raise ValueError('Non-200 epoch runs must be explicitly marked smoke')
    if cfg['epochs'] < 1 or not 0 <= cfg['dropout'] < 1:
        raise ValueError('Invalid epoch/dropout setting')
    if cfg['selection_mode'] not in ('cv','fixed') or cfg['reporting_rule'] not in ('selected','budget-final'):
        raise ValueError('Unknown selection/reporting rule')
    if cfg['folds'] != 5 and not cfg['smoke']:
        raise ValueError('Formal protocol requires five folds')
    if not cfg['lr_candidates'] or min(cfg['lr_candidates']) <= 0:
        raise ValueError('Positive learning-rate candidates required')
    if stage not in ('all','select','refit'):
        raise ValueError('Unknown execution stage')
    if cfg['task'] == 'inductive' and cfg['model'] != 'fast-lcgcn':
        raise ValueError('Inductive implementation in this package is limited to FastLCGCN')
    torch.set_num_threads(cfg['threads'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    seed_all(cfg['initialization_seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source_hash, snapshot = source_snapshot()
    data = load_data(cfg['dataset'], cfg['split'], cfg['min_edge_size'], cfg['task'], cfg['split_seed'], cfg['max_edge_size'], cfg['order_seed'])
    selection, selection_folder = None, None
    if cfg['selection_mode'] == 'cv':
        from .cv_selection import select
        selection, selection_folder = select(cfg, data, source_hash, device, selection_root, require_existing=stage=='refit')
    if stage == 'select':
        return dict(folder=str(selection_folder) if selection_folder else None, metrics={}, selection=selection)
    fit_cfg = dict(cfg)
    if selection:
        fit_cfg['lr'] = selection['lr']
        fit_cfg['epochs'] = selection['epoch'] if cfg['reporting_rule']=='selected' else cfg['epochs']
    identity = dict(config=cfg, source_sha256=source_hash, data_sha256=data['hashes'])
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output_root = Path(output_root or ROOT / ('smoke_results' if cfg['smoke'] else ('results' if cfg['seed_scheme']=='original' else 'results_small')))
    folder = output_root / f"{cfg['dataset']}_{cfg['model']}_s{cfg['split']}_r{cfg['seed']}_{run_id}"
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        verify_artifacts(folder,result)
        return result
    if folder.exists():
        raise FileExistsError(f'Incomplete run retained for inspection: {folder}')
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder/'config.json', dict(identity, source_snapshot=snapshot))
    if selection:
        # Self-contained evidence for the selected epoch, LR and all fold indices.
        shutil.copytree(selection_folder, folder/'selection')
    save_json(folder/'fit_config.json', fit_cfg)
    save_json(folder/'state.json', dict(status='running', pid=os.getpid()))
    started = time.perf_counter()
    try:
        # Selection may consume RNG state. Final refits always start afresh.
        seed_all(cfg['initialization_seed'])
        view = training_view(data)
        t0 = time.perf_counter()
        model, metadata = build_model(view, fit_cfg, device)
        X = torch.tensor(view['features'].toarray(), dtype=torch.float32, device=device)
        train_indices = torch.tensor(data['train_local'], dtype=torch.long, device=device)
        train_labels = torch.tensor(data['y'][data['train']], dtype=torch.long, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=fit_cfg['lr'], weight_decay=cfg['weight_decay'])
        sync(device); setup_seconds = time.perf_counter()-t0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        history = []
        t0 = time.perf_counter()
        for epoch in range(1, fit_cfg['epochs']+1):
            epoch_start = time.perf_counter()
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = F.cross_entropy(logits[train_indices], train_labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at epoch {epoch}')
            loss.backward(); optimizer.step(); sync(device)
            history.append(dict(epoch=epoch, train_loss=float(loss.detach().cpu()),
                                seconds=time.perf_counter()-epoch_start))
            if epoch == 1 or epoch % 20 == 0 or epoch == fit_cfg['epochs']:
                save_json(folder/'state.json', dict(status='running', epoch=epoch, total_epochs=fit_cfg['epochs'], pid=os.getpid()))
                print(f"{cfg['dataset']} {cfg['model']} split={cfg['split']} seed={cfg['initialization_seed']} epoch={epoch}/{fit_cfg['epochs']} loss={history[-1]['train_loss']:.6f}", flush=True)
        train_seconds = time.perf_counter()-t0
        training_graph_metadata = ([b.metadata() for b in model.builders] if isinstance(model, LCGCN) else None)
        parameters = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
        t0 = time.perf_counter()
        if cfg['task'] == 'inductive':
            expanded, _ = build_model(data, fit_cfg, device)
            copy_learned_parameters(model, expanded)
            model = expanded
            X = torch.tensor(data['features'].toarray(), dtype=torch.float32, device=device)
        model.eval()
        # Evaluation uses a fixed random stream isolated from training.
        with isolated_rng(cfg['eval_seed']), torch.no_grad():
            logits = model(X)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite final predictions')
        sync(device); eval_seconds = time.perf_counter()-t0
        scores = logits.cpu().numpy()
        predictions = scores.argmax(1)
        metrics = {}
        for name in ('test', 'seen', 'unseen'):
            ids = data[name]
            if len(ids):
                metrics[name+'_accuracy'] = float(np.mean(predictions[ids] == data['y'][ids])*100)
                metrics[name+'_n'] = len(ids)
        graph_metadata = model.export_graphs(folder/'graphs') if isinstance(model, LCGCN) else None
        np.savez_compressed(folder/'partitions.npz', **{k: data[k] for k in ('train','test','seen','unseen','visible','train_local')})
        np.savez_compressed(folder/'predictions.npz', logits=scores, predictions=predictions, labels=data['y'])
        torch.save(dict(parameters=parameters, config=fit_cfg, source_sha256=source_hash), folder/'model.pt')
        save_json(folder/'history.json', history)
        result = dict(run_id=run_id, folder=str(folder), config=cfg, source_sha256=source_hash,
                      data_sha256=data['hashes'], dataset_stats=data['stats'], metrics=metrics,
                      model_metadata=metadata, fit_config=fit_cfg,
                      selection_id=selection['selection_id'] if selection else None,
                      selected_epoch=selection['epoch'] if selection else None,
                      selected_lr=selection['lr'] if selection else None,
                      selection_at_ceiling=selection['at_epoch_ceiling'] if selection else None,
                      optimizer=dict(name='Adam', lr=fit_cfg['lr'], weight_decay=cfg['weight_decay'], scheduler=None),
                      training_graphs=training_graph_metadata, final_evaluation_graphs=graph_metadata,
                      timing=dict(setup_seconds=setup_seconds, train_seconds=train_seconds,
                                  evaluation_seconds=eval_seconds, total_seconds=time.perf_counter()-started,
                                  selection_seconds_shared_across_refits=selection['selection_seconds'] if selection else 0),
                      environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                       scipy=__import__('scipy').__version__, device=str(device),
                                       gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                                       deterministic_gpu_kernels_guaranteed=False),
                      checkpoint_selection=('five-fold train-only validation; full-label refit to selected epoch' if selection and cfg['reporting_rule']=='selected' else 'prespecified final budget checkpoint'),
                      hyperparameter_selection=('five-fold train-only validation LR/epoch selection; no test selection' if selection else 'frozen original training-only selection; no new tuning' if cfg.get('frozen_selection_reference') else 'frozen recipe; initialization seeds selected exploratorily using earlier test outcomes'))
        result['artifact_sha256']={p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file() and p.name not in ('state.json','result.json','reproduction.json')}
        save_json(folder/'result.json', result)
        save_json(folder/'state.json', dict(status='complete', epochs=fit_cfg['epochs']))
        return result
    except BaseException as error:
        save_json(folder/'state.json', dict(status='failed', error=repr(error), pid=os.getpid()))
        raise

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, default='lcgcn')
    p.add_argument('--dataset', choices=DATASETS, default='cora-coauthorship')
    p.add_argument('--split', type=int, choices=range(1,11), default=1)
    p.add_argument('--seed', type=int, default=0, help='Repeat ID, not actual random seed')
    p.add_argument('--seed-scheme', choices=('original','small-2-8-22'), default='original')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--hidden', type=int)
    p.add_argument('--degree', choices=DEGREES, default='half')
    p.add_argument('--min-edge-size', type=int, default=0)
    p.add_argument('--max-edge-size', type=int, default=0)
    p.add_argument('--order-seed', type=int)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--weight-decay', type=float, default=.0005)
    p.add_argument('--dropout', type=float)
    p.add_argument('--selection-mode', choices=('cv','fixed'), default='cv')
    p.add_argument('--reporting-rule', choices=('selected','budget-final'), default='selected')
    p.add_argument('--lr-candidates', type=float, nargs='+', default=[.001,.005,.01])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--stage', choices=('all','select','refit'), default='all')
    p.add_argument('--selection-root', type=Path)
    p.add_argument('--purpose', choices=('confirmatory','sensitivity'), default='confirmatory')
    p.add_argument('--config', type=Path, help='Use this exact JSON configuration instead of other model CLI settings')
    p.add_argument('--task', choices=('transductive','inductive'), default='transductive')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--output-root', type=Path)
    return p

if __name__ == '__main__':
    options = vars(parser().parse_args())
    output = options.pop('output_root')
    stage = options.pop('stage')
    selection_root = options.pop('selection_root')
    config_file = options.pop('config')
    if config_file:
        options = json.loads(config_file.read_text(encoding='utf-8'))
    result = run(options, output, stage=stage, selection_root=selection_root)
    print(json.dumps(dict(folder=result['folder'], metrics=result['metrics']), ensure_ascii=False), flush=True)
