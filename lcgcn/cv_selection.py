"""Training-label-only five-fold selection; no test scores enter this module."""
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
from .data_loader import ROOT, training_view

SELECTION_FIELDS = ('protocol','model','dataset','split','epochs','weight_decay','dropout','hidden',
                    'degree','min_edge_size','max_edge_size','order_seed','task','device','threads',
                    'folds','fold_seed','selection_seed','lr_candidates','smoke','selection_mode')


def folds_for(data, cfg):
    train = np.asarray(data['train'], dtype=np.int64)
    labels = data['y'][train]
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < cfg['folds']:
        raise ValueError('Every training class must have at least one label per fold')
    splitter = StratifiedKFold(cfg['folds'], shuffle=True, random_state=cfg['fold_seed'])
    folds = [(train[a], train[b]) for a, b in splitter.split(train, labels)]
    return folds


def choose(curves, weights):
    """curves: LR -> folds x epochs. Exact ties: earlier epoch, smaller LR."""
    choices, averages = [], {}
    for lr, values in curves.items():
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError('All fold/epoch validation losses must be finite')
        avg = np.average(values, axis=0, weights=weights)
        averages[str(lr)] = avg.tolist()
        choices.extend((float(loss), epoch + 1, float(lr)) for epoch, loss in enumerate(avg))
    loss, epoch, lr = min(choices)
    return dict(lr=lr, epoch=epoch, validation_cross_entropy=loss,
                weighted_validation_curves=averages)


def selection_identity(cfg, data, source):
    value = dict(config={k: cfg[k] for k in SELECTION_FIELDS}, source_sha256=source,
                 data_sha256=data['hashes'], train_indices=data['train'].tolist(),
                 visible_indices_sha256=hashlib.sha256(data['visible'].tobytes()).hexdigest())
    key = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return key, value


def select(cfg, data, source, device, cache_root=None, require_existing=False):
    # Imported lazily to keep train.py importable without circular dependencies.
    from .train import build_model, isolated_rng, seed_all, sync, save_json
    key, identity = selection_identity(cfg, data, source)
    root = Path(cache_root or ROOT / ('smoke_selections' if cfg['smoke'] else 'selections'))
    folder = root / key
    report_path = folder / 'selection.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['identity'] != identity or report['status'] != 'complete':
            raise ValueError('Selection cache identity mismatch')
        for name, expected in report['artifact_sha256'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Selection cache artifact changed: {name}')
        curves={lr:[[r['validation_cross_entropy'] for r in json.loads(
            (folder/f'lr{lr}_fold{fold}.json').read_text(encoding='utf-8'))['history']]
            for fold in range(cfg['folds'])] for lr in cfg['lr_candidates']}
        verified=choose(curves,[r['validation_nodes'] for r in report['folds']])
        if any(report[k]!=verified[k] for k in verified):
            raise ValueError('Selection result does not match its validation histories')
        return report, folder
    if require_existing:
        raise FileNotFoundError('Run the selection stage before refitting: ' + str(folder))
    if folder.exists():
        raise FileExistsError('Incomplete selection retained; inspect before retry: ' + str(folder))
    folder.mkdir(parents=True)
    save_json(folder/'identity.json', identity)
    started = time.perf_counter()
    try:
        folds = folds_for(data, cfg)
        inverse = np.full(len(data['y']), -1, dtype=np.int64)
        inverse[data['visible']] = np.arange(len(data['visible']))
        view = training_view(data)
        X = torch.tensor(view['features'].toarray(), dtype=torch.float32, device=device)
        curves, fold_records = {}, []
        partitions = {}
        for fold, (fit, val) in enumerate(folds):
            partitions[f'fold{fold}_train'] = fit
            partitions[f'fold{fold}_validation'] = val
            fold_records.append(dict(fold=fold, train_nodes=len(fit), validation_nodes=len(val),
                train_per_class=np.bincount(data['y'][fit], minlength=data['stats']['classes']).tolist(),
                validation_per_class=np.bincount(data['y'][val], minlength=data['stats']['classes']).tolist(),
                initialization_seed=cfg['selection_seed']+100003*fold,
                graph_seed=cfg['selection_seed']+100003*fold+10000019,
                evaluation_seed=cfg['selection_seed']+100003*fold+20000033))
        np.savez_compressed(folder/'fold_indices.npz', **partitions)
        for lr in cfg['lr_candidates']:
            curves[lr] = []
            for fold, (fit, val) in enumerate(folds):
                settings = fold_records[fold]
                fold_cfg = dict(cfg, lr=lr, initialization_seed=settings['initialization_seed'],
                                graph_seed=settings['graph_seed'], eval_seed=settings['evaluation_seed'])
                # Reuse initial random streams across LR candidates, not across folds.
                seed_all(fold_cfg['initialization_seed'])
                model, metadata = build_model(view, fold_cfg, device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
                fit_ids = torch.as_tensor(inverse[fit], dtype=torch.long, device=device)
                val_ids = torch.as_tensor(inverse[val], dtype=torch.long, device=device)
                fit_y = torch.as_tensor(data['y'][fit], dtype=torch.long, device=device)
                val_y = torch.as_tensor(data['y'][val], dtype=torch.long, device=device)
                history = []
                for epoch in range(1, cfg['epochs']+1):
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    train_loss = F.cross_entropy(model(X)[fit_ids], fit_y)
                    if not torch.isfinite(train_loss):
                        raise FloatingPointError('Nonfinite inner training loss')
                    train_loss.backward(); optimizer.step()
                    model.eval()
                    with isolated_rng(fold_cfg['eval_seed']), torch.no_grad():
                        validation_loss = F.cross_entropy(model(X)[val_ids], val_y)
                    if not torch.isfinite(validation_loss):
                        raise FloatingPointError('Nonfinite validation loss')
                    history.append(dict(epoch=epoch, train_loss=float(train_loss.detach().cpu()),
                                        validation_cross_entropy=float(validation_loss.cpu())))
                curves[lr].append([r['validation_cross_entropy'] for r in history])
                save_json(folder/f'lr{lr}_fold{fold}.json', dict(config=fold_cfg, metadata=metadata, history=history))
                print(f"selection {cfg['dataset']} {cfg['model']} split={cfg['split']} lr={lr} fold={fold+1}/{cfg['folds']} complete", flush=True)
                del model, optimizer
                gc.collect()
        winner = choose(curves, [len(v) for _, v in folds])
        sync(device)
        files = [p for p in folder.iterdir() if p.is_file()]
        report = dict(status='complete', selection_id=key, identity=identity, **winner,
                      folds=fold_records, selection_fit_count=len(cfg['lr_candidates'])*len(folds),
                      at_epoch_ceiling=winner['epoch']==cfg['epochs'],
                      selection_seconds=time.perf_counter()-started,
                      criterion='validation-node-weighted mean cross entropy across inner folds',
                      test_labels_used=False,
                      artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        save_json(report_path, report)
        return report, folder
    except BaseException as error:
        save_json(folder/'failed.json', dict(error=repr(error)))
        raise
