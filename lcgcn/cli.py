"""Train the frozen LCGCN main-table recipe on the supplied preprocessed datasets."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from .reporting import DATASETS, MODELS, make_report, format_result, format_summary

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
ROOT = PROJECT if (PROJECT / 'main.py').is_file() else Path.cwd()

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['lcgcn', 'native-lcgcn', 'fast-lcgcn'], default='lcgcn')
    p.add_argument('--dataset', choices=['dblp', 'cora-coauthorship', 'pubmed', 'citeseer', 'cora-cocitation'], default='cora-coauthorship')
    p.add_argument('--split', type=int, choices=range(1, 11), default=1)
    p.add_argument('--seed', type=int, choices=[1, 8, 17], default=1)
    p.add_argument('--all', action='store_true', help='450 real fits, 200 epochs each')
    p.add_argument('--smoke', action='store_true', help='3 real epochs; not paper reproduction')
    p.add_argument('--list', action='store_true', help='Show selected frozen configurations only')
    p.add_argument('--check', action='store_true', help='Verify data, environment and all 50 splits without training')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    a = p.parse_args()
    configs = json.loads((PACKAGE / 'configs/main_450.json').read_text('utf-8'))
    if not a.all:
        configs = [c for c in configs if (c['model'], c['dataset'], c['split'], c['initialization_seed']) == (a.model, a.dataset, a.split, a.seed)]
    if a.list:
        print(json.dumps(configs, indent=2)); return
    data_dir = a.data_dir.resolve()
    if not (data_dir / 'manifest.json').is_file():
        p.error('Data missing. This source repository excludes third-party datasets. Supply the authorized data directory with --data-dir (including manifest.json); see docs/DATA.md.')
    manifest = json.loads((data_dir / 'manifest.json').read_text('utf-8'))
    for rel, expected in manifest.items():
        f = data_dir / rel
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != expected:
            p.error(f'Data missing or changed: {f}. Restore the authorized input files matching manifest.json; see docs/DATA.md.')
    os.environ['LCGCN_DATA_DIR'] = str(data_dir)
    try:
        import numpy, scipy, sklearn, torch
    except ImportError as exc:
        p.error(f'{exc}. Select the Python 3.10 environment and follow README installation commands.')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable in this interpreter. Use --device cpu or install the CUDA PyTorch wheel per README.')
    # Exercise the actual sparse operator dependency on the chosen device.
    v = torch.ones(1, device=device, requires_grad=True)
    s = torch.sparse_coo_tensor(torch.tensor([[0], [0]], device=device), v, (1, 1)).coalesce()
    torch.sparse.mm(s, torch.ones((1, 1), device=device)).sum().backward()
    print(f'Environment: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | '
          f'NumPy {numpy.__version__} | SciPy {scipy.__version__} | sklearn {sklearn.__version__} | {device}', flush=True)
    if a.check:
        from .data_loader import DATASETS as DATASET_PATHS, load_data
        rows = [dict(dataset=n, split=i, **load_data(n, i)['stats']) for n in DATASET_PATHS for i in range(1, 11)]
        print(f'Data checks PASS: {len(manifest)} file hashes and {len(rows)} splits.', flush=True)
        print('Dataset       Nodes    Features    Classes    Hyperedges    Splits')
        for name in DATASETS:
            group = [r for r in rows if r['dataset'] == name]
            first = group[0]
            print(f"{DATASETS[name]:<14}{first['nodes']:<9}{first['features']:<12}"
                  f"{first['classes']:<11}{first['hyperedges']:<14}{len(group)}")
        return
    from .train import run
    from .reproduce import verify
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    out = a.output_dir.resolve() / (('smoke-' if a.smoke else 'train-') + stamp)
    out.mkdir(parents=True)
    os.environ['LCGCN_SNAPSHOT_DIR'] = str(out / 'source_snapshots')
    reports = []
    print(f'\nTraining plan: {len(configs)} fits, {3 if a.smoke else 200} epochs each.'
          + (' [SMOKE: execution check only]' if a.smoke else ' [Actual training; final checkpoint]'), flush=True)
    for index, raw in enumerate(configs, 1):
        cfg = dict(raw, device=device)
        if a.smoke: cfg.update(smoke=True, epochs=3)
        print(f"\n[{index}/{len(configs)}] Start: {MODELS[cfg['model']]} | "
              f"{DATASETS[cfg['dataset']]} | split={cfg['split']} | seed={cfg['initialization_seed']}", flush=True)
        result = run(cfg, output_root=out)
        try:
            reload = verify(result['folder'])
        except AssertionError as exc:
            reload = dict(passed=False, error=str(exc), note='Dynamic graph rebuilding can differ numerically. Training artifacts retained; investigate, do not select a better rerun.')
        reports.append(make_report(result, reload))
        (out / 'summary.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
        print(format_result(reports[-1], index, len(configs)), flush=True)
        summary = format_summary(reports, out)
        (out / 'summary.txt').write_text(summary + '\n', encoding='utf-8')
    print(format_summary(reports, out), flush=True)
    if any(not r['reload']['passed'] for r in reports): return 2
    return 0

if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
