"""Readable console summaries of actual training results."""
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

MODELS = {'lcgcn': 'LCGCN', 'native-lcgcn': 'NativeLCGCN', 'fast-lcgcn': 'FastLCGCN'}
DATASETS = {'dblp': 'DBLP', 'cora-coauthorship': 'Cora-CA', 'cora-cocitation': 'Cora-CC',
            'citeseer': 'Citeseer', 'pubmed': 'Pubmed'}


def paper_statistics(rows):
    """Main-table unit: ten split means, each averaging seeds 1, 8, 17."""
    expected = {(split, seed) for split in range(1, 11) for seed in (1, 8, 17)}
    keys = [(r['split'], r['seed']) for r in rows]
    if (len(rows) != 30 or set(keys) != expected
            or len({(r['model'], r['dataset'], r['device']) for r in rows}) != 1
            or any(r['smoke'] or r['epochs'] != 200 for r in rows)):
        return None
    splits = [mean(r['metrics']['test_accuracy'] for r in rows if r['split'] == s)
              for s in range(1, 11)]
    return dict(mean=mean(splits), sd=stdev(splits), split_means=splits,
                reload_passed=all(r['reload']['passed'] for r in rows))


def make_report(result, reload):
    cfg = result['fit_config']
    return dict(folder=result['folder'], metrics=result['metrics'], reload=reload,
                model=cfg['model'], dataset=cfg['dataset'], split=cfg['split'],
                seed=cfg['initialization_seed'], epochs=cfg['epochs'],
                smoke=cfg['smoke'], device=cfg['device'], timing=result['timing'])


def format_result(report, index, total):
    r = report
    status = 'PASS' if r['reload']['passed'] else 'FAIL (training artifacts retained; investigate)'
    lines = [f"\n{'=' * 68}", f"[{index}/{total}] Training result",
             f"Model              : {MODELS[r['model']]}",
             f"Dataset            : {DATASETS[r['dataset']]}",
             f"Split              : {r['split']}",
             f"Initialization seed: {r['seed']}",
             f"Epochs             : {r['epochs']} (final checkpoint)",
             f"Device             : {r['device']}",
             f"Test accuracy      : {r['metrics']['test_accuracy']:.4f}%",
             f"Test nodes         : {r['metrics']['test_n']}",
             f"Training time      : {r['timing']['train_seconds']:.2f} s",
             f"Fit time           : {r['timing']['total_seconds']:.2f} s (excluding reload verification)",
             f"Reload verification: {status}",
             f"Saved to           : {Path(r['folder'])}"]
    if r['smoke']:
        lines.append('SMOKE: three epochs only; this is not a 200-epoch paper result.')
    if not r['reload']['passed']:
        lines.append('Verification details: ' + r['reload'].get('error', 'See summary.json'))
    return '\n'.join(lines)


def format_summary(reports, out):
    groups = defaultdict(list)
    for r in reports:
        groups[(r['model'], r['dataset'], r['epochs'], r['smoke'])].append(r)
    lines = [f"\n{'=' * 90}", 'Run summary',
             f"Completed fits: {len(reports)}; reload checks passed: {sum(r['reload']['passed'] for r in reports)}",
             'Model         Dataset      Epochs    N    Test mean (%)    Sample SD (pp)    Train sum (s)',
             '-' * 90]
    for (model, dataset, epochs, smoke), rows in sorted(groups.items()):
        scores = [r['metrics']['test_accuracy'] for r in rows]
        sd = f'{stdev(scores):.4f}' if len(scores) > 1 else 'N/A'
        lines.append(f'{MODELS[model]:<14}{DATASETS[dataset]:<13}{epochs:<10}{len(rows):<5}'
                     f'{mean(scores):<17.4f}{sd:<18}{sum(r["timing"]["train_seconds"] for r in rows):.2f}'
                     + ('  [SMOKE]' if smoke else ''))
    lines.extend(['The table above describes individual fits; its SD is not the paper-table SD.',
                  'N is the number of completed fits. SD is the sample SD of fit accuracies (percentage points, ddof=1).',
                  'SD is N/A for one fit. Completed fits with failed reload checks remain included; scores are not selected.',
                  'Only this invocation is summarized. Historical results are not loaded; partial/smoke runs are not full 450-fit reproduction.',
                  f'Output directory: {out}', f'Text summary: {Path(out) / "summary.txt"}',
                  f'Detailed results: {Path(out) / "summary.json"}'])
    lines.extend(['', 'Paper statistics: average the three seeds within each split, then compute mean +/- sample SD across ten split means.'])
    for (model, dataset, epochs, smoke), rows in sorted(groups.items()):
        stats = paper_statistics(rows)
        label = f'{MODELS[model]} / {DATASETS[dataset]} / {epochs} epochs'
        if stats is None:
            lines.append(f'{label}: N/A (requires 10 splits x 3 seeds, 200 epochs, one device, no duplicates; currently {len(rows)} fits)')
        else:
            lines.append(f"{label}: {stats['mean']:.4f} ± {stats['sd']:.4f}%"
                         + (' [all reload checks passed]' if stats['reload_passed'] else ' [reload failure present; investigate]'))
    return '\n'.join(lines)
