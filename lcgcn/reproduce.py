"""Reload a saved final checkpoint and verify its predictions and graph hashes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data_loader import load_data
from .models import LCGCN
from .train import build_model, isolated_rng, seed_all, source_snapshot, save_json, verify_artifacts

def verify(folder):
    folder = Path(folder)
    result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    verify_artifacts(folder,result)
    cfg = result.get('fit_config', result['config'])
    current_hash, _ = source_snapshot()
    if current_hash != result['source_sha256']:
        raise ValueError('Code differs from the saved run; use its source_snapshots archive')
    torch.set_num_threads(cfg['threads'])
    seed_all(cfg['initialization_seed'])
    data = load_data(cfg['dataset'], cfg['split'], cfg['min_edge_size'], cfg['task'], cfg['split_seed'], cfg['max_edge_size'], cfg['order_seed'])
    if data['hashes'] != result['data_sha256']:
        raise ValueError('Data files differ from saved run')
    device = torch.device(cfg['device'])
    # Both transductive and final inductive evaluation operate on the full graph.
    model, _ = build_model(data, cfg, device)
    saved = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
    if set(saved['parameters']) != set(dict(model.named_parameters())):
        raise ValueError('Checkpoint parameters differ')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(saved['parameters'][name].to(device))
    model.eval()
    X = torch.tensor(data['features'].toarray(), dtype=torch.float32, device=device)
    with isolated_rng(cfg['eval_seed']), torch.no_grad():
        actual = model(X).cpu().numpy()
    with np.load(folder/'predictions.npz') as archive:
        expected = archive['logits']
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    graph_equal = None
    if isinstance(model, LCGCN):
        graph_equal = all(b.metadata()['operator_sha256'] == old['operator_sha256']
                          for b, old in zip(model.builders, result['final_evaluation_graphs']))
        if not graph_equal:
            raise AssertionError('Rebuilt graph differs from saved final graph')
    report = dict(passed=True, max_absolute_logit_difference=float(np.max(np.abs(actual-expected))),
                  identical_predictions=bool(np.array_equal(actual.argmax(1), expected.argmax(1))),
                  graph_hashes_match=graph_equal, folder=str(folder))
    save_json(folder/'reproduction.json', report)
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    print(json.dumps(verify(p.parse_args().folder), ensure_ascii=False, indent=2))
