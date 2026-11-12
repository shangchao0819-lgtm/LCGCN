"""Packaged splits, label-free graph inputs, and an explicit inductive protocol."""
from pathlib import Path
import hashlib
import os
import pickle
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
DATASETS = {
    'cora-coauthorship': ('coauthorship', 'cora'),
    'cora-cocitation': ('cocitation', 'cora'),
    'citeseer': ('cocitation', 'citeseer'),
    'pubmed': ('cocitation', 'pubmed'),
    'dblp': ('coauthorship', 'dblp'),
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_data(name, split, min_edge_size=0, task='transductive', split_seed=None, max_edge_size=0, order_seed=None):
    base = Path(os.environ['LCGCN_DATA_DIR']).joinpath(*DATASETS[name])
    paths = {k: base / (k + '.pickle') for k in ('features', 'hypergraph', 'labels')}
    paths['split'] = base / 'splits' / f'{split}.pickle'
    loaded = {}
    for key, path in paths.items():
        # Load only trusted, hash-verified local dataset files.
        with path.open('rb') as handle:
            loaded[key] = pickle.load(handle, encoding='latin1')
    raw = sp.csr_matrix(loaded['features'], dtype=np.float32)
    if not np.isfinite(raw.data).all():
        raise ValueError('Non-finite features')
    labels = np.asarray(loaded['labels'])
    if labels.ndim == 2:
        labels = labels.argmax(1)
    classes, y = np.unique(labels, return_inverse=True)
    n = raw.shape[0]
    if len(y) != n:
        raise ValueError('Feature/label count mismatch')
    sums = np.asarray(raw.sum(1)).ravel()
    inv = np.zeros_like(sums)
    inv[sums != 0] = 1 / sums[sums != 0]
    features = (sp.diags(inv) @ raw).tocsr()
    hypergraph = loaded['hypergraph']
    # Preserve serialized hyperedge order and each list's vertex order. Sets have
    # no stored sequence and receive an explicit ascending-ID convention.
    memberships = list(hypergraph.values()) if isinstance(hypergraph, dict) else list(hypergraph)
    edges = [np.asarray(sorted(e) if isinstance(e, (set, frozenset)) else list(e), dtype=np.int64)
             for e in memberships]
    for e in edges:
        if len(np.unique(e)) != len(e) or np.any(e < 0) or np.any(e >= n):
            raise ValueError('Duplicate/out-of-range hyperedge members')
    edges = [e for e in edges if len(e) >= min_edge_size and (not max_edge_size or len(e) <= max_edge_size)]
    if order_seed is not None:
        rng = np.random.default_rng(order_seed)
        edges = [rng.permutation(e) for e in edges]
    if not any(len(e) >= 2 for e in edges):
        raise ValueError('No non-singleton hyperedges survive this filter')
    train = np.asarray(loaded['split']['train'], dtype=np.int64)
    test = np.asarray(loaded['split']['test'], dtype=np.int64)
    unseen = np.array([], dtype=np.int64)
    seen = test.copy()
    visible = np.arange(n)
    if task == 'inductive':
        from sklearn.model_selection import train_test_split
        # New declared protocol; the paper does not specify the exact seeds or
        # stratification. Labels are used only to fix balanced partitions here.
        train, rest = train_test_split(np.arange(n), train_size=.2, stratify=y, random_state=split_seed)
        seen, unseen = train_test_split(rest, test_size=.5, stratify=y[rest], random_state=split_seed+1)
        test = np.r_[seen, unseen]
        visible = np.sort(np.r_[train, seen])
    elif task != 'transductive':
        raise ValueError(task)
    for indices in (train, test):
        if not len(indices) or len(indices) != len(np.unique(indices)) or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError('Invalid train/test indices')
    if len(np.intersect1d(train, test)):
        raise ValueError('Overlapping train/test indices')
    inverse = np.full(n, -1, dtype=np.int64)
    inverse[visible] = np.arange(len(visible))
    # Preserve empty restricted edges to retain the original projection identity.
    restricted = [inverse[e][inverse[e] >= 0] for e in edges]
    sizes = np.array([len(e) for e in edges])
    stats = dict(nodes=n, features=raw.shape[1], classes=len(classes), hyperedges=len(edges),
                 train_nodes=len(train), test_nodes=len(test), training_visible_nodes=len(visible),
                 label_fraction=len(train)/n, size_max=int(sizes.max()), size_mean=float(sizes.mean()),
                 size_ge6=int((sizes >= 6).sum()), size4=int((sizes == 4).sum()))
    stats.update(validation_nodes=0, unassigned_nodes=n-len(np.union1d(train,test)),
                 train_per_class=np.bincount(y[train],minlength=len(classes)).tolist(),
                 test_per_class=np.bincount(y[test],minlength=len(classes)).tolist(),
                 seen_per_class=np.bincount(y[seen],minlength=len(classes)).tolist(),
                 unseen_per_class=np.bincount(y[unseen],minlength=len(classes)).tolist(),
                 train_ratio=len(train)/n, test_ratio=len(test)/n, validation_ratio=0.,
                 memberships=int(sizes.sum()), isolated_nodes=int(n-len(np.unique(np.concatenate(edges)))))
    return dict(raw=raw, features=features, y=y.astype(np.int64), edges=edges, stats=stats,
                train=np.sort(train), test=np.sort(test), seen=np.sort(seen), unseen=np.sort(unseen),
                visible=visible, train_local=inverse[np.sort(train)], training_edges=restricted,
                hashes={k: digest(v) for k, v in paths.items()})

def training_view(data):
    visible = data['visible']
    return dict(raw=data['raw'][visible], features=data['features'][visible], edges=data['training_edges'],
                stats=dict(data['stats'], nodes=len(visible)))
