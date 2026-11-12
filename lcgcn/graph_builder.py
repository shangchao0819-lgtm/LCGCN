"""Local circulant graph construction with fixed projections and unit-mass factors."""
from functools import lru_cache
import hashlib
import time
import numpy as np
import scipy.sparse as sp
import torch

DEGREES = ('half', '2', 'third', 'two-thirds', 'three-quarters', 'complete')

@lru_cache(maxsize=2048)
def generators(s, degree='half', kind='circulant'):
    if s < 2:
        return ()
    if kind == 'clique' or (kind == 'circulant' and s <= 4):
        return tuple(range(1, s))
    if kind == 'cycle':
        return tuple(sorted({1, s - 1}))
    if kind != 'circulant':
        raise ValueError(kind)
    targets = {'half': s // 2, 'third': s // 3, 'two-thirds': 2*s//3,
               'three-quarters': 3*s//4, 'complete': s-1, '2': 2}
    d = min(s-1, max(2, targets[degree]))
    if s % 2 and d % 2:
        d += 1
    positive = list(range(1, d//2 + 1))
    result = {a % s for a in positive} | {(-a) % s for a in positive}
    if d % 2:
        result.add(s//2)
    return tuple(sorted(result))

@lru_cache(maxsize=2048)
def local_pairs(s, degree='half', kind='circulant'):
    if s < 2:
        return np.empty((0, 2), dtype=np.int64)
    if kind == 'double-star':
        pairs = {(0, s-1)} | {(0, i) for i in range(1, s-1)} | {(i, s-1) for i in range(1, s-1)}
    else:
        pairs = {(min(i, (i+a)%s), max(i, (i+a)%s)) for i in range(s) for a in generators(s, degree, kind)}
    result = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    result.flags.writeable = False
    return result

def normalise(A):
    B = (A + sp.eye(A.shape[0], format='csr', dtype=np.float32)).tocsr()
    inverse = np.asarray(B.sum(1)).ravel() ** -.5
    D = sp.diags(inverse)
    return (D @ B @ D).astype(np.float32).tocsr()

def sparse_tensor(A, device):
    coo = A.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, A.shape).coalesce().to(device)

def matrix_hash(A):
    return hashlib.sha256(A.indptr.tobytes() + A.indices.tobytes() + A.data.tobytes()).hexdigest()

class FactorBuilder:
    def __init__(self, n, edges, width, seed, order='projected', degree='half', kind='circulant', device='cpu'):
        self.n, self.width, self.seed = n, width, int(seed)
        self.edges = [np.asarray(e, dtype=np.int64).copy() for e in edges]
        for e in self.edges:
            if len(set(e.tolist())) != len(e) or np.any(e < 0) or np.any(e >= n):
                raise ValueError('Invalid hyperedge membership')
        self.order, self.degree, self.kind, self.device = order, degree, kind, device
        if order not in ('projected', 'native'):
            raise ValueError(order)
        # Implementation convention: one independent PCG64 Uniform[0,1) float32
        # vector per stored hyperedge, sampled once and retained for this layer.
        self.projections = (np.random.default_rng(seed).random((len(edges), width), dtype=np.float32)
                            if order == 'projected' else None)
        self.projection_hash = hashlib.sha256(self.projections.tobytes()).hexdigest() if self.projections is not None else None
        self.calls, self.seconds = 0, 0.
        self.last_raw = self.last_operator = None
        self.last_orders = None

    def build(self, features=None):
        started = time.perf_counter()
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if self.order == 'projected' and (features is None or features.shape != (self.n, self.width)):
            raise ValueError('Projection must use the layer INPUT width and all current nodes')
        if self.order == 'projected' and not np.isfinite(features).all():
            raise ValueError('Non-finite layer input')
        rows, cols, values, orders = [], [], [], []
        for index, edge in enumerate(self.edges):
            s = len(edge)
            if self.order == 'projected':
                scores = features[edge] @ self.projections[index]
                members = edge[np.lexsort((edge, scores))]
            else:
                members = edge
            orders.append(members.copy())
            if s < 2:
                continue
            pairs = local_pairs(s, self.degree, self.kind)
            u, v = members[pairs[:, 0]], members[pairs[:, 1]]
            w = np.full(len(pairs), 1. / len(pairs), dtype=np.float32)
            rows.extend([u, v]); cols.extend([v, u]); values.extend([w, w])
        if rows:
            A = sp.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(self.n, self.n), dtype=np.float32).tocsr()
            A.sum_duplicates(); A.sort_indices()
        else:
            A = sp.csr_matrix((self.n, self.n), dtype=np.float32)
        self.last_raw, self.last_operator, self.last_orders = A, normalise(A), orders
        tensor = sparse_tensor(self.last_operator, self.device)
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return tensor

    def metadata(self):
        return dict(kind=self.kind, order=self.order, degree=self.degree, seed=self.seed,
                    projection_distribution='PCG64 Uniform[0,1), float32, independent per hyperedge and layer; fixed for training and evaluation',
                    projection_sha256=self.projection_hash, calls=self.calls, seconds=self.seconds,
                    raw_adjacency_sha256=matrix_hash(self.last_raw), operator_sha256=matrix_hash(self.last_operator),
                    undirected_edges=self.last_raw.nnz//2, nodes=self.n)

    def export(self, folder, label):
        from pathlib import Path
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        sp.save_npz(folder / f'{label}_adjacency.npz', self.last_raw)
        sp.save_npz(folder / f'{label}_operator.npz', self.last_operator)
        lengths = np.asarray([len(e) for e in self.last_orders], dtype=np.int64)
        np.savez_compressed(folder / f'{label}_orders.npz', offsets=np.r_[0, np.cumsum(lengths)],
                            members=np.concatenate(self.last_orders) if len(lengths) else np.array([], dtype=np.int64))

