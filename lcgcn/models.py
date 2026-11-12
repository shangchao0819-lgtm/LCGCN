"""Two-layer LCGCN classification models with no output ReLU or bias."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .graph_builder import FactorBuilder

VARIANTS = ('lcgcn', 'native-lcgcn', 'fast-lcgcn')
CONTROLS = ('control-clique', 'control-cycle', 'control-double-star')

class LCGCN(nn.Module):
    def __init__(self, features, edges, classes, config, device):
        super().__init__()
        self.config = dict(config)
        self.variant = config['model']
        if self.variant not in VARIANTS + CONTROLS:
            raise ValueError(self.variant)
        n, p = features.shape
        hidden = config['hidden']
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(p, hidden)), nn.Parameter(torch.empty(hidden, classes))])
        for w in self.weights:
            nn.init.uniform_(w, -1/math.sqrt(w.shape[1]), 1/math.sqrt(w.shape[1]))
        self.to(device)
        kind = self.variant.replace('control-', '') if self.variant in CONTROLS else 'circulant'
        order = 'native' if self.variant in ('native-lcgcn', 'control-clique') else 'projected'
        count = 2 if self.variant == 'lcgcn' else 1
        self.builders = [FactorBuilder(n, edges, p if i == 0 else hidden, config['graph_seed'] + i*104729,
                                      order=order, degree=config['degree'], kind=kind, device=device) for i in range(count)]
        # H^(0)=X is fixed, hence its graph may be cached without changing Eq.(5).
        self.first_operator = self.builders[0].build(features)

    def forward(self, X):
        H = torch.sparse.mm(self.first_operator, X @ self.weights[0])
        H = F.dropout(F.relu(H), p=self.config['dropout'], training=self.training)
        second = self.builders[1].build(H) if self.variant == 'lcgcn' else self.first_operator
        return torch.sparse.mm(second, H @ self.weights[1])

    def export_graphs(self, folder):
        for i, builder in enumerate(self.builders):
            builder.export(folder, f'layer{i+1}')
        return [b.metadata() for b in self.builders]
