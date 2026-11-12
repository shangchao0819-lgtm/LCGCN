"""Small numerical contracts; no bundled dataset or GPU is required."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import lcgcn
from lcgcn.graph_builder import FactorBuilder
from lcgcn.models import LCGCN
from lcgcn.train import defaults


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.x = np.random.default_rng(1).random((7, 4), dtype=np.float32)
        self.edges = [np.array([0, 1, 2, 3, 4]), np.array([2, 3, 5, 6])]

    def test_local_mass_overlap_and_normalization(self):
        builder = FactorBuilder(7, self.edges, 4, 1, order='native')
        builder.build(self.x)
        raw = builder.last_raw.toarray()
        self.assertAlmostEqual(float(raw.sum()) / 2, 2, places=6)
        self.assertAlmostEqual(float(raw[2, 3]), 1 / 5 + 1 / 6, places=6)
        b = raw + np.eye(7)
        d = 1 / np.sqrt(b.sum(1))
        np.testing.assert_allclose(builder.last_operator.toarray(), d[:, None] * b * d[None, :], rtol=1e-6)

    def test_dynamic_graph_uses_post_dropout_hidden(self):
        model = LCGCN(self.x, self.edges, 3, defaults(hidden=4), 'cpu')
        x = torch.tensor(self.x)
        captured = []
        builder = model.builders[1]
        original = builder.build
        def capture(h):
            captured.append(h.detach().clone())
            return original(h)
        model.train()
        with patch.object(builder, 'build', side_effect=capture), patch('lcgcn.models.F.dropout', side_effect=lambda h, **kwargs: h * 2):
            logits = model(x)
        expected = F.relu(torch.sparse.mm(model.first_operator, x @ model.weights[0])) * 2
        torch.testing.assert_close(captured[0], expected)
        logits.sum().backward()
        self.assertTrue(all(w.grad is not None and torch.isfinite(w.grad).all() for w in model.weights))
        self.assertEqual(model.builders[0].calls, 1)
        model.eval()
        calls = builder.calls
        torch.testing.assert_close(model(x), model(x))
        self.assertEqual(builder.calls, calls + 2)

    def test_fixed_variants_cache_graph(self):
        for variant in ('native-lcgcn', 'fast-lcgcn'):
            model = LCGCN(self.x, self.edges, 3, defaults(model=variant, hidden=4), 'cpu')
            model.eval()
            logits = model(torch.tensor(self.x))
            self.assertEqual(len(model.builders), 1)
            self.assertEqual(model.builders[0].calls, 1)
            self.assertTrue(torch.equal(logits.argmax(1), logits.softmax(1).argmax(1)))

    def test_packaged_recipe_and_seed_mapping(self):
        configs = json.loads((Path(lcgcn.__file__).parent / 'configs/main_450.json').read_text('utf8'))
        keys = {(c['model'], c['dataset'], c['split'], c['initialization_seed']) for c in configs}
        self.assertEqual(len(keys), 450)
        for c in configs:
            actual = defaults(**c)
            self.assertEqual(actual['initialization_seed'], c['initialization_seed'])
            self.assertEqual(c['seed'], {1: 9, 8: 1, 17: 23}[c['initialization_seed']])
            self.assertEqual(c['epochs'], 200)
            self.assertEqual(c['reporting_rule'], 'budget-final')


if __name__ == '__main__':
    unittest.main()
