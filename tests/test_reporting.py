"""The statistical unit is a split mean, not an individual fit."""
from statistics import stdev
import unittest
from lcgcn.reporting import paper_statistics


class PaperStatistics(unittest.TestCase):
    def rows(self):
        return [dict(model='lcgcn', dataset='test-fixture', device='cpu',
                     split=s, seed=seed, smoke=False, epochs=200,
                     metrics={'test_accuracy': 50 + s + offset}, reload={'passed': True})
                for s in range(1, 11) for seed, offset in [(1, -3), (8, 0), (17, 3)]]

    def test_seed_variation_does_not_inflate_split_sd(self):
        result = paper_statistics(self.rows())
        self.assertAlmostEqual(result['mean'], 55.5)
        self.assertAlmostEqual(result['sd'], stdev(range(51, 61)))
        self.assertEqual(result['split_means'], list(range(51, 61)))

    def test_reject_incomplete_duplicate_smoke_or_mixed_devices(self):
        rows = self.rows()
        self.assertIsNone(paper_statistics(rows[:-1]))
        self.assertIsNone(paper_statistics(rows[:-1] + [rows[0]]))
        for changes in [dict(smoke=True), dict(epochs=3), dict(device='cuda')]:
            altered = rows[:-1] + [dict(rows[-1], **changes)]
            self.assertIsNone(paper_statistics(altered))


if __name__ == '__main__':
    unittest.main()
