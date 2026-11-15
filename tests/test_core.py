import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from paper_cutting import solve
from paper_cutting.core import summarize


class CuttingTests(unittest.TestCase):
    def test_exact_fit(self):
        r = solve(100, {60: 3, 40: 3})
        self.assertEqual((r['n_rolls'], r['n_patterns'], r['trim_loss']), (3, 1, 0))

    def test_single_width(self):
        r = solve(100, {30: 7})
        self.assertEqual((r['n_rolls'], r['trim_loss']), (3, 90))

    def test_bad_inputs(self):
        for W, d in [(0, {1: 1}), (100, {}), (100, {101: 2}), (100, {0: 2}),
                     (100, {20: -1}), (100, {20: 1.5}), (100, {20: True}),
                     (100, {20: 0}), (True, {1: 1}), (100, {1: 100001})]:
            with self.subTest(W=W, d=d), self.assertRaises(ValueError):
                solve(W, d)

    def test_feasibility_guard(self):
        for bins in [[[60, 60]], [[40]], [[20]], [[]]]:
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                summarize(100, {60: 1, 40: 1}, bins, 'invalid', 0)

    def test_surplus_not_trim(self):
        r = summarize(100, {40: 1}, [[40, 40]], 'test', 0)
        self.assertEqual(r['trim_loss'], 20)
        self.assertEqual(r['overproduction'], {'40': 1})

    def test_random_feasibility_and_conservation(self):
        rng = random.Random(42)
        for _ in range(100):
            d = {w: rng.randint(1, 10) for w in rng.sample(range(1, 101), 8)}
            r = solve(100, d)
            self.assertGreaterEqual(r['n_rolls'], r['lower_bound'])
            self.assertEqual(r['trim_loss'] + sum(w*q for w,q in d.items()), 100*r['n_rolls'])
            self.assertTrue(all(q == 0 for q in r['overproduction'].values()))

    def test_tiny_exact_oracle(self):
        # Independent exhaustive bin-packing oracle on small multisets.
        def optimum(items, bins=()):
            if not items:
                return len(bins)
            w, *rest = items
            choices = [optimum(rest, bins + (w,))]
            for i, used in enumerate(bins):
                if used + w <= 10:
                    choices.append(optimum(rest, bins[:i] + (used+w,) + bins[i+1:]))
            return min(choices)
        for items in itertools.combinations_with_replacement([2, 4, 6, 8], 5):
            demands = {w: items.count(w) for w in set(items)}
            r = solve(10, demands)
            opt = optimum(list(items))
            self.assertLessEqual(r['lower_bound'], opt)
            self.assertGreaterEqual(r['n_rolls'], opt)

    def test_cli(self):
        root = Path(__file__).resolve().parents[1]
        p = subprocess.run([sys.executable, '-m', 'paper_cutting', 'examples/synthetic.json'],
                           cwd=root, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(p.stdout)['method'], 'ffd')


if __name__ == '__main__':
    unittest.main()
