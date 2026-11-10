"""Mathematical invariants, erasure model, budget trap and search regression."""
import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo import Belief, Hypothesis, Reading, Setting, Slot, Topology
from ambitopo.planners import MCTS, SUBMIT
from ambitopo.shapley import CoalitionGame, SearchMCTS, ShapleyMCTS


def make_belief(fn, n=2, weights=None):
    slots = [Slot(chr(65 + j), "toy", [Reading(0, "0"), Reading(1, "1")]) for j in range(n)]
    topo = Topology(slots)
    return Belief(topo, [Hypothesis(a, frozenset({fn(a)})) for a in itertools.product((0, 1), repeat=n)], prior=weights, threshold=1.0)


class Tests(unittest.TestCase):
    def test_xor_synergy(self):
        b = make_belief(lambda a: a[0] ^ a[1])
        g = CoalitionGame(b, b.root, b.slots)
        self.assertEqual(g.raw, [0.0, 0.0, 0.0, 0.5])
        self.assertEqual(g.attribution(), {"A": 0.25, "B": 0.25})
        self.assertEqual(g.attribution(1), {"A": 0.0, "B": 0.0})
        self.assertEqual(g.best(1, 0.05), ((), 0.0))
        self.assertAlmostEqual(g.best(2, 0.05)[1], 0.4)

    def test_dummy_and_efficiency(self):
        b = make_belief(lambda a: a[0], n=3)
        g = CoalitionGame(b, b.root, b.slots, 0.2)
        phi = g.attribution()
        self.assertAlmostEqual(phi["A"], 0.4)
        self.assertAlmostEqual(phi["B"], 0.0)
        self.assertAlmostEqual(phi["C"], 0.0)
        self.assertAlmostEqual(sum(phi.values()), g.gain[-1])

    def test_noise_exact_enumeration(self):
        b = make_belief(lambda a: (a[0] and a[1]) ^ a[2], n=3, weights=[1,2,3,4,5,6,7,8])
        g = CoalitionGame(b, b.root, b.slots, 0.3)
        for mask in range(8):
            expected = 0.0
            for sub in range(8):
                if sub & mask == sub:
                    expected += 0.7 ** sub.bit_count() * 0.3 ** (mask.bit_count() - sub.bit_count()) * g.raw[sub]
            self.assertAlmostEqual(g.gain[mask], expected)
        self.assertAlmostEqual(sum(g.attribution().values()), g.gain[-1])
        for j, slot in enumerate(b.slots):
            self.assertAlmostEqual(g.attribution(1)[slot], g.gain[1 << j])

    def test_conditioned_belief_and_duplicate_results(self):
        b = make_belief(lambda a: a[0] ^ a[1])
        alive = b.observe(b.root, "A", 0)
        g = CoalitionGame(b, alive, ["B"])
        self.assertAlmostEqual(g.attribution()["B"], 0.5)
        hyps = b.hyps + b.hyps
        dup = Belief(b.topo, hyps, threshold=1.0)
        self.assertEqual(CoalitionGame(dup, dup.root, dup.slots).attribution(), CoalitionGame(b, b.root, b.slots).attribution())

    def test_budget_limited_computation_matches_full(self):
        b = make_belief(lambda a: (a[0] and a[1]) ^ a[2], n=3, weights=[1,2,3,4,5,6,7,8])
        full = CoalitionGame(b, b.root, b.slots, 0.2)
        for k in (1, 2, 3):
            small = CoalitionGame(b, b.root, b.slots, 0.2, max_size=k)
            self.assertEqual(small.attribution(k), full.attribution(k))
            self.assertEqual(small.best(k, 0.05), full.best(k, 0.05))
            self.assertEqual(small.member_scores(k, 0.05), full.member_scores(k, 0.05))

    def test_k1_stops_but_k2_escapes(self):
        b = make_belief(lambda a: a[0] ^ a[1])
        for guide in ("shapley", "coalition"):
            p = ShapleyMCTS(n_sim=300, seed=1, guide=guide, guided_rollout=True)
            st = Setting(b, b.slots, budget=1, lam=0.05)
            self.assertEqual(p.act(st, b.root, 0), SUBMIT)
            st.budget = 2
            self.assertEqual(p.act(st, b.root, 0).kind, "ask")

    def test_original_search_equivalence(self):
        b = make_belief(lambda a: a[0] and a[1])
        st = Setting(b, b.slots, budget=2, lam=0.05, noise=0.2)
        for seed in range(5):
            kw = dict(n_sim=30, seed=seed, prior="lookahead", rollout="order")
            self.assertEqual(MCTS(**kw).act(st, b.root, 0), SearchMCTS(**kw).act(st, b.root, 0))

    def test_time_and_noise_edges(self):
        b = make_belief(lambda a: a[0] ^ a[1])
        self.assertEqual(CoalitionGame(b, b.root, b.slots, 1.0).attribution(), {"A": 0.0, "B": 0.0})
        st = Setting(b, b.slots, budget=2, lam=0.05)
        p = ShapleyMCTS(n_sim=100000, time_limit_ms=1)
        p.act(st, b.root, 0)
        self.assertGreaterEqual(p.decisions[-1]["simulations"], 1)
        self.assertGreater(p.decisions[-1]["ms"], 0)
        self.assertGreaterEqual(p.decisions[-1]["overshoot_ms"], 0)


if __name__ == "__main__":
    unittest.main()
