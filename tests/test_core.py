"""Offline tests on a toy belief where question order matters.

Slots A (2 readings) and B (3 readings). Results:
  A=0: B decides the result (three different classes)
  A=1: B does not matter (one class)
So asking A first is wasted when A=1, and asking B first is wasted when A=1 as well; the optimal
order depends on the prior. With a uniform prior over the 6 joint readings, there are 4 classes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ambitopo import (AskAll, Belief, EIGReadings, cluster, Greedy, Hypothesis, MCTS, Natural, NoAsk, Optimal, Reading,
                      PlainMCTS, Setting, ShowLoop, SimulatedUser, TopoMCTS, Slot, Topology, run_episode)
from ambitopo.query import Filter, QuerySpec


def toy():
    topo = Topology([Slot("A", "x", [Reading(0, "a0"), Reading(1, "a1", "align")]),
                     Slot("B", "y", [Reading(0, "b0"), Reading(1, "b1", "align"), Reading(2, "b2", "align")])])
    res = {(0, 0): {1, 2}, (0, 1): {3, 4}, (0, 2): {5, 6}, (1, 0): {7, 8}, (1, 1): {7, 8}, (1, 2): {7, 8}}
    hyps = [Hypothesis(k, frozenset(v)) for k, v in res.items()]
    return Belief(topo, hyps)


def test_classes_and_gain():
    b = toy()
    assert b.n_classes == 4
    assert cluster([frozenset({7, 8}), frozenset({7, 8, 9})], 0.8) == [0, 1]   # Jaccard 0.67
    assert cluster([frozenset({7, 8}), frozenset({7, 8, 9})], 0.6) == [0, 0]
    assert b.gain(b.root, "B") > 0 and b.gain(b.root, "A") > 0
    assert abs(b.acc_stop(b.root) - 3 / 6) < 1e-9


def evaluate(st, policy):
    out = [run_episode(st, policy, SimulatedUser(st.belief, t)) for t in st.belief.root]
    return sum(r.success for r in out) / len(out), sum(r.asks for r in out) / len(out)


def test_policies_order():
    b = toy()
    st = Setting(b, askable=["A", "B"], lam=1e-3)
    acc = {p.name: evaluate(st, p) for p in (NoAsk(), AskAll(), Natural(), Greedy(), Optimal(), MCTS(n_sim=300))}
    for name in ("ask_all", "natural", "greedy_ec2", "optimal", "mcts"):
        assert acc[name][0] == 1.0, (name, acc[name])
    assert acc["no_ask"][0] < 1.0
    assert acc["optimal"][1] <= acc["greedy_ec2"][1] + 1e-9 <= acc["ask_all"][1] + 1e-9
    assert abs(acc["mcts"][1] - acc["optimal"][1]) < 0.34, acc


def test_mcts_variants():
    b = toy()
    st = Setting(b, askable=["A", "B"], lam=1e-3)
    opt = evaluate(st, Optimal())[1]
    for p in (TopoMCTS(n_sim=300), PlainMCTS(n_sim=300), MCTS(n_sim=300, rave=True, transpose=False)):
        acc, asks = evaluate(st, p)
        assert acc == 1.0 and asks <= opt + 0.5 + 1e-9, (p.name, acc, asks)


def test_eig_readings_wastes():
    # in toy, when A=1 the B question cannot change the result; reading-level EIG still asks it
    b = toy()
    st = Setting(b, askable=["A", "B"], lam=1e-3)
    acc, asks = evaluate(st, EIGReadings())
    assert acc == 1.0 and asks >= evaluate(st, Optimal())[1]


def test_budget_one():
    b = toy()
    st = Setting(b, askable=["A", "B"], budget=1)
    opt = Optimal()
    v = opt.value(st)
    acc_opt, _ = evaluate(st, opt)
    assert abs(acc_opt - v) < 1e-9
    assert acc_opt >= evaluate(st, Greedy())[0] - 1e-9
    assert acc_opt >= evaluate(st, Natural())[0] - 1e-9


def test_noise_keeps_truth():
    b = toy()
    st = Setting(b, askable=["A", "B"], noise=0.5, lam=1e-3)
    for t in b.root:
        r = run_episode(st, Greedy(), SimulatedUser(b, t, noise=0.5, seed=t))
        assert r.asks <= st.asks_left(0)


def test_show_channel():
    # only B is askable-by-taxonomy-elsewhere; here nothing is askable, so show is the only channel
    b = toy()
    st = Setting(b, askable=[], budget=3, show_cost=1e-3)
    for p in (ShowLoop(), Greedy(), Optimal(), MCTS(n_sim=300)):
        acc, _ = evaluate(st, p)
        assert acc == 1.0, (p.name, acc)          # 4 classes, 3 shows: rejecting 3 leaves the truth
    st1 = Setting(b, askable=[], budget=1, show_cost=1e-3)
    assert abs(evaluate(st1, Optimal())[0] - (0.5 + 0.5 * (1 / 3))) < 1e-9
    st_mix = Setting(b, askable=["A"], budget=2, show_cost=1e-3, lam=1e-3)
    assert evaluate(st_mix, Optimal())[0] >= evaluate(st_mix, ShowLoop())[0]


def test_query_render_and_match():
    q = ('[out:json][timeout:25];{{geocodeArea:"Udine"}}->.searchArea;'
         '(nwr["building"="school"]["amenity"!~".*"](area.searchArea););out meta;')
    spec = QuerySpec.parse(q, main_tag="building=school")
    assert spec.area_name == "Udine" and spec.main == 0 and len(spec.secondary) == 1
    assert '["building"="school"]["amenity"!~".*"]' in spec.render("3600000001")
    assert '["amenity"' not in spec.render("3600000001", drop=0)
    f = Filter.parse('["amenity"!~".*"]')
    assert f.match({"amenity": ""}) and not f.match({"amenity": "school"})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
