"""Checks for the topology theory (RESULTS.md, "Theory checks").

check 1 (forest closing structure): on each AmbiOverpass-M item, the closing graph at the root has an edge
u -> v when some answer to u (with positive mass) makes v stop mattering. Items are grouped as
no closing edge / forest (in-degree <= 1, acyclic) / other. The topological-order policy asks only slots
that no other undecided, still-mattering slot can close (ties: importance), until the class is certain.
Compared with greedy and the exact optimum: asks to certainty and success at one question (noise 0),
exact expectations over the truths.

check 2 (state abstraction): states reachable by answering slot questions, counted by alive set (current
DAG) and by canonical form (Belief.canon); then TopoMCTS with and without the canonical keying at
10 / 30 simulations on M (realistic utility) and on the gate family.

  python scripts/theory_checks.py 1|2
"""
import hashlib
import json
import os
import random
import statistics
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ambitopo import Greedy, Optimal, Setting, SimulatedUser, TopoMCTS, run_episode
from ambitopo.planners import SUBMIT, Action, Policy
from ambitopo.data import build, load_rows, to_sample
from ambitopo.overpass import Executor
from relation import fam_gate, matters

BENCH = {}
for l in open("out/ambioverpass_m.jsonl"):
    it = json.loads(l)
    if not it["flags"]:
        BENCH[it["sample_id"]] = tuple(v if v != "dist" else "area" for v in it["weakened"])


def closing_graph(b, S, slots):
    amb = [v for v in slots if matters(b, S, v)]
    edges = set()
    for u in amb:
        for h in b.split(S, u).values():
            for v in amb:
                if v != u and (u, v) not in edges and not matters(b, h, v):
                    edges.add((u, v))
    return amb, edges


def shape(amb, edges):
    if not edges:
        return "no closing"
    indeg = {v: sum(1 for u, w in edges if w == v) for v in amb}
    if max(indeg.values()) > 1:
        return "other"
    # in-degree <= 1: acyclic iff following parents never loops
    parent = {w: u for u, w in edges}
    for v in amb:
        seen, x = set(), v
        while x in parent:
            if x in seen:
                return "other"
            seen.add(x)
            x = parent[x]
    return "forest"


class TopoOrder(Policy):
    """Ask a still-mattering slot that no other undecided, still-mattering slot can close."""
    name = "topo_order"

    def act(self, st, alive, used):
        b = st.belief
        if b.certain(alive) or st.left(used) == 0:
            return SUBMIT
        cands = [v for v in st.candidates(alive) if matters(b, alive, v)]
        if not cands:
            return SUBMIT
        _, edges = closing_graph(b, alive, cands)
        roots = [v for v in cands if not any(w == v for _, w in edges)] or cands
        return Action("ask", max(roots, key=lambda v: (b.gain(alive, v), -st.askable.index(v))))


def item_belief(row):
    sid = row["sample_id"]
    if sid not in BENCH:
        return None, None
    s = build(to_sample(row), Executor("out/cache"), vary=BENCH[sid], offline=True)
    if s is None or s.belief is None:
        return None, None
    return sid, s.belief


def exact_eval(b, pol_fn, budget, lam):
    amb = [v for v in b.slots if b.splits(b.root, v)]
    st = Setting(b, askable=amb, budget=budget, lam=lam, noise=0.0, show_cost=None)
    succ = asks = 0.0
    for t in b.root:
        e = run_episode(st, pol_fn(), SimulatedUser(b, t, noise=0.0, seed=t))
        succ += b.w[t] * e.success
        asks += b.w[t] * e.asks
    return succ, asks


def one1(row):
    sid, b = item_belief(row)
    if b is None:
        return None
    amb, edges = closing_graph(b, b.root, b.slots)
    if len(amb) < 2:
        return None
    rec = {"id": sid, "shape": shape(amb, edges), "slots": len(amb), "edges": sorted(edges)}
    for name, fn in (("greedy", Greedy), ("topo_order", TopoOrder), ("optimal", Optimal)):
        rec[name + "_asks"] = exact_eval(b, fn, None, 0.001)[1]
        rec[name + "_s1"] = exact_eval(b, fn, 1, 0.001)[0]
    return rec


def check1():
    with Pool(48) as pool:
        R = [r for r in pool.imap_unordered(one1, load_rows(), chunksize=1) if r]
    json.dump(R, open("out/theory_check1.json", "w"))
    print("items with >= 2 slots that matter: %d" % len(R))
    print("%-12s %4s | %-30s | %-30s | %s" % ("shape", "n", "asks to certainty g / order / opt", "success @1q g / order / opt",
                                            "order = optimal (asks)"))
    for sh in ("no closing", "forest", "other"):
        rs = [r for r in R if r["shape"] == sh]
        if not rs:
            continue
        m = lambda k: statistics.mean(r[k] for r in rs)
        eq = sum(abs(r["topo_order_asks"] - r["optimal_asks"]) < 1e-6 for r in rs)
        geq = sum(abs(r["greedy_asks"] - r["optimal_asks"]) < 1e-6 for r in rs)
        print("%-12s %4d | %.3f / %.3f / %.3f %10s | %.3f / %.3f / %.3f %10s | %d/%d (greedy %d/%d)" % (
            sh, len(rs), m("greedy_asks"), m("topo_order_asks"), m("optimal_asks"), "",
            m("greedy_s1"), m("topo_order_s1"), m("optimal_s1"), "", eq, len(rs), geq, len(rs)))
    bad = [r for r in R if r["shape"] == "forest" and r["topo_order_asks"] - r["optimal_asks"] > 1e-6]
    for r in bad[:5]:
        print("  forest item where order != optimal:", r["id"], r["edges"], "%.3f vs %.3f" % (r["topo_order_asks"], r["optimal_asks"]))


# ------------------------------------------------------------------------------------------------
def reachable(b, slots):
    seen, todo = {b.root}, [b.root]
    while todo:
        S = todo.pop()
        for v in slots:
            for h in b.split(S, v).values():
                if h and h not in seen:
                    seen.add(h)
                    todo.append(h)
    return len(seen), len({b.canon(S) for S in seen})


def mcts_eval(b, sims, abstract, truths, lam=0.05, noise=0.2):
    amb = [v for v in b.slots if b.splits(b.root, v)]
    st = Setting(b, askable=amb, budget=None, lam=lam, noise=noise, show_cost=None)
    us = []
    for j, t in enumerate(truths):
        e = run_episode(st, TopoMCTS(n_sim=sims, abstract=abstract, seed=j), SimulatedUser(b, t, noise=noise, seed=j))
        us.append(e.success - lam * e.asks)
    return statistics.mean(us)


def one2(row):
    sid, b = item_belief(row)
    if b is None:
        return None
    amb = [v for v in b.slots if b.splits(b.root, v)]
    if len(amb) < 2:
        return None
    n_alive, n_canon = reachable(b, amb)
    rng = random.Random(int(hashlib.md5(sid.encode()).hexdigest()[:8], 16))
    truths = rng.choices(b.root, weights=b.w, k=20)
    rec = {"id": sid, "states": n_alive, "canon": n_canon}
    for sims in (10, 30):
        rec["dag@%d" % sims] = mcts_eval(b, sims, False, truths)
        rec["canon@%d" % sims] = mcts_eval(b, sims, True, truths)
    return rec


def gate_rec(m):
    b = fam_gate(m, 3, 0.5, 1.0)
    amb = [v for v in b.slots if b.splits(b.root, v)]
    n_alive, n_canon = reachable(b, amb)
    rng = random.Random(m)
    truths = rng.choices(b.root, weights=b.w, k=200)
    rec = {"m": m, "states": n_alive, "canon": n_canon}
    for sims in (10, 30):
        rec["dag@%d" % sims] = mcts_eval(b, sims, False, truths)
        rec["canon@%d" % sims] = mcts_eval(b, sims, True, truths)
    return rec


def check2():
    with Pool(48) as pool:
        R = [r for r in pool.imap_unordered(one2, load_rows(), chunksize=1) if r]
        G = list(pool.imap(gate_rec, [4, 5, 6]))
    json.dump({"real": R, "gate": G}, open("out/theory_check2.json", "w"))
    m = lambda k, rs: statistics.mean(r[k] for r in rs)
    print("M items %d: reachable states %.1f (alive sets) vs %.1f (canonical forms); items with any merge %d" % (
        len(R), m("states", R), m("canon", R), sum(r["canon"] < r["states"] for r in R)))
    print("  utility (cost 0.05, 20%% noise): DAG @10 %.3f @30 %.3f | canonical @10 %.3f @30 %.3f" % (
        m("dag@10", R), m("dag@30", R), m("canon@10", R), m("canon@30", R)))
    for g in G:
        print("gate m=%d: states %d vs canonical %d | DAG @10 %.3f @30 %.3f | canonical @10 %.3f @30 %.3f" % (
            g["m"], g["states"], g["canon"], g["dag@10"], g["dag@30"], g["canon@10"], g["canon@30"]))


if __name__ == "__main__":
    check1() if sys.argv[1] == "1" else check2()
