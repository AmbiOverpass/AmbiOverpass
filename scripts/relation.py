"""Topology <-> TopoMCTS: does the topology predict where look-ahead beats greedy? (RESULTS.md, "how they relate")

A slot v MATTERS in state S if two readings in S that differ only in v give different result classes
(EC² gain is not usable for this: any question that eliminates readings has positive gain).
Closing power, at the root state S of a belief, over the slots that matter:
  close(u->v) = P(after the answer to u, v no longer matters)
  Sav(u)      = sum_v close(u->v)            expected other questions made unnecessary by asking u
  B           = max_u Sav(u) - Sav(greedy's first choice)          greedy's blind spot for ORDER
  factorised  = no interlock (no slot matters under some readings of another but not under others)
  stop trap   = no single question pays for itself now (one-step accuracy gain < lam = 0.05), although
                asking does pay overall (optimal utility > answering now): greedy's blind spot for STOPPING,
                arises when questions are complementary (several must be answered before any helps)

  real   benchmark items: closing statistics joined with the per-item results of run_structural.py
         (ask=all, noise 0, lam 0.001, no show; budgets 1, 2 and "to certainty")
  synth  families with exact expectations over the prior: factorised, interlocked tree, gate(p, m),
         partial gate (a fraction q of the children silenced by the gate's closing reading)

  python scripts/relation.py real [out/structural_bench_uniform.jsonl]
  python scripts/relation.py synth
"""
import json
import math
import os
import statistics
import sys
from itertools import product
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo import (Belief, Greedy, Hypothesis, Optimal, Reading, Setting, SimulatedUser, Slot, Topology, TopoMCTS,
                      run_episode)

EPS = 1e-12
LAM = 0.05


def matters(b, S, v):
    k = b.slots.index(v)
    groups = {}
    for i in S:
        a = b.hyps[i].assign
        groups.setdefault(a[:k] + a[k + 1:], set()).add(b.cls[i])
    return any(len(c) > 1 for c in groups.values())


def closing(b, slots):
    """close(u->v), Sav(u), greedy's first choice and the blind spot B at the root."""
    S = b.root
    amb = [v for v in slots if matters(b, S, v)]
    tot = b.mass(S)
    close = {}
    for u in amb:
        parts = b.split(S, u)
        for v in amb:
            if v == u:
                continue
            close[(u, v)] = sum(b.mass(h) / tot for h in parts.values() if not matters(b, h, v))
    sav = {u: sum(close[(u, v)] for v in amb if v != u) for u in amb}
    if not amb:
        return {"slots": 0, "B": 0.0, "factorised": True, "sav_max": 0.0}
    g = max(amb, key=lambda v: (b.gain(S, v), -amb.index(v)))
    inter = any(0 < close[k] < 1 for k in close)
    now = b.acc_stop(S)
    one_step = max(sum(b.mass(h) / tot * b.acc_stop(h) for h in b.split(S, v).values()) - now for v in amb)
    st = Setting(b, askable=[v for v in b.slots if b.splits(S, v)], budget=None, lam=LAM, noise=0.0, show_cost=None)
    opt = Optimal().value(st)
    return {"slots": len(amb), "B": max(sav.values()) - sav[g], "sav_max": max(sav.values()),
            "sav_greedy": sav[g], "factorised": not inter, "greedy_first": g,
            "best_closer": max(amb, key=lambda u: sav[u]), "one_step": one_step, "acc_now": now,
            "opt_value": opt, "stop_trap": one_step < LAM and opt > now + 1e-9}


# ---------------------------------------------------------------------------------------------
# real benchmark
def real_one(args):
    row, bench = args
    from ambitopo.data import build, to_sample
    from ambitopo.overpass import Executor
    s = to_sample(row)
    if s is None or s.sample_id not in bench:
        return None
    build(s, Executor("out/cache"), vary=bench[s.sample_id], offline=True)
    if s.belief is None:
        return None
    b = s.belief
    return {"id": s.sample_id, **closing(b, b.slots)}


def real(path):
    from ambitopo.data import load_rows
    bench = {}
    for l in open("out/ambioverpass_m.jsonl"):
        it = json.loads(l)
        if not it["flags"]:
            bench[it["sample_id"]] = tuple(v if v != "dist" else "area" for v in it["weakened"])
    with Pool(48) as pool:
        C = {r["id"]: r for r in pool.imap_unordered(real_one, [(r, bench) for r in load_rows()]) if r}
    res = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
    rows = []
    for i, c in C.items():
        if i not in res or not res[i].get("runs") or c["slots"] < 2:
            continue
        rec = dict(c)
        for B, lam in (("1", "0.001"), ("2", "0.001"), ("None", "0.050")):
            r = res[i]["runs"].get("ask=all|B=%s|noise=0.0|lam=%s|show=None" % (B, lam))
            if not r:
                continue
            for p in ("greedy_ec2", "topo_mcts", "optimal"):
                rec["%s@%s" % (p, B)] = r[p]["success"] if B != "None" else r[p]["utility"]
        rows.append(rec)
    json.dump(rows, open("out/relation_real.json", "w"))
    report(rows, "real benchmark items with >= 2 ambiguous slots")


def report(rows, title):
    print("\n== %s: %d ==" % (title, len(rows)))
    strata = [("factorised (no interlock)", lambda r: r["factorised"]),
              ("interlocked, B = 0", lambda r: not r["factorised"] and r["B"] <= 1e-9),
              ("interlocked, 0 < B <= 0.5", lambda r: not r["factorised"] and 1e-9 < r["B"] <= 0.5),
              ("interlocked, B > 0.5", lambda r: not r["factorised"] and r["B"] > 0.5)]
    print("%-28s %5s | %-26s | %-26s | %-27s" % ("stratum", "n", "success @1 q: opt-greedy / mcts-greedy",
                                                  "@2 q: opt-g / mcts-g", "utility (lam 0.05): g / mcts / opt"))
    for name, f in strata:
        rs = [r for r in rows if f(r) and "greedy_ec2@1" in r]
        if not rs:
            print("%-28s %5d" % (name, 0))
            continue
        m = lambda k: statistics.mean(r[k] for r in rs)
        print("%-28s %5d | %+.3f / %+.3f %13s | %+.3f / %+.3f %13s | %.2f / %.2f / %.2f" % (
            name, len(rs), m("optimal@1") - m("greedy_ec2@1"), m("topo_mcts@1") - m("greedy_ec2@1"), "",
            m("optimal@2") - m("greedy_ec2@2"), m("topo_mcts@2") - m("greedy_ec2@2"), "",
            m("greedy_ec2@None"), m("topo_mcts@None"), m("optimal@None")))
    print("by stopping (utility, lam 0.05, no budget):")
    for name, f in (("no stop trap", lambda r: not r["stop_trap"]), ("stop trap", lambda r: r["stop_trap"])):
        rs = [r for r in rows if f(r) and "greedy_ec2@None" in r]
        if rs:
            m = lambda k: statistics.mean(r[k] for r in rs)
            print("  %-26s %5d | greedy %.3f  TopoMCTS %.3f  optimum %.3f" % (name, len(rs), m("greedy_ec2@None"),
                                                                            m("topo_mcts@None"), m("optimal@None")))
    fz = [r for r in rows if r["factorised"] and not r["stop_trap"] and "greedy_ec2@None" in r]
    if fz:
        same = sum(abs(r["greedy_ec2@None"] - r["optimal@None"]) < 1e-6 for r in fz)
        print("Proposition 1 check (factorised, no stop trap): greedy's utility == optimum in %d / %d" % (same, len(fz)))


# ---------------------------------------------------------------------------------------------
# synthetic families
def slot(name, R, parents=()):
    return Slot(name, "x", [Reading(j, "%s=%d" % (name, j), "gold" if j == 0 else "align") for j in range(R)],
                gold=0, handling="ask", parents=list(parents))


def fam_factorised(m, R):
    topo = Topology([slot("c%d" % i, R) for i in range(m)])
    hyps = [Hypothesis(idx, frozenset({hash(idx) % 10 ** 9})) for idx in product(range(R), repeat=m)]
    return Belief(topo, hyps)


def fam_gate(m, R, p, q):
    """Gate g: reading 1 (prior p) silences the first ceil(q*m) children; reading 0 lets all matter."""
    k = math.ceil(q * m)
    topo = Topology([Slot("g", "gate", [Reading(0, "g=0"), Reading(1, "g=1", "align")], gold=0, handling="ask")]
                    + [slot("c%d" % i, R, [("g", "candidates")]) for i in range(m)])
    hyps, w = [], []
    for idx in product(range(2), *[range(R)] * m):
        key = idx if idx[0] == 0 else (1,) + tuple(idx[1 + k:])
        hyps.append(Hypothesis(tuple(idx), frozenset({hash((7,) + key) % 10 ** 9})))
        w.append((p if idx[0] == 1 else 1 - p) / R ** m)
    return Belief(topo, hyps, prior=w)


def fam_tree(depth, R):
    n = 2 ** depth - 1
    topo = Topology([slot("s%d" % i, R, [("s%d" % ((i - 1) // 2), "candidates")] if i else []) for i in range(n)])
    hyps = []
    for idx in product(range(R), repeat=n):
        path, i = [], 0
        while i < n:
            path.append((i, idx[i]))
            i = 2 * i + (1 if idx[i] == 0 else 2)
        hyps.append(Hypothesis(tuple(idx), frozenset({hash(tuple(path)) % 10 ** 9})))
    return Belief(topo, hyps)


def exact(b, pol_fn, budget, lam):
    amb = [v for v in b.slots if b.splits(b.root, v)]
    st = Setting(b, askable=amb, budget=budget, lam=lam, noise=0.0, show_cost=None)
    succ = asks = 0.0
    for t in b.root:
        e = run_episode(st, pol_fn(), SimulatedUser(b, t, noise=0.0, seed=t))
        succ += b.w[t] * e.success
        asks += b.w[t] * e.asks
    return succ, succ - lam * asks


def synth_one(cfg):
    name, b = cfg[0], {"factorised": lambda: fam_factorised(*cfg[1]), "gate": lambda: fam_gate(*cfg[1]),
                       "tree": lambda: fam_tree(*cfg[1])}[cfg[0]]()
    rec = {"family": name, "params": cfg[1], **closing(b, b.slots)}
    for B in (1, 2, None):
        for pname, fn in (("greedy_ec2", Greedy), ("topo_mcts", lambda: TopoMCTS(n_sim=30)),
                          ("topo_mcts100", lambda: TopoMCTS(n_sim=100)), ("optimal", Optimal)):
            s, u = exact(b, fn, B, 0.001 if B is not None else LAM)
            rec["%s@%s" % (pname, B)] = s if B is not None else u
    return rec


def synth():
    cfgs = [("factorised", (m, 3)) for m in (2, 3, 4)] + [("tree", (2, 3)), ("tree", (3, 2))]
    cfgs += [("gate", (m, 3, p, 1.0)) for m in (2, 3, 4) for p in (0.1, 0.3, 0.5, 0.7)]
    cfgs += [("gate", (4, 3, 0.5, q)) for q in (0.25, 0.5, 0.75)]
    with Pool(min(os.cpu_count() or 8, len(cfgs))) as pool:
        rows = list(pool.imap(synth_one, cfgs))
    json.dump(rows, open("out/relation_synth.json", "w"))
    print("%-11s %-18s %5s %6s %5s | %-21s | %-21s | %s" % ("family", "params", "slots", "B", "fact",
                                                         "@1 q: greedy/mcts30/mcts100/opt", "@2 q: same",
                                                         "utility lam 0.05: same"))
    for r in rows:
        f = lambda B: " / ".join("%.3f" % r["%s@%s" % (p, B)] for p in ("greedy_ec2", "topo_mcts", "topo_mcts100", "optimal"))
        print("%-11s %-18s %5d %6.2f %5s | %s | %s | %s" % (r["family"], r["params"], r["slots"], r["B"],
                                                         "yes" if r["factorised"] else "no", f(1), f(2), f(None)))
    report(rows, "synthetic configurations")


if __name__ == "__main__":
    if sys.argv[1] == "real":
        real(sys.argv[2] if len(sys.argv) > 2 else "out/structural_bench_uniform.jsonl")
    else:
        synth()
