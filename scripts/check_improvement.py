"""Check of the policy-improvement guarantee (paper, Shapley look-ahead): for every evaluated landmark item and budget, the
exact expected utility under the belief of ShapleyRollout is at least that of ShapleyGreedy (and at most the exact
planner's). Policies are evaluated exactly by recursion over (alive set, questions left) with the noise model.
  RAW_FILE=... CALIB_FILE=... python scripts/check_improvement.py [m3|m4] [test|test2]"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import eval_m3 as E
from landmark_prior import build
sys.argv = _argv
from ambitopo import Belief, Optimal, Setting, ShapleyGreedy, ShapleyRollout
from ambitopo.data import load_rows
from ambitopo.overpass import Executor

BENCH = sys.argv[1] if len(sys.argv) > 1 else "m3"
PART = sys.argv[2] if len(sys.argv) > 2 else "test"
ROWS = None


def value(st, pol, alive, k, memo):
    key = (alive, k)
    if key in memo:
        return memo[key]
    b = st.belief
    a = pol.act(st, alive, st.left(0) - k)
    if a.kind != "ask":
        v = b.acc_stop(alive)
    else:
        tot = b.mass(alive)
        v = -st.lam + st.noise * value(st, pol, alive, k - 1, memo) + (1 - st.noise) * sum(
            b.mass(h) / tot * value(st, pol, h, k - 1, memo) for h in b.split(alive, a.slot).values())
    memo[key] = v
    return v


def job(it):
    r = build(it, ROWS, Executor("out/cache"))
    if r is None:
        return []
    topo, b0, gold = r
    P = E.llm_prior(None, it["x_vague"], topo, it["item_id"])
    w = []
    for h in b0.hyps:
        x = 1.0
        for v, rr in zip(b0.slots, h.assign):
            if v in P:
                x *= P[v][rr]
        w.append(x)
    b = Belief(topo, b0.hyps, prior=w)
    amb = [v for v in b.slots if b.splits(b.root, v)]
    out = []
    for K in (1, 2, 3, None):
        st = Setting(b, askable=amb, budget=K, lam=E.LAM, noise=E.NOISE)
        k = st.left(0)
        vs = {}
        for name, pol in (("greedy", ShapleyGreedy()), ("rollout", ShapleyRollout())):
            vs[name] = value(st, pol, b.root, k, {})
        vs["exact"] = Optimal(canonical=True).value(st, b.root, 0) if K is not None and len(amb) <= 7 else None
        out.append((it["item_id"], K, vs))
    return out


def main():
    global ROWS
    ROWS = {r["sample_id"]: r for r in load_rows()}
    f = "out/ambioverpass_%s.jsonl" % (BENCH if PART != "test2" else BENCH + "x")
    items = [json.loads(l) for l in open(f)]
    if PART == "test":
        items = [it for it in items if it["split"] == "test"]
    with Pool(16) as pool:
        R = [x for rs in pool.map(job, items) for x in rs]
    bad = [(i, K, v) for i, K, v in R if v["rollout"] < v["greedy"] - 1e-9]
    over = [(i, K, v) for i, K, v in R if v["exact"] is not None and v["rollout"] > v["exact"] + 1e-9]
    gain = [v["rollout"] - v["greedy"] for i, K, v in R]
    print("%s %s: %d (item, budget) pairs | rollout < greedy: %d | rollout > exact: %d | mean gain under belief %.4f, max %.4f, pairs with gain > 0: %d"
          % (BENCH, PART, len(R), len(bad), len(over), sum(gain) / len(gain), max(gain), sum(g > 1e-9 for g in gain)))
    for x in bad[:5]:
        print("  violation", x)


if __name__ == "__main__":
    main()
