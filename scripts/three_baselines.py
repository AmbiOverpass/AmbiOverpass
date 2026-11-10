"""Paired comparison of TopoMCTS (final configuration) with three mechanism baselines, on the existing TEST splits:
  B1 result-class information gain (ask the max class-entropy-drop slot only if (1-eps)*Delta_v - lam > 0; a tau=0.9
     confidence-stop variant is reported separately), B2 Shapley greedy, B3 one-step Shapley rollout.
Same items, priors, gold, seeds and user random streams as the paper's main runs (user seed = seed*1000003 + md5 of the
sample id for single-object / single-landmark items, of the item id for two-landmark items). Every episode is logged
with its question trace, stop reason and online planning time. Each method gets its own freshly built belief so that
cached quantities are not shared across methods (fair runtime).

  CALIB_FILE=out/prior_calib_landmark.json python scripts/three_baselines.py   -> out/three_baselines.jsonl
"""
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import eval_m3 as E
sys.argv = _argv
from ambitopo import Belief, EIGClasses, Setting, ShapleyGreedy, ShapleyRollout, SimulatedUser, TopoMCTS
from ambitopo.data import build, load_rows, to_sample
from ambitopo.near import build_near, build_near2
from ambitopo.overpass import Executor
from ambitopo.readings import lookup_areas

LAM, NOISE, SEEDS = 0.05, 0.2, 5
CELLS = json.loads(os.environ["TB_CELLS"]) if os.environ.get("TB_CELLS") else {"m4": (2, 3), "m3": (3,), "m2b": (2,)}
METHODS = tuple(os.environ["TB_METHODS"].split(";")) if os.environ.get("TB_METHODS") else ("TopoMCTS", "B1 RC-IG", "B1' RC-IG (tau stop)", "B2 Shapley greedy", "B3 One-step rollout")
SPLIT = os.environ.get("TB_SPLIT", "test")
OUTF = os.environ.get("TB_OUT", "out/three_baselines.jsonl")
ROWS = {r["sample_id"]: r for r in load_rows()}
PRIOR_M2 = json.load(open("out/llm_prior_m2.json"))
CALIB_M2 = json.load(open("out/prior_calib.json"))


def policy(name, seed):
    return {"TopoMCTS": lambda: TopoMCTS(n_sim=300, seed=seed, prior="lookahead_shapley", rollout="shapley"),
            "B1 RC-IG": lambda: EIGClasses(stop="delta"),
            "B1' RC-IG (tau stop)": lambda: EIGClasses(stop="tau"),
            "B2 Shapley greedy": lambda: ShapleyGreedy(),
            "B3 One-step rollout": lambda: ShapleyRollout()}[name]()


def item_belief(b_name, it):
    """-> (topo, hyps, prior weights, gold index, user-seed base) exactly as in the paper's runs."""
    row = ROWS[it["sample_id"]]
    ex = Executor("out/cache")
    if b_name == "m2b":
        s = build(to_sample(row), ex, vary=tuple(it["vary"]), offline=True)
        if s is None or s.belief is None:
            return None
        b = s.belief
        P = {}
        for v, p in PRIOR_M2.get(it.get("prior_from", it["item_id"]), {}).items():
            q = [max(x, 1e-9) ** CALIB_M2.get(v, 1.0) for x in p]
            P[v] = [x / sum(q) for x in q]
        ov = it.get("gold_override", {})
        gold = tuple(ov.get(n, s.topo.slots[n].gold) for n in s.topo.order)
        topo, hyps = b.topo, b.hyps
        base = int(hashlib.md5(it["item_id"].encode()).hexdigest()[:8], 16)
    else:
        s = to_sample(row)
        areas = lookup_areas(s.spec.area_name, ex, offline=True)
        if b_name == "m3":
            base = int(hashlib.md5(it["sample_id"].encode()).hexdigest()[:8], 16)
            built = build_near(s.spec, s.tag_candidates, areas, it["anchor_kind"], ex, offline=True, seed=base, gold=it["gold"])
        else:
            base = int(hashlib.md5(it["item_id"].encode()).hexdigest()[:8], 16)
            built = build_near2(s.spec, s.tag_candidates, areas, tuple(it["anchor_kinds"]), ex, offline=True, seed=base, gold=it["gold"])
        if built is None:
            return None
        topo, b, gold = built
        hyps = b.hyps
        P = E.llm_prior(None, it["x_vague"], topo, it["item_id"])
    w = []
    for h in hyps:
        x = 1.0
        for v, r in zip(b.slots, h.assign):
            if v in P:
                x *= P[v][r]
        w.append(x)
    gi = [i for i, h in enumerate(hyps) if tuple(h.assign) == tuple(gold)]
    if not gi or not [v for v in b.slots if b.splits(b.root, v)]:
        return None
    return topo, hyps, w, gi[0], base


def episode(st, pol, user, K):
    b = st.belief
    alive, asks, t, trace, asked = b.root, 0, 0.0, [], []
    while True:
        t0 = time.perf_counter()
        a = pol.act(st, alive, asks)
        t += time.perf_counter() - t0
        if a.kind != "ask":
            break
        ans = user.answer(a.slot)
        asks += 1
        asked.append(a.slot)
        if ans is not None:
            nxt = b.observe(alive, a.slot, ans)
            if nxt:
                alive = nxt
        trace.append([a.slot, "no preference" if ans is None else b.topo.slots[a.slot].readings[ans].label, len(alive)])
        if asks > 50:
            break
    cls = b.map_class(alive)
    stop = "certain" if b.certain(alive) else ("budget" if K is not None and asks >= K else "policy_stop")
    return cls, asks, t, stop, asked, trace


def one(job):
    b_name, it = job
    got = item_belief(b_name, it)
    if got is None:
        return []
    topo, hyps, w, g, base = got
    out = []
    for m in METHODS:
        b = Belief(topo, hyps, prior=w)                 # fresh belief per method
        amb = [v for v in b.slots if b.splits(b.root, v)]
        for K in CELLS[b_name]:
            st = Setting(b, askable=amb, budget=K, lam=LAM, noise=NOISE, show_cost=None)
            for seed in range(SEEDS):
                user = SimulatedUser(b, g, noise=NOISE, seed=seed * 1000003 + base)
                cls, q, t, stop, asked, trace = episode(st, policy(m, seed), user, K)
                ex = float(cls == b.cls[g])                  # tau = 1: same class <=> identical result set
                out.append({"sample_id": it["item_id"], "subset": b_name, "method": m, "seed": seed, "K": K,
                            "true_result_class": int(b.cls[g]), "predicted_result_class": int(cls), "EX": ex,
                            "question_count": q, "utility": ex - LAM * q, "runtime_ms": 1000 * t, "stop_reason": stop,
                            "asked_slots": asked, "answer_trace": trace})
    return out


if __name__ == "__main__":
    jobs = []
    for b_name in CELLS:
        for l in open("out/ambioverpass_%s.jsonl" % b_name):
            it = json.loads(l)
            if it.get("split") == SPLIT and not it.get("flags"):
                jobs.append((b_name, it))
    with Pool(48) as pool, open(OUTF, "w") as f:
        for recs in pool.imap_unordered(one, jobs):
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("done", len(jobs), "items")
