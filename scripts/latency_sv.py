"""[latency_sv.py: final configuration] Per-decision planning time at the root (ms), single process: greedy, TopoMCTS (100 sims), TopoMCTS-LA (300 sims,
look-ahead prior), TopoMCTS-SV (300 sims, Shapley prior + rollout), Shapley values, exact (cold start, canonical memo). M test / M3 / M4, 40 items each, budget K = 3 and no budget.

  python scripts/latency_m4.py
"""
import hashlib
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo import Greedy, Optimal, Setting, TopoMCTS
from ambitopo.planners import ShapleyGreedy
from ambitopo.data import build, load_rows, to_sample
from ambitopo.near import build_near, build_near2
from ambitopo.overpass import Executor
from ambitopo.readings import lookup_areas

rows = {r["sample_id"]: r for r in load_rows()}
ex = Executor("out/cache")


def beliefs(bench, n=40):
    its = [json.loads(l) for l in open("out/ambioverpass_%s.jsonl" % bench)]
    random.Random(0).shuffle(its)
    for it in its:
        s = to_sample(rows[it["sample_id"]])
        if bench == "m2b":
            if it["flags"] or it["split"] != "test":
                continue
            s = build(s, ex, vary=tuple(it["vary"]), offline=True)
            b = s.belief if s else None
        else:
            areas = lookup_areas(s.spec.area_name, ex, offline=True)
            if bench == "m3":
                r = build_near(s.spec, s.tag_candidates, areas, it["anchor_kind"], ex, offline=True,
                               seed=int(hashlib.md5(it["sample_id"].encode()).hexdigest()[:8], 16), gold=it["gold"])
            else:
                r = build_near2(s.spec, s.tag_candidates, areas, tuple(it["anchor_kinds"]), ex, offline=True,
                                seed=int(hashlib.md5(it["item_id"].encode()).hexdigest()[:8], 16), gold=it["gold"])
            b = r[1] if r else None
        if b is not None:
            yield b
            n -= 1
            if n == 0:
                return


out = {}
for bench in ("m2b", "m3", "m4"):
    T = {}
    for b in beliefs(bench):
        amb = [v for v in b.slots if b.splits(b.root, v)]
        for K in (3, None):
            st = Setting(b, askable=amb, budget=K, lam=0.05, noise=0.2, show_cost=None)
            for name, mk in (("Greedy", lambda: Greedy()), ("TopoMCTS", lambda: TopoMCTS(n_sim=100)),
                             ("TopoMCTS-LA", lambda: TopoMCTS(n_sim=300, prior="lookahead")),
                             ("TopoMCTS-SV", lambda: TopoMCTS(n_sim=300, prior="lookahead_shapley", rollout="shapley")),
                             ("ShapleyGreedy", lambda: ShapleyGreedy()), ("Exact", lambda: Optimal(canonical=True))):
                pol = mk()
                t0 = time.perf_counter()
                pol.act(st, b.root, 0)
                T.setdefault("%s|K=%s" % (name, K), []).append(1000 * (time.perf_counter() - t0))
    out[bench] = {k: {"mean": statistics.mean(v), "median": statistics.median(v), "max": max(v)} for k, v in T.items()}
    for k, v in out[bench].items():
        print(bench, "%-22s mean %8.1f ms  median %8.1f  max %9.1f" % (k, v["mean"], v["median"], v["max"]), flush=True)
json.dump(out, open("out/latency_sv.json", "w"))
