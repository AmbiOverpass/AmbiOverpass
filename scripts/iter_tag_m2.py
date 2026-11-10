"""Iteration 1, single-object subset: sampled choices for the target-tag slot of the
M2 items behind the M2b items of the chosen splits, and step A (held-out log-likelihood with a power alpha fitted by 4-fold
source cross-validation, top-1) against the official single-object prior (4-order averaged).
  SPLITS=dev python scripts/iter_tag_m2.py   -> out/iter_tag_samples_m2_<splits>.json"""
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter_tag_sampling as T
from ambitopo.data import build, load_rows, to_sample
from ambitopo.overpass import Executor

SPLITS = set(os.environ.get("SPLITS", "dev").split(","))
OUT = "out/iter_%s_samples_m2_%s%s%s%s.json" % (T.SLOT, "-".join(sorted(SPLITS)), "_from%d" % T.T0 if T.T0 else "", "_" + T.MTAG if T.MTAG else "", "_counts" if T.COUNTS else "")
rows = {r["sample_id"]: r for r in load_rows()}
m2 = {json.loads(l)["item_id"]: json.loads(l) for l in open("out/ambioverpass_m2.jsonl")}
gold, split = {}, {}
for l in open("out/ambioverpass_m2b.jsonl"):
    it = json.loads(l)
    if it["split"] in SPLITS and T.SLOT not in it["gold_override"]:
        split[it["prior_from"]] = it["split"]
jobs, info = [], {}
for iid in split:
    it = m2[iid]
    s = build(to_sample(rows[it["sample_id"]]), Executor("out/cache"), vary=tuple(it["vary"]), offline=True)
    sl = s.topo.slots.get(T.SLOT)
    if sl is None or len(sl.readings) < 2:
        continue
    labels = [r.label for r in sl.readings]
    if T.COUNTS and s.belief is not None:
        n = T.counts_of(s.belief, T.SLOT)
        jobs.append((iid, it["x_vague"], labels, ["%s (%d matching objects)" % (l, n.get(j, 0)) for j, l in enumerate(labels)]))
    else:
        jobs.append((iid, it["x_vague"], labels))
    info[iid] = sl.gold
with ThreadPoolExecutor(32) as pool:
    res = dict(pool.map(T.one, jobs))
json.dump({"N": T.N, "counts": res}, open(OUT, "w"))
if SPLITS != {"dev"} or T.MTAG or T.COUNTS:          # step A is a DEV check only; other splits only store their samples
    sys.exit(0)
official = json.load(open("out/llm_prior_m2_perm4.json"))
data = []
for iid, c in res.items():
    p1 = [(x + 0.5) / (T.N + 0.5 * len(c)) for x in c]
    p0 = official[iid][T.SLOT]
    z = [math.sqrt(a * b) for a, b in zip(p0, p1)]
    data.append((m2[iid]["sample_id"], info[iid], {"official": p0, "C1": p1, "C2": [x / sum(z) for x in z]}))
srcs = sorted({s for s, _, _ in data})
random.Random(0).shuffle(srcs)
fold = {s: j % 4 for j, s in enumerate(srcs)}
GRID = [i / 10 for i in range(31)]


def ll(pairs, a):
    return sum(math.log(max(p[g], 1e-9) ** a / sum(max(x, 1e-9) ** a for x in p)) for p, g in pairs)


for name in ("official", "C1", "C2"):
    tot, n = 0.0, 0
    for f in range(4):
        tr = [(P[name], g) for s, g, P in data if fold[s] != f]
        te = [(P[name], g) for s, g, P in data if fold[s] == f]
        a = max(GRID, key=lambda x: ll(tr, x))
        tot += ll(te, a)
        n += len(te)
    top = sum(max(range(len(P[name])), key=lambda i: P[name][i]) == g for s, g, P in data)
    print("%-8s items %d | held-out mean log-lik %.4f | top-1 %.3f" % (name, len(data), tot / n, top / len(data)))
