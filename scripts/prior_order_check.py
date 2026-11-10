"""Position check of the LLM prior (2026-10-22 audit). In the benchmark the gold reading of the area, tag and type
slots is always option 1 of the prompt. Re-ask the same prompt with the options in a random order (fixed seed per
item and slot) and compare with the stored priors: top-1 accuracy and the gold reading's probability. NPERM > 1 averages
the probabilities over NPERM random orders (permutation-debiased prior).
  python scripts/prior_order_check.py                                    -> out/llm_prior_raw_landmark_shuffled.json
  NPERM=4 OUT=out/llm_prior_raw_landmark_perm4.json python scripts/prior_order_check.py
  BENCHES=m3x,m4x AMBI_EXTRA_ROWS=out/onl_rows.jsonl RAW_IN=out/llm_prior_raw_test2.json OUT=... python scripts/prior_order_check.py"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import landmark_prior as L
import eval_m3 as E
sys.argv = _argv
from ambitopo.data import load_rows
from ambitopo.llm import OpenAICompatible
from ambitopo.overpass import Executor

OUT = os.environ.get("OUT", "out/llm_prior_raw_landmark_shuffled.json")
NPERM = int(os.environ.get("NPERM", "1"))
raw = json.load(open(os.environ.get("RAW_IN", "out/llm_prior_raw_landmark.json")))
rows = {r["sample_id"]: r for r in load_rows()}
ex = Executor("out/cache")
jobs = []
for b in os.environ.get("BENCHES", "m3,m4").split(","):
    for l in open("out/ambioverpass_%s.jsonl" % b):
        it = json.loads(l)
        r = L.build(it, rows, ex)
        if r is None or it["item_id"] not in raw:
            continue
        topo, bel, g = r
        for v, gi in zip(topo.order, g):
            if v not in raw[it["item_id"]]:
                continue
            labels = [x.label for x in topo.slots[v].readings]
            perms = []
            for t in range(NPERM):
                perm = list(range(len(labels)))
                random.Random("%s|%s" % (it["item_id"], v) + ("|%d" % t if t else "")).shuffle(perm)   # perm[j] = original index at position j
                perms.append(perm)
            jobs.append((it["item_id"], it["split"], it["x_vague"], v, labels, perms, gi))
llm = OpenAICompatible()


def run(j):
    iid, sp, req, v, labels, perms, gi = j
    q = [0.0] * len(labels)
    for perm in perms:
        p = L.ask(llm, req, E.WHAT.get(v, v), [labels[k] for k in perm])
        for pos, k in enumerate(perm):
            q[k] += p[pos] / len(perms)
    return q


with ThreadPoolExecutor(32) as pool:
    res = list(pool.map(run, jobs))
out = {}
for (iid, sp, req, v, labels, perms, gi), q in zip(jobs, res):
    out.setdefault(iid, {})[v] = q
json.dump(out, open(OUT, "w"))
stat = {}
for (iid, sp, req, v, labels, perms, gi), q in zip(jobs, res):
    perm = perms[0]
    k = L.KIND.get(v, v)
    p = raw[iid][v]
    s = stat.setdefault(k, {"n": 0, "top_orig": 0, "top_shuf": 0, "pg_orig": 0.0, "pg_shuf": 0.0, "uniform": 0.0, "top_shown_first": 0})
    s["n"] += 1
    s["top_orig"] += max(range(len(p)), key=lambda i: p[i]) == gi
    s["top_shuf"] += max(range(len(q)), key=lambda i: q[i]) == gi
    s["top_shown_first"] += max(range(len(q)), key=lambda i: q[i]) == perm[0]
    s["pg_orig"] += p[gi]
    s["pg_shuf"] += q[gi]
    s["uniform"] += 1 / len(p)
for k, s in stat.items():
    n = s["n"]
    print("%-6s n=%3d  top-1 gold: original order %.3f  shuffled %.3f | top = option shown first (shuffled) %.3f | "
          "mean p(gold): original %.3f  shuffled %.3f  uniform %.3f" % (k, n, s["top_orig"] / n, s["top_shuf"] / n,
          s["top_shown_first"] / n, s["pg_orig"] / n, s["pg_shuf"] / n, s["uniform"] / n))
