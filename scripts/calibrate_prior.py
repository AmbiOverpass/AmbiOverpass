"""Calibrate the LLM prior per slot type on the DEV split of M2b: p_alpha(r) ∝ p(r)^alpha (alpha on a grid in [0, 3];
a monotone recalibration that never inverts the LLM's ranking), chosen to maximise
the dev log-likelihood of the gold reading. Reports dev and test log-likelihood, top-1 accuracy and confidence
before and after. Writes out/prior_calib.json {slot: alpha}.
"""
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo.data import build, load_rows, to_sample
from ambitopo.overpass import Executor

GRID = [round(0.1 * i, 1) for i in range(31)]      # 0.0 .. 3.0: calibration may soften or sharpen the LLM, never invert it


def apply(p, a):
    q = [max(x, 1e-9) ** a for x in p]
    z = sum(q)
    return [x / z for x in q]


def main():
    rows = {r["sample_id"]: r for r in load_rows()}
    P = json.load(open(os.environ.get("PRIOR_FILE", "out/llm_prior_m2.json")))
    data = collections.defaultdict(lambda: {"dev": [], "test": []})       # slot -> split -> [(p, gold)]
    for l in open("out/ambioverpass_m2b.jsonl"):
        it = json.loads(l)
        s = build(to_sample(rows[it["sample_id"]]), Executor("out/cache"), vary=tuple(it["vary"]), offline=True)
        for v, p in P.get(it["prior_from"], {}).items():
            g = it["gold_override"].get(v, s.topo.slots[v].gold)
            data[v][it["split"]].append((p, g))
    calib = {}
    for v, d in data.items():
        ll = lambda a, xs: sum(math.log(apply(p, a)[g]) for p, g in xs) / max(1, len(xs))
        best = max(GRID, key=lambda a: ll(a, d["dev"]))
        calib[v] = best
        for split in ("dev", "test"):
            xs = d[split]
            if not xs:
                continue
            acc = sum(p.index(max(p)) == g for p, g in xs) / len(xs)
            conf0 = sum(max(p) for p, _ in xs) / len(xs)
            conf1 = sum(max(apply(p, best)) for p, _ in xs) / len(xs)
            acc1 = sum(apply(p, best).index(max(apply(p, best))) == g for p, g in xs) / len(xs)
            print("%-6s %-4s n=%4d | loglik raw %.3f -> calibrated %.3f (alpha %.1f) | top-1 acc %.2f -> %.2f | confidence %.2f -> %.2f"
                  % (v, split, len(xs), ll(1.0, xs), ll(best, xs), best, acc, acc1, conf0, conf1))
    json.dump(calib, open(os.environ.get("CAL_OUT", "out/prior_calib.json"), "w"))
    print("alpha per slot:", calib)


if __name__ == "__main__":
    main()
