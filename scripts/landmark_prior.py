"""Raw LLM priors for every slot of every landmark item (M3 + M4), and a per-slot-kind calibration p^alpha
fitted on the DEV split only (alpha in [0, 3], maximising the gold reading's log-likelihood; alpha = 0 is uniform).

  python scripts/landmark_prior.py      ->  out/llm_prior_raw_landmark.json, out/prior_calib_landmark.json
"""
import hashlib
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import eval_m3 as E
sys.argv = _argv
from ambitopo.data import load_rows, to_sample
from ambitopo.llm import OpenAICompatible
from ambitopo.near import build_near, build_near2
from ambitopo.overpass import Executor
from ambitopo.readings import lookup_areas

KIND = {"anchor1": "anchor", "anchor2": "anchor", "dist1": "dist", "dist2": "dist"}
OUT_RAW, OUT_CAL = "out/llm_prior_raw_landmark.json", "out/prior_calib_landmark.json"


def build(it, rows, ex):
    s = to_sample(rows[it["sample_id"]])
    areas = lookup_areas(s.spec.area_name, ex, offline=True)
    if "anchor_kinds" in it:
        r = build_near2(s.spec, s.tag_candidates, areas, tuple(it["anchor_kinds"]), ex, offline=True,
                        seed=int(hashlib.md5(it["item_id"].encode()).hexdigest()[:8], 16), gold=it["gold"])
    else:
        r = build_near(s.spec, s.tag_candidates, areas, it["anchor_kind"], ex, offline=True,
                       seed=int(hashlib.md5(it["sample_id"].encode()).hexdigest()[:8], 16), gold=it["gold"])
    return r


def ask(llm, request, what, labels):
    opts = "\n".join("%d. %s" % (i + 1, l) for i, l in enumerate(labels))
    txt = llm.chat([{"role": "user", "content": "A user typed this map search: \"%s\"\nFor the question of %s, how likely is "
                     "each option to be what the user means? Options:\n%s\nGive a probability for every option (they should sum "
                     "to 1). Output JSON only: a list of numbers in option order." % (request, what, opts)}],
                   temperature=0.0, max_tokens=120)
    m = re.search(r"\[[^\]]*\]", txt)
    try:
        p = [max(0.0, float(x)) for x in json.loads(m.group(0))][:len(labels)]
        assert len(p) == len(labels) and sum(p) > 0
        return [0.05 / len(p) + 0.95 * x / sum(p) for x in p]
    except Exception:
        return [1.0 / len(labels)] * len(labels)


def main():
    rows = {r["sample_id"]: r for r in load_rows()}
    ex = Executor("out/cache")
    items = [json.loads(l) for b in ("m3", "m4") for l in open("out/ambioverpass_%s.jsonl" % b)]
    raw = json.load(open(OUT_RAW)) if os.path.exists(OUT_RAW) else {}
    jobs, gold = [], {}
    for it in items:
        r = build(it, rows, ex)
        if r is None:
            continue
        topo, b, g = r
        gold[it["item_id"]] = (it["split"], dict(zip(topo.order, g)), {v: len(topo.slots[v].readings) for v in topo.order})
        for v in topo.order:
            labels = [x.label for x in topo.slots[v].readings]
            if len(labels) >= 2 and v not in raw.get(it["item_id"], {}):
                jobs.append((it["item_id"], it["x_vague"], v, labels))
    llm = OpenAICompatible()
    with ThreadPoolExecutor(32) as pool:
        for (iid, _, v, _), p in zip(jobs, pool.map(lambda j: ask(llm, j[1], E.WHAT.get(j[2], j[2]), j[3]), jobs)):
            raw.setdefault(iid, {})[v] = p
    json.dump(raw, open(OUT_RAW, "w"))
    # calibration on DEV only
    cal = {}
    for kind in ("area", "tag", "anchor", "dist", "type"):
        pairs = [(raw[iid][v], g[v]) for iid, (sp, g, n) in gold.items() if sp == "dev"
                 for v in raw.get(iid, {}) if KIND.get(v, v) == kind]
        if not pairs:
            continue

        def ll(a):
            t = 0.0
            for p, gi in pairs:
                q = [x ** a for x in p]
                t += math.log(q[gi] / sum(q))
            return t
        grid = [i / 20 for i in range(61)]
        best = max(grid, key=ll)
        top1 = sum(max(range(len(p)), key=lambda i: p[i]) == gi for p, gi in pairs) / len(pairs)
        cal[kind] = best
        print("%-6s n=%4d  LLM top-1 acc %.2f  log-lik uniform %.1f -> alpha=%.2f %.1f" % (kind, len(pairs), top1, ll(0.0), best, ll(best)))
    json.dump(cal, open(OUT_CAL, "w"))
    print("calibration", cal)


if __name__ == "__main__":
    main()
