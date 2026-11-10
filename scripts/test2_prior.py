"""Raw LLM slot probabilities for the test2 items (out/ambioverpass_m3x.jsonl, out/ambioverpass_m4x.jsonl), same prompt
and parsing as scripts/landmark_prior.py. No calibration is fitted here: the evaluation uses the DEV-fitted alphas in
out/prior_calib_landmark.json unchanged.

  AMBI_EXTRA_ROWS=out/onl_rows.jsonl python scripts/test2_prior.py   -> out/llm_prior_raw_test2.json
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import eval_m3 as E
sys.argv = _argv
from landmark_prior import ask, build
from ambitopo.data import load_rows
from ambitopo.llm import OpenAICompatible
from ambitopo.overpass import Executor

OUT_RAW = os.environ.get("OUT_RAW", "out/llm_prior_raw_test2.json")          # test3: OUT_RAW=out/llm_prior_raw_test3.json BENCHES=m3y,m4y


def main():
    rows = {r["sample_id"]: r for r in load_rows()}
    ex = Executor("out/cache")
    items = [json.loads(l) for b in os.environ.get("BENCHES", "m3x,m4x").split(",") for l in open("out/ambioverpass_%s.jsonl" % b)]
    raw = json.load(open(OUT_RAW)) if os.path.exists(OUT_RAW) else {}
    jobs, n_items = [], 0
    for it in items:
        r = build(it, rows, ex)
        if r is None:
            continue
        n_items += 1
        topo = r[0]
        for v in topo.order:
            labels = [x.label for x in topo.slots[v].readings]
            if len(labels) >= 2 and v not in raw.get(it["item_id"], {}):
                jobs.append((it["item_id"], it["x_vague"], v, labels))
    llm = OpenAICompatible()
    with ThreadPoolExecutor(32) as pool:
        for (iid, _, v, _), p in zip(jobs, pool.map(lambda j: ask(llm, j[1], E.WHAT.get(j[2], j[2]), j[3]), jobs)):
            raw.setdefault(iid, {})[v] = p
    json.dump(raw, open(OUT_RAW, "w"))
    print("items %d | new slot priors %d | file %s" % (n_items, len(jobs), OUT_RAW))


if __name__ == "__main__":
    main()
