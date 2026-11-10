"""LLM prior over readings: for each M2 item and each ambiguous slot, the local LLM scores how likely each
candidate reading is what the user means, given only the vague request. The prior over joint readings is
the product over slots (shared by every planner in main_table.py with MT_PRIOR=llm).

  python scripts/llm_prior.py           # writes out/llm_prior_m2.json {item_id: {slot: [p per reading]}}
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo.data import build, load_rows, to_sample
from ambitopo.llm import OpenAICompatible
from ambitopo.overpass import Executor

WHAT = {"area": "which place is meant", "tag": "which kind of map object is meant",
        "type": "which geometry is meant (node = point, way = line or outline, relation = grouped object, nwr = any)",
        "filter": "whether this extra condition is meant", "dist": "which search radius in metres is meant"}


def scores(llm, request, slot, labels):
    opts = "\n".join("%d. %s" % (i + 1, l) for i, l in enumerate(labels))
    prompt = ("A user typed this map search: \"%s\"\nFor the question of %s, how likely is each option to be what the user "
              "means? Options:\n%s\nGive a probability for every option (they should sum to 1). Output JSON only: "
              "a list of numbers in option order." % (request, WHAT.get(slot, slot), opts))
    out = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=120)
    m = re.search(r"\[[^\]]*\]", out)
    try:
        p = [max(0.0, float(x)) for x in json.loads(m.group(0))][:len(labels)]
    except Exception:
        p = []
    if len(p) != len(labels) or sum(p) <= 0:
        return [1.0 / len(labels)] * len(labels)
    z = sum(p)
    return [0.05 / len(labels) + 0.95 * x / z for x in p]       # floor: never rule a reading out


def main():
    llm = OpenAICompatible()
    rows = {r["sample_id"]: r for r in load_rows()}
    items = [json.loads(l) for l in open("out/ambioverpass_m2.jsonl")]
    items = [it for it in items if not it["flags"]]

    def one(it):
        s = build(to_sample(rows[it["sample_id"]]), Executor("out/cache"), vary=tuple(it["vary"]), offline=True)
        out = {}
        for v in s.topo.order:
            sl = s.topo.slots[v]
            if len(sl.readings) > 1:
                out[v] = scores(llm, it["x_vague"], v, [r.label for r in sl.readings])
        return it["item_id"], out

    with ThreadPoolExecutor(24) as pool:
        P = dict(pool.map(one, items))
    json.dump(P, open("out/llm_prior_m2.json", "w"))
    print("items", len(P), "slots scored", sum(len(v) for v in P.values()))


if __name__ == "__main__":
    main()
