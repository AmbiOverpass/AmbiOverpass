"""Iteration 1: sampled choices for the target-tag slot. For every item, N times: the candidate
tags in a fresh random order, temperature 1.0, the model picks one option; the counts per candidate are stored.
  SPLITS=dev BENCHES=m3,m4 N=12 python scripts/iter_tag_sampling.py   -> out/iter_tag_samples_<benches>_<splits>.json"""
import json
import os
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import landmark_prior as L
sys.argv = _argv
from ambitopo.data import load_rows
from ambitopo.overpass import Executor

N = int(os.environ.get("N", "12"))
T0 = int(os.environ.get("T0", "0"))          # first sample index (T0=12: 12 further samples with new orders, iteration 5)
BENCHES = os.environ.get("BENCHES", "m3,m4").split(",")
SPLITS = set(os.environ.get("SPLITS", "dev").split(","))
SLOT = os.environ.get("SLOT", "tag")          # SLOT=type: the same sampled choices for the element-type slot (iteration 4)
COUNTS = os.environ.get("COUNTS") == "1"     # iteration 11: show each option's number of matching objects
MTAG = os.environ.get("MODEL_TAG", "")      # MODEL_TAG=9b: samples of a second model (iteration 7)
OUT = "out/iter_%s_samples_%s_%s%s%s.json" % (SLOT, "-".join(BENCHES), "-".join(sorted(SPLITS)), "_from%d" % T0 if T0 else "", "_" + MTAG if MTAG else "") if not COUNTS else "out/iter_%s_samples_%s_%s_counts.json" % (SLOT, "-".join(BENCHES), "-".join(sorted(SPLITS)))
URL = os.environ["AMBITOPO_LLM_BASE_URL"].rstrip("/") + "/chat/completions"
MODEL = os.environ["AMBITOPO_LLM_MODEL"]


def chat(prompt, seed):
    req = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 1.0, "max_tokens": 8,
           "seed": seed, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL, data=json.dumps(req).encode(), headers={"Content-Type": "application/json",
                                                                         "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(r, timeout=120) as f:
        return json.loads(f.read())["choices"][0]["message"]["content"].strip()


def counts_of(b, slot):
    """Largest result size over the readings with each value of `slot` (execution results only)."""
    k = b.slots.index(slot)
    n = {}
    for h in b.hyps:
        n[h.assign[k]] = max(n.get(h.assign[k], 0), len(h.result))
    return n


def one(job):
    iid, request, labels = job[:3]
    shown = job[3] if len(job) > 3 else labels
    counts = [0] * len(labels)
    for t in range(T0, T0 + N):
        perm = list(range(len(labels)))
        random.Random("%s|%s|%d" % (iid, SLOT, t)).shuffle(perm)
        opts = "\n".join("%d. %s" % (j + 1, shown[k]) for j, k in enumerate(perm))
        what = {"tag": "Which kind of map object does the user mean?",
                "type": "Which geometry does the user mean (node = point, way = line or outline, relation = grouped object, nwr = any)?",
                "area": "Which place does the user mean?"}[SLOT]
        prompt = ("A user typed this map search: \"%s\"\n%s Options:\n%s\n"
                  "Answer with the number of one option only." % (request, what, opts))
        m = re.search(r"\d+", chat(prompt, seed=t))
        if m and 1 <= int(m.group(0)) <= len(labels):
            counts[perm[int(m.group(0)) - 1]] += 1
    return iid, counts


def main():
    rows = {r["sample_id"]: r for r in load_rows()}
    ex = Executor("out/cache")
    jobs = []
    for b in BENCHES:
        for l in open("out/ambioverpass_%s.jsonl" % b):
            it = json.loads(l)
            if (it.get("split") or "test2") not in SPLITS:
                continue
            r = L.build(it, rows, ex)
            if r is None:
                continue
            topo = r[0]
            labels = [x.label for x in topo.slots[SLOT].readings]
            if len(labels) >= 2:
                if COUNTS:
                    n = counts_of(r[1], SLOT)
                    jobs.append((it["item_id"], it["x_vague"], labels, ["%s (%d matching objects)" % (l, n.get(j, 0)) for j, l in enumerate(labels)]))
                else:
                    jobs.append((it["item_id"], it["x_vague"], labels))
    with ThreadPoolExecutor(32) as pool:
        res = dict(pool.map(one, jobs))
    json.dump({"N": N, "counts": res}, open(OUT, "w"))
    print("items", len(res), "->", OUT)


if __name__ == "__main__":
    main()
