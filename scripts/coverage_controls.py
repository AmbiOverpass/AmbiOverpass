"""Controls for coverage_check.py: is the gain of multi-step rewriting due to the rewrite operations,
or just to more LLM calls / more diverse sampling?

  resample@5x10   align the hard request 5 times (temperature 0.8), union         -- same #calls as steps
  paraphrase      4 plain paraphrases (no CORRECT/RELAX instruction), align each + original to 10
Reads out/hard_items.jsonl and out/coverage_check.json (for the steps / oneshot numbers on the same items).
"""
import json
import os
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo.llm import OpenAICompatible
from coverage_check import SYS, norm, tags


def align_t(llm, request, n, temperature):
    prompt = ("Request: %s\nList %d different OpenStreetMap tags (key=value) that could describe the MAIN kind of "
              "object the user wants, most likely first. Output a JSON list of strings like [\"amenity=cafe\", ...]."
              % (request, n))
    return tags(llm.chat([{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                         temperature=temperature, max_tokens=60 + 14 * n))[:n]


def paraphrase(llm, request):
    prompt = ("Request: %s\nWrite 4 different paraphrases of this request that keep its meaning. "
              "Output a JSON list of 4 strings." % request)
    out = llm.chat([{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                   temperature=0.5, max_tokens=400)
    try:
        lst = json.loads(re.search(r"\[.*\]", out, re.S).group(0))
        return [x for x in lst if isinstance(x, str)][:4]
    except Exception:
        return [l.strip("-• ").strip() for l in out.splitlines() if len(l.strip()) > 10][:4]


def union(lists):
    out = []
    for l in lists:
        for t in l:
            if t not in out:
                out.append(t)
    return out


def one(args):
    llm, it = args
    gold = norm(it["gold_tag"])
    res = union(align_t(llm, it["x_hard"], 10, 0.8) for _ in range(5))
    par = union([align_t(llm, it["x_hard"], 10, 0.3)] + [align_t(llm, p, 10, 0.3) for p in paraphrase(llm, it["x_hard"])])
    return {"id": it["sample_id"], "resample": gold in res, "n_resample": len(res),
            "paraphrase": gold in par, "n_paraphrase": len(par)}


def main():
    llm = OpenAICompatible()
    base = {r["id"]: r for r in json.load(open(sys.argv[1] if len(sys.argv) > 1 else "out/coverage_check.json"))}
    items = [json.loads(l) for l in open("out/hard_items.jsonl")]
    items = [it for it in items if it["sample_id"] in base]
    with ThreadPoolExecutor(16) as pool:
        R = list(pool.map(one, [(llm, it) for it in items]))
    for r in R:
        r.update({k: base[r["id"]][k] for k in ("oneshot8", "oneshot50", "steps", "n50", "nsteps", "perturbations")})
    json.dump(R, open("out/coverage_controls.json", "w"), ensure_ascii=False)
    m = lambda k, rs=R: 100 * statistics.mean(r[k] for r in rs)
    n = lambda k: statistics.mean(r[k] for r in R)
    print("items %d" % len(R))
    print("coverage %%  (candidates):  oneshot@50 %.1f (%.1f) | resample 5x10 %.1f (%.1f) | paraphrase %.1f (%.1f) | "
          "rewrite steps %.1f (%.1f)" % (m("oneshot50"), n("n50"), m("resample"), n("n_resample"),
                                         m("paraphrase"), n("n_paraphrase"), m("steps"), n("nsteps")))
    for k in ("everyday", "concept_wrong"):
        rs = [r for r in R if r["perturbations"][0] == k]
        print("  %-14s n=%3d  oneshot@50 %.1f  resample %.1f  paraphrase %.1f  steps %.1f"
              % (k, len(rs), m("oneshot50", rs), m("resample", rs), m("paraphrase", rs), m("steps", rs)))
    only = lambda a, b: sum(r[a] and not r[b] for r in R)
    print("steps-only %d vs resample-only %d | steps-only %d vs paraphrase-only %d"
          % (only("steps", "resample"), only("resample", "steps"), only("steps", "paraphrase"), only("paraphrase", "steps")))


if __name__ == "__main__":
    main()
