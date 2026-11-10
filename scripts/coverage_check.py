"""Go/no-go for MCTS-built topologies: does multi-step rewriting cover the gold object tag better
than asking the LLM once for the same number of candidates?

  oneshot@8    one call, 8 candidate tags
  oneshot@50   one call, 50 candidate tags
  steps@~50    rewrite the request first (CORRECT unmodelled properties, RELAX unsupported requirements:
               4 alternative precise requests), then ALIGN each (and the original) to 10 tags
Coverage = gold main tag (key=value) among the candidates; symbolic, so a lower bound (a different tag
returning the same OSM objects would also satisfy the user). Bar: steps − oneshot@50 >= 15 points.
"""
import json
import os
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo.llm import OpenAICompatible

TAG = re.compile(r'"?([a-z_:]+)\s*=\s*([A-Za-z0-9_:;\- ]+?)"?(?=[,\]\n"]|$)')
SYS = "You map natural-language map requests to OpenStreetMap. Answer with JSON only."


def tags(text):
    raw = []
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            raw = [x for x in json.loads(m.group(0)) if isinstance(x, str)]
        except Exception:
            raw = []
    if not raw:
        raw = ["%s=%s" % kv for kv in TAG.findall(text)]
    out = []
    for x in raw:
        if "=" not in x:
            continue
        k, v = x.split("=", 1)
        t = "%s=%s" % (k.strip().strip('"[]').lower(), v.strip().strip('"[]').lower().replace(" ", "_"))
        if t not in out:
            out.append(t)
    return out


def align(llm, request, n):
    prompt = ("Request: %s\nList %d different OpenStreetMap tags (key=value) that could describe the MAIN kind of "
              "object the user wants, most likely first. Output a JSON list of strings like [\"amenity=cafe\", ...]."
              % (request, n))
    return tags(llm.chat([{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                         temperature=0.3, max_tokens=60 + 14 * n))[:n]


def rewrite(llm, request):
    prompt = ("Request: %s\nThe request may describe objects through properties a map database does not record "
              "(e.g. 'best', 'five-star', 'new') or add requirements it cannot check (e.g. 'quiet', 'cheap'). "
              "Write 4 alternative precise versions of the request that a map database can answer: replace "
              "unrecordable properties with the closest recordable kinds of objects and drop requirements it cannot "
              "check. Each version should commit to a different plausible kind of object. Output a JSON list of 4 "
              "strings." % request)
    out = llm.chat([{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                   temperature=0.5, max_tokens=400)
    try:
        lst = json.loads(re.search(r"\[.*\]", out, re.S).group(0))
        return [x for x in lst if isinstance(x, str)][:4]
    except Exception:
        return [l.strip("-• ").strip() for l in out.splitlines() if len(l.strip()) > 10][:4]


def norm(t):
    return t.lower().replace(" ", "_")


def one(args):
    llm, it = args
    gold = norm(it["gold_tag"])
    r = {"id": it["sample_id"], "perturbations": it["perturbations"]}
    c8 = align(llm, it["x_hard"], 8)
    c50 = align(llm, it["x_hard"], 50)
    steps = list(align(llm, it["x_hard"], 10))
    rewrites = rewrite(llm, it["x_hard"])
    for rw in rewrites:
        for t in align(llm, rw, 10):
            if t not in steps:
                steps.append(t)
    r.update({"oneshot8": gold in c8, "oneshot50": gold in c50, "steps": gold in steps,
              "n8": len(c8), "n50": len(c50), "nsteps": len(steps), "rewrites": rewrites,
              "clear8": gold in align(llm, it["x_clear"], 8)})
    return r


def main():
    llm = OpenAICompatible()
    items = [json.loads(l) for l in open("out/hard_items.jsonl")]
    items = [it for it in items if not it["flags"]]
    if len(sys.argv) > 1:
        items = items[:int(sys.argv[1])]
    with ThreadPoolExecutor(16) as pool:
        R = list(pool.map(one, [(llm, it) for it in items]))
    json.dump(R, open("out/coverage_check.json", "w"), ensure_ascii=False)
    m = lambda k: 100 * statistics.mean(r[k] for r in R)
    print("items %d" % len(R))
    print("coverage of the gold object tag (%%): clear request @8 %.1f | hard: oneshot@8 %.1f  oneshot@50 %.1f  steps %.1f"
          % (m("clear8"), m("oneshot8"), m("oneshot50"), m("steps")))
    print("candidates per item: oneshot@50 %.1f  steps %.1f" % (statistics.mean(r["n50"] for r in R),
                                                               statistics.mean(r["nsteps"] for r in R)))
    by = {}
    for r in R:
        by.setdefault(r["perturbations"][0], []).append(r)
    for k, rs in by.items():
        print("  %-14s n=%3d  oneshot@8 %.1f  oneshot@50 %.1f  steps %.1f" % (
            k, len(rs), 100 * statistics.mean(x["oneshot8"] for x in rs), 100 * statistics.mean(x["oneshot50"] for x in rs),
            100 * statistics.mean(x["steps"] for x in rs)))


if __name__ == "__main__":
    main()
