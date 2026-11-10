"""AmbiOverpass-H: harder requests whose gold query needs several rewrite steps to recover.

From each gold query (single statement, named area) we compose 2-3 perturbations, AST-grounded:
the LLM only rewrites the parts named in the instructions, and we know the gold reading.
  concept_wrong       object described through a property OSM does not record   -> needs CORRECT
  everyday            object named in vague everyday words fitting several kinds -> needs ALIGN
  spurious_condition  an added requirement OSM cannot express                    -> needs RELAX
  bare_place          place given only by its bare name                          -> needs ALIGN (area)
The object is always perturbed (concept_wrong or everyday), plus one or two of the others.
Tier 2 (--tier 2, AmbiOverpass-H2): the object gets both at once (indirect property AND vague words), so
recovering it takes two rewrite steps.
Writes out/hard_items.jsonl (tier 1) or out/hard2_items.jsonl. Needs the local LLM server (scripts/serve_llm.sh).
--heldout: build from the rows NOT used above (no named area: bbox / area-id / coordinates), one object tag
per query; only the object slot is evaluated there, so no place perturbation (out/hard{,2}_heldout.jsonl).
"""
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo.data import load_rows, to_sample
from ambitopo.llm import OpenAICompatible
from ambitopo.query import QuerySpec

SYSTEM = """You simulate a non-expert user asking for map data.
You get a precise request and instructions for how to make it harder to interpret. Produce ONE rewritten
request that a real person might type.
Rules: same subject; follow every instruction and change nothing else; never mention OpenStreetMap tags,
keys or query syntax; one natural sentence. Output only the rewritten request."""

PERTURB = {
    "concept_wrong": "Describe the main kind of object ({value}) through a property OpenStreetMap does not record, "
                     "one step off from what it really is (style examples: 'the best restaurants', 'five-star hotels', "
                     "'new buildings'). The real kind of object must still be inferable, but do not name it.",
    "everyday": "Replace the name of the main kind of object ({value}) with vague everyday words that could fit "
                "several related kinds of objects. Do not use the original word.",
    "spurious_condition": "Add one natural-sounding requirement that a map database cannot express, such as being "
                          "quiet, popular, cheap, nice or well-reviewed.",
    "bare_place": "Refer to the place only by its bare name, without country, region or administrative words.",
    "double": "Refer to the main kind of object ({value}) only indirectly: through a property OpenStreetMap does "
              "not record (what it is good for, what people say about it, how it feels), in vague everyday words "
              "that could fit several kinds of objects. Do not name it or any close synonym. A careful reader must "
              "still be able to work out what kind of object is meant.",
}
OPS = {"concept_wrong": "correct", "everyday": "align", "spurious_condition": "relax", "bare_place": "align_area",
       "double": "correct+align"}
TIER = int(sys.argv[sys.argv.index("--tier") + 1]) if "--tier" in sys.argv else 1
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0     # seed 1 = held-out re-draw


def choose(rng):
    obj = "double" if TIER == 2 else rng.choice(["concept_wrong", "everyday"])
    extra = rng.sample(["spurious_condition", "bare_place"], k=rng.choice([1, 2]))
    return [obj] + extra


HELDOUT = "--heldout" in sys.argv
TAGF = re.compile(r'\[\s*"([^"]+)"\s*=\s*"([^"]+)"\s*\]')


def heldout_items(rng):
    out = []
    for row in load_rows():
        s = to_sample(row)
        if s is not None and isinstance(s.spec, QuerySpec):
            continue                                  # used in AmbiOverpass-H
        tags = TAGF.findall(row["gold_query"])
        if not tags or len(set(tags)) > 1:
            continue                                  # exactly one distinct object tag
        k, v = tags[0]
        if k in ("name", "ref", "type") or "~" in v:
            continue
        pert = choose(rng)
        pert = [p for p in pert if p != "bare_place"] or pert[:1]
        if len(pert) == 1:
            pert.append("spurious_condition")
        out.append({"sample_id": row["sample_id"], "x_clear": row["x_clear"], "gold_query": row["gold_query"],
                    "gold_tag": "%s=%s" % (k, v), "gold_conditions": [], "area_name": None, "perturbations": pert,
                    "ops_needed": [OPS[p] for p in pert], "value": v})
    return out


def flags(clear, hard, value):
    bad = []
    if re.search(r"[=\[\]]|\btagged\b|\bnodes?\b|\brelations?\b", hard):
        bad.append("syntax")
    if value and value.replace("_", " ").lower() in hard.lower():
        bad.append("object word kept")
    if hard.strip().lower() == clear.strip().lower() or len(hard) < 10:
        bad.append("degenerate")
    return bad


def main():
    llm = OpenAICompatible()
    rng = random.Random(SEED)
    items = []
    if HELDOUT:
        items = heldout_items(rng)
    for row in [] if HELDOUT else load_rows():
        s = to_sample(row)
        if s is None or not isinstance(s.spec, QuerySpec):
            continue
        main = s.spec.filters[s.spec.main]
        pert = choose(rng)
        items.append({"sample_id": s.sample_id, "x_clear": s.x_clear, "gold_query": s.spec.source,
                      "gold_tag": "%s=%s" % (main.key, main.val), "gold_conditions": [f.raw for f in s.spec.secondary],
                      "area_name": s.spec.area_name, "perturbations": pert, "ops_needed": [OPS[p] for p in pert],
                      "value": main.val})

    def work(it):
        ins = [PERTURB[p].format(value=it["value"].replace("_", " ")) for p in it["perturbations"]]
        user = "Precise request: %s\nInstructions:\n%s\nRewritten request:" % (it["x_clear"], "\n".join("- " + i for i in ins))
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        if SEED:                                   # a different draw of the wording (vLLM caches nothing)
            msgs[0]["content"] += "\n(Variant %d.)" % SEED
        out = llm.chat(msgs, temperature=0.7, max_tokens=200)
        it["x_hard"] = out.strip().strip('"').splitlines()[0] if out.strip() else ""
        it["flags"] = flags(it["x_clear"], it["x_hard"], it["value"])
        return it

    with ThreadPoolExecutor(16) as pool:
        done = list(pool.map(work, items))
    dst = ("out/hard_items" if TIER == 1 else "out/hard2_items") + ("_s%d" % SEED if SEED else "") + ".jsonl"
    if HELDOUT:
        dst = "out/hard_heldout.jsonl" if TIER == 1 else "out/hard2_heldout.jsonl"
    with open(dst, "w") as f:
        for it in done:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    ok = [it for it in done if not it["flags"]]
    print("items %d | pass checks %d" % (len(done), len(ok)))
    for it in ok[:5]:
        print(it["perturbations"], "|", it["x_clear"], "=>", it["x_hard"])


if __name__ == "__main__":
    main()
