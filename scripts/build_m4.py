"""AmbiOverpass-M4 ("target near anchor 1 and near anchor 2"), built offline from the Overpass requests cached for
M3 (one per source sample, anchor kind and area); the two-anchor join is local. Vague requests by the local LLM.

  python scripts/build_m4.py            # writes out/ambioverpass_m4.jsonl (skips items already written)
"""
import hashlib
import itertools
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo.data import load_rows, to_sample
from ambitopo.llm import OpenAICompatible
from ambitopo.near import ANCHOR_WORDS, ANCHORS, RADII, build_near2
from ambitopo.overpass import Executor
from ambitopo.readings import lookup_areas

PROMPT = """A precise map request: "{clear}"
Extend it so that it only asks for these objects that are within {r1} metres of {a1} ({t1}) and also within {r2}
metres of {a2} ({t2}), then write how a non-expert would type it. In the vague version: describe the main objects in
vague everyday words that could fit several related kinds; replace both exact distances by "near" or "close to";
name both landmarks in everyday words that could fit several kinds; keep the place name. Never mention
OpenStreetMap tags or query syntax. Output JSON only: {{"precise": "...", "vague": "..."}}"""


def main():
    ex = Executor("out/cache")
    llm = OpenAICompatible()
    rows = {r["sample_id"]: r for r in load_rows()}
    m3 = [json.loads(l) for l in open(os.environ.get("M4_IN", "out/ambioverpass_m3.jsonl"))]
    split = {it["sample_id"]: it["split"] for it in m3}
    path = os.environ.get("M4_OUT", "out/ambioverpass_m4.jsonl")
    done = {json.loads(l)["item_id"] for l in open(path)} if os.path.exists(path) else set()
    out = open(path, "a")
    for sid in sorted(split):
        s = to_sample(rows[sid])
        areas = lookup_areas(s.spec.area_name, ex, offline=True)
        for kinds in itertools.combinations(sorted(ANCHORS), 2):
            iid = sid + "|near:" + "+".join(kinds)
            if iid in done:
                continue
            seed = int(hashlib.md5(iid.encode()).hexdigest()[:8], 16)
            try:
                built = build_near2(s.spec, s.tag_candidates, areas, kinds, ex, offline=True, seed=seed)
            except Exception:
                built = None
            if built is None:
                continue
            topo, b, gold = built
            o = topo.order
            g = dict(zip(o, gold))
            txt = llm.chat([{"role": "user", "content": PROMPT.format(
                clear=s.x_clear, r1=RADII[g["dist1"]], a1=ANCHOR_WORDS[kinds[0]], t1=ANCHORS[kinds[0]][g["anchor1"]],
                r2=RADII[g["dist2"]], a2=ANCHOR_WORDS[kinds[1]], t2=ANCHORS[kinds[1]][g["anchor2"]])}],
                temperature=0.7, max_tokens=300)
            m = re.search(r"\{.*\}", txt, re.S)
            try:
                d = json.loads(m.group(0))
            except Exception:
                d = {"precise": "", "vague": ""}
            amb = [v for v in b.slots if b.splits(b.root, v)]
            rec = {"sample_id": sid, "item_id": iid, "anchor_kinds": list(kinds), "gold": list(gold),
                   "x_clear": d.get("precise", ""), "x_vague": d.get("vague", ""), "split": split[sid],
                   "hyps": len(b.hyps), "classes": b.n_classes, "slots": amb, "areas": len(areas)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(iid, "hyps", len(b.hyps), "classes", b.n_classes, "slots", len(amb), "|", rec["x_vague"][:90], flush=True)


if __name__ == "__main__":
    main()
