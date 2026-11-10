"""AmbiOverpass-M3 ("target near anchor"): pick M2b source samples whose gold area is city/district level, attach an
anchor kind, execute one Overpass request per candidate area (online, cached), and write the precise and vague
requests with the local LLM.

  python scripts/build_m3.py N [offset]      # N source samples (pilot: 15); writes/extends out/ambioverpass_m3.jsonl
  M3_POOL=ext M3_OUT=out/ambioverpass_m3x.jsonl python scripts/build_m3.py N
      # expansion: source samples NOT used by M2b/M3 (fresh held-out test set, split "test2"; methods are frozen)
"""
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo.data import load_rows, to_sample
from ambitopo.llm import OpenAICompatible
from ambitopo.near import ANCHOR_WORDS, ANCHORS, RADII, build_near
from ambitopo.overpass import Executor
from ambitopo.query import QuerySpec
from ambitopo.readings import lookup_areas

LOCAL = {"6", "7", "8", "9", "10"}

PROMPT = """A precise map request: "{clear}"
Extend it so that it only asks for these objects within {r} metres of {anchor} ({anchor_tag}), then write how a
non-expert would type it. In the vague version: describe the main objects in vague everyday words that could fit
several related kinds; replace the exact distance by "near" or "close to"; name the anchor in everyday words that
could fit several kinds of {anchor_kind}; keep the place name. Never mention OpenStreetMap tags or query syntax.
Output JSON only: {{"precise": "...", "vague": "..."}}"""


def main():
    n = int(sys.argv[1])
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ex = Executor("out/cache", sleep=3)
    llm = OpenAICompatible()
    m2b = [json.loads(l) for l in open("out/ambioverpass_m2b.jsonl")]
    split = {it["sample_id"]: it["split"] for it in m2b}
    rows = {r["sample_id"]: r for r in load_rows()}
    EXT = os.environ.get("M3_POOL") == "ext"
    OUT = os.environ.get("M3_OUT", "out/ambioverpass_m3.jsonl")
    if EXT:
        used = set(split)
        split = {sid: "test2" for sid in rows if sid not in used}
    cands = []
    ROWS_FILE = os.environ.get("M3_ROWS")             # lazy mode over a shuffled rows file: stop after n eligible sources
    if EXT and ROWS_FILE:
        order = [json.loads(l)["sample_id"] for l in open(ROWS_FILE)]
        split = {sid: "test2" for sid in order}
        for sid in order:
            if len(cands) >= n + offset:
                break
            s = to_sample(rows[sid])
            if s is None or not isinstance(s.spec, QuerySpec) or s.spec.main is None:
                continue
            areas = lookup_areas(s.spec.area_name, ex, offline=False)
            if areas and (areas[0].get("admin_level") in LOCAL or areas[0].get("place") in ("city", "town", "suburb")):
                cands.append((sid, s, areas))
                print("eligible", len(cands), sid, s.spec.area_name, flush=True)
    for sid in ([] if (EXT and ROWS_FILE) else sorted(split)):
        s = to_sample(rows[sid])
        if s is None or not isinstance(s.spec, QuerySpec) or s.spec.main is None:
            continue
        areas = lookup_areas(s.spec.area_name, ex, offline=not EXT)
        if areas and (areas[0].get("admin_level") in LOCAL or areas[0].get("place") in ("city", "town", "suburb")):
            cands.append((sid, s, areas))
    if not (EXT and ROWS_FILE):
        random.Random(0).shuffle(cands)
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["item_id"] for l in open(OUT)}
    per = int(os.environ.get("M3_KINDS", 1))          # anchor kinds per source sample
    out = open(OUT, "a")
    print("eligible source samples:", len(cands), flush=True)
    for sid, s, areas in cands[offset:offset + n]:
      seed = int(hashlib.md5(sid.encode()).hexdigest()[:8], 16)
      first = random.Random(seed).choice(sorted(ANCHORS))
      others = [k for k in sorted(ANCHORS) if k != first]
      random.Random(seed + 1).shuffle(others)
      for kind in ([first] + others)[:per]:
        if sid + "|near:" + kind in done:
            continue
        built = build_near(s.spec, s.tag_candidates, areas, kind, ex, offline=False, seed=seed)
        if built is None:
            print("skip", sid, "(no non-empty gold)", flush=True)
            continue
        topo, b, gold = built
        r = RADII[gold[topo.order.index("dist")]]
        at = ANCHORS[kind][gold[topo.order.index("anchor")]]
        txt = llm.chat([{"role": "user", "content": PROMPT.format(clear=s.x_clear, r=r, anchor=ANCHOR_WORDS[kind],
                                                                   anchor_tag=at, anchor_kind=kind)}],
                       temperature=0.7, max_tokens=300)
        m = re.search(r"\{.*\}", txt, re.S)
        try:
            d = json.loads(m.group(0))
        except Exception:
            d = {"precise": "", "vague": ""}
        amb = [v for v in b.slots if b.splits(b.root, v)]
        rec = {"sample_id": sid, "item_id": sid + "|near:" + kind, "anchor_kind": kind, "gold": list(gold),
               "x_clear": d.get("precise", ""), "x_vague": d.get("vague", ""), "split": split[sid],
               "hyps": len(b.hyps), "classes": b.n_classes, "slots": amb, "areas": len(areas)}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(sid, kind, "hyps", len(b.hyps), "classes", b.n_classes, "slots", amb, "|", rec["x_vague"][:90], flush=True)


if __name__ == "__main__":
    main()
