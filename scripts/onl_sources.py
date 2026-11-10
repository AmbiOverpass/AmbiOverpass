"""New source rows for the test2 expansion, from the public OverpassNL release (train + dev + test, 8352 pairs).
Mirrors the original AmbiOverpass source pipeline (tools/build_ambi.py): the target is the query's main tag; its
candidate readings come from structural rules (tags co-occurring with it in OverpassNL queries, same value under another
key, same key with an overlapping value word), then an LLM keeps only those a vague description could mean (the
original used qwen-plus; here the local Qwen3-14B). Queries already used by AmbiOverpass are excluded (exact gold match).
Fixed before running; no selection by planner results.

  python scripts/onl_sources.py   -> out/onl_rows.jsonl (row format of the AmbiOverpass source files)
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo.llm import OpenAICompatible
from ambitopo.query import QuerySpec

SRC = os.environ.get("OVERPASSNL_DIR", "OverpassNL/dataset/v1/dataset")   # OverpassNL release (Staniek et al., 2024)
OLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "source")
TAG = re.compile(r'\["([^"!~]+)"="([^"]+)"\]')
FILTER = """You judge which OpenStreetMap tags a vague request could plausibly mean.

You get a request in which the kind of object may be described in everyday words, and a list of candidate tags.
Return the subset that a reasonable person could have meant by that kind of object.

Rules:
- Keep a candidate only if the request could genuinely be asking for it.
- Always keep the tag marked (original).
- Reply with the kept tags only, one per line, exactly as given. No explanation."""


def main():
    pairs = []
    for split in ("train", "dev", "test"):
        nl = open("%s/dataset.%s.nl" % (SRC, split)).read().split("\n")
        qs = open("%s/dataset.%s.query" % (SRC, split)).read().split("\n")
        pairs += [("onl_%s_%04d" % (split, i), n.strip(), q.strip()) for i, (n, q) in enumerate(zip(nl, qs)) if n.strip() and q.strip()]
    used = set()
    for f in os.listdir(OLD):
        if f.endswith(".jsonl"):
            for l in open(os.path.join(OLD, f)):
                try:
                    used.add(json.loads(l).get("gold_query", "").strip())
                except Exception:
                    pass
    # co-occurrence / value index over all OverpassNL queries (structural candidate pool, as in the original)
    cooc, val_index, same_key = defaultdict(Counter), defaultdict(set), defaultdict(Counter)
    for _, _, q in pairs:
        tags = sorted({"%s=%s" % m for m in TAG.findall(q)})
        for t in tags:
            k, v = t.split("=", 1)
            same_key[k][v] += 1
            val_index[v].add(t)
            for u in tags:
                if u != t:
                    cooc[t][u] += 1

    def candidates(target, max_n=6):
        key, val = target.split("=", 1)
        out, seen = [], {target}
        for u, _ in cooc[target].most_common(6):
            if u not in seen:
                seen.add(u); out.append(u)
        for v, ts in val_index.items():
            if v == val or (len(val) >= 5 and (val in v or v in val)):
                for t2 in sorted(ts):
                    if t2 not in seen:
                        seen.add(t2); out.append(t2)
        for v, _ in same_key[key].most_common(30):
            t2 = "%s=%s" % (key, v)
            if t2 not in seen and set(val.split("_")) & set(v.split("_")):
                seen.add(t2); out.append(t2)
        return out[:max_n]

    rows = []
    for sid, nl, q in pairs:
        if q in used:
            continue
        tags = ["%s=%s" % m for m in TAG.findall(q)]
        if not tags:
            continue
        spec = QuerySpec.parse(q, main_tag=tags[0])
        if spec is None or spec.main is None or "bbox" in q:
            continue
        rows.append({"sample_id": sid, "x_clear": nl, "x_amb": nl, "t": "t3_osm_semantics", "S": "z_tag", "c": tags[0],
                     "gold_query": q, "verify": "", "_cands": candidates(tags[0])})
    print("parsable new sources:", len(rows), "of", len(pairs), "| excluded as already used:", sum(q in used for _, _, q in pairs), flush=True)
    llm = OpenAICompatible()

    def filt(r):
        cands = [r["c"]] + r["_cands"]
        if len(cands) > 1:
            listing = "\n".join("%s%s" % (c, "  (original)" if c == r["c"] else "") for c in cands)
            txt = llm.chat([{"role": "system", "content": FILTER},
                            {"role": "user", "content": "Request: %s\n\nCandidates:\n%s" % (r["x_clear"], listing)}],
                           temperature=0.0, max_tokens=200)
            keep = {l.strip().split()[0] for l in txt.splitlines() if l.strip()} | {r["c"]}
            cands = [c for c in cands if c in keep]
        r["Q"] = [{"value": c} for c in cands]
        r["K"] = len(cands)
        del r["_cands"]
        return r

    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(filt, rows))
    random.Random(0).shuffle(rows)
    with open("out/onl_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote out/onl_rows.jsonl", len(rows), "| K distribution", dict(Counter(r["K"] for r in rows)))


if __name__ == "__main__":
    main()
