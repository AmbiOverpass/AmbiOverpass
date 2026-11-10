"""AmbiOverpass-M2: every (collected sample, set of weakened slots) whose belief has >= 2 slots that matter and
a non-empty gold reading, from the existing execution cache (no new Overpass requests). Vague requests are
written by the local open model (Qwen3-14B via scripts/serve_llm.sh) under the AST-grounded instructions
of ambitopo/compound.py, one retry when the quality checks flag the rewrite. Records the closing blind spot B.

  python scripts/build_m2.py            # writes out/ambioverpass_m2.jsonl
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ambitopo.compound import rewrite
from ambitopo.data import build, load_rows, to_sample
from ambitopo.llm import OpenAICompatible
from ambitopo.overpass import Executor
from ambitopo.query import AroundSpec
from relation import closing

SUBS = [s for n in (2, 3, 4) for s in itertools.combinations(("area", "tag", "type", "filter"), n) if "tag" in s]
SUBS_AROUND = [("area", "type"), ("area", "filter"), ("area", "type", "filter")]


def one(row):
    s0 = to_sample(row)
    if s0 is None:
        return []
    llm = OpenAICompatible()
    around = isinstance(s0.spec, AroundSpec)
    out = []
    for sub in (SUBS_AROUND if around else SUBS):
        s = build(to_sample(row), Executor("out/cache"), vary=sub, offline=True)
        if s is None or s.belief is None:
            continue
        b = s.belief
        c = closing(b, b.slots)
        gold = tuple(s.topo.slots[n].gold for n in s.topo.order)
        if c["slots"] < 2 or not any(h.assign == gold for h in b.hyps):
            continue
        names = [("dist" if (around and v == "area") else v) for v in sub]
        r = rewrite(s, names, llm)
        if r["flags"]:
            r2 = rewrite(s, names, llm)
            if len(r2["flags"]) < len(r["flags"]):
                r = r2
        out.append({"item_id": "%s|%s" % (row["sample_id"], "+".join(sub)), "sample_id": row["sample_id"],
                    "vary": list(sub), "weakened": names, "x_clear": s.x_clear, "x_vague": r["x_vague"],
                    "flags": r["flags"], "B": c["B"], "n_matter": c["slots"], "gold_query": s.spec.source})
    return out


if __name__ == "__main__":
    with Pool(24) as pool:
        R = [x for r in pool.imap_unordered(one, load_rows(), chunksize=1) for x in r]
    with open("out/ambioverpass_m2.jsonl", "w") as f:
        for x in R:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    ok = [x for x in R if not x["flags"]]
    print("M2 items %d | pass checks %d | B>0 %d | samples %d" % (len(R), len(ok), sum(x["B"] > 1e-9 for x in ok),
                                                                 len({x["sample_id"] for x in ok})))
    for x in ok[:6]:
        print(" ", x["weakened"], "|", x["x_clear"], "=>", x["x_vague"])
