"""Benchmark structure for the paper: difficulty regime (Theorem 1), size, and search-state counts (Proposition 4)
for AmbiOverpass-M3 / M4. History-tree nodes are counted by memoised recursion over (state, asked slots).

  python scripts/structure_stats.py m3|m4
"""
import collections
import hashlib
import json
import math
import os
import statistics
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory_checks import closing_graph, shape
from ambitopo.data import build, load_rows, to_sample
from ambitopo.near import build_near, build_near2
from ambitopo.overpass import Executor
from ambitopo.readings import lookup_areas

BENCH = sys.argv[1]
ROWS = None


def regime(b):
    amb, edges = closing_graph(b, b.root, b.slots)
    full = len(b.hyps) == math.prod(len({h.assign[k] for h in b.hyps}) for k in range(len(b.slots)))
    sh = shape(amb, edges)
    if sh == "no closing" and full:
        return "factorisable"
    if sh == "forest" and full:
        return "gated forest"
    return "general"


def counts(b):
    amb = [v for v in b.slots if b.splits(b.root, v)]
    memo, alive = {}, set()

    def nodes(S, asked):
        key = (S, asked)
        if key in memo:
            return memo[key]
        alive.add(S)
        n = 1
        if not b.certain(S):
            for v in amb:
                if v in asked or not b.splits(S, v):
                    continue
                for h in b.split(S, v).values():
                    n += nodes(h, asked | {v})
        memo[key] = n
        return n

    hist = nodes(b.root, frozenset())
    canon = {b.canon(S) for S in alive}
    return hist, len(alive), len(canon)


def blind_spot(b):
    """B = max_u Sav(u) − Sav(greedy's choice) at the root (Section 4.6)."""
    from ambitopo import Greedy, Setting
    amb = [v for v in b.slots if b.splits(b.root, v)]
    n = b.mass(b.root)
    sav = {}
    for u in amb:
        sp = b.split(b.root, u)
        sav[u] = sum(sum(b.mass(h) / n * (not b.matters(h, w)) for h in sp.values()) for w in amb if w != u)
    a = Greedy().act(Setting(b, askable=amb, budget=1, lam=0.0, noise=0.0), b.root, 0)
    return (max(sav.values()) - sav[a.slot]) if a.kind == "ask" else 0.0


def one(it):
    s = to_sample(ROWS[it["sample_id"]])
    ex = Executor("out/cache")
    if BENCH == "m2b":
        s = build(s, ex, vary=tuple(it["vary"]), offline=True)
        if s is None or s.belief is None:
            return None
        b = s.belief
        amb = [v for v in b.slots if b.splits(b.root, v)]
        h, a, c = counts(b)
        import math
        full = math.prod(len(s.topo.slots[v].readings) for v in s.topo.order)
        return {"id": it["item_id"], "split": it["split"], "regime": regime(b), "hyps": len(b.hyps), "full": full,
                "classes": b.n_classes, "slots": len(amb), "hist": h, "dag": a, "canon": c, "B": blind_spot(b)}
    areas = lookup_areas(s.spec.area_name, ex, offline=True)
    if "anchor_kinds" not in it:                 # single landmark (m3, m3x)
        r = build_near(s.spec, s.tag_candidates, areas, it["anchor_kind"], ex, offline=True,
                       seed=int(hashlib.md5(it["sample_id"].encode()).hexdigest()[:8], 16), gold=it["gold"])
    else:
        r = build_near2(s.spec, s.tag_candidates, areas, tuple(it["anchor_kinds"]), ex, offline=True,
                        seed=int(hashlib.md5(it["item_id"].encode()).hexdigest()[:8], 16), gold=it["gold"])
    if r is None:
        return None
    topo, b, gold = r
    amb = [v for v in b.slots if b.splits(b.root, v)]
    h, a, c = counts(b)
    return {"id": it["item_id"], "split": it["split"], "regime": regime(b), "hyps": len(b.hyps), "classes": b.n_classes,
            "slots": len(amb), "hist": h, "dag": a, "canon": c, "B": blind_spot(b)}


def main():
    global ROWS
    ROWS = {r["sample_id"]: r for r in load_rows()}
    its = [json.loads(l) for l in open("out/ambioverpass_%s.jsonl" % BENCH)]
    if BENCH == "m2b":
        its = [it for it in its if not it["flags"]]
    with Pool(int(os.environ.get("NPROC", 48))) as pool:
        R = [r for r in pool.imap_unordered(one, its) if r]
    json.dump(R, open("out/structure_%s.json" % BENCH, "w"))
    m = lambda k: statistics.mean(r[k] for r in R)
    print(BENCH, len(R), "items | split", dict(collections.Counter(r["split"] for r in R)))
    for part in ("test", "all"):
        RR = [r for r in R if part == "all" or r["split"] == part]
        print("regimes (%s, %d)" % (part, len(RR)), dict(collections.Counter(r["regime"] for r in RR)),
              "| B > 0:", sum(r.get("B", 0) > 1e-9 for r in RR), "| readings %.1f classes %.1f" % (
                  statistics.mean(r["hyps"] for r in RR), statistics.mean(r["classes"] for r in RR)),
              "| full product %.1f" % statistics.mean(r["full"] for r in RR) if "full" in RR[0] else "")
    print("readings %.1f classes %.1f slots %.2f | slot histogram %s" % (m("hyps"), m("classes"), m("slots"),
          dict(sorted(collections.Counter(r["slots"] for r in R).items()))))
    print("states: history tree %.1f | DAG %.1f | canonical %.1f (medians %d / %d / %d)" % (
        m("hist"), m("dag"), m("canon"), statistics.median(r["hist"] for r in R), statistics.median(r["dag"] for r in R),
        statistics.median(r["canon"] for r in R)))


if __name__ == "__main__":
    main()
