"""Execution cost of superset execution vs one request per joint reading, measured live on the Overpass mirror.

M  (one object kind): superset = one request per candidate area; per-reading = one request per (area, tag, type,
   dropped condition). M3 (target near anchor): superset = one exact-around request per area;
   per-reading = one Overpass `around` request per (area, tag, anchor, radius, type).
Per item: number of requests (exact count), and for a random subset of at most CAP per-reading requests and all
superset requests: wall time per request and response bytes, measured with a fresh cache (no cache hits). The
per-reading total is (count) x (mean measured time). For M3 the per-reading requests also give Overpass's own
`around` answer, compared with the superset result (identical sets expected).

  python scripts/exec_cost.py M|M3 N_ITEMS [CAP]
"""
import hashlib
import json
import os
import random
import sys
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo.belief import jaccard
from ambitopo.data import build, load_rows, to_sample
from ambitopo.near import RADII, _targets, exact_query, fetch_exact
from ambitopo.overpass import Executor, csv_rows, superset_query
from ambitopo.query import ELEMENT_TYPES, Filter, QuerySpec
from ambitopo.readings import lookup_areas

GAP = 3.0
live = Executor("out/cache_timing", sleep=0.0, retries=2)
live.retry_failed = True
_last = [0.0]


def timed(q):
    g = GAP - (time.time() - _last[0])
    if g > 0:
        time.sleep(g)
    p = live._path(q)
    if os.path.exists(p):
        os.remove(p)
    t0 = time.time()
    r = live.run(q)
    _last[0] = time.time()
    return r, _last[0] - t0


def csv_ids(text, ncol=2):
    return frozenset(row[0][0] + row[1] for row in csv_rows(text, ncol))


def per_reading_m(spec, area_id, etype, tag, drop):
    conds = [tag] + [f for j, f in enumerate(spec.secondary) if j != drop]
    return '[out:csv(::type,::id;false;"\t")][timeout:90];area(id:%s)->.a;(%s%s(area.a););out;' % (
        area_id, etype, "".join(f.raw for f in conds))


def per_reading_m3(area_id, tag, anchor, radius, etype):
    return ('[out:csv(::type,::id;false;"\t")][timeout:120];area(id:%s)->.a;nwr%s(area.a)->.an;'
            '(%s%s(area.a)(around.an:%s););out;' % (area_id, anchor.raw, etype, tag.raw, radius))


def item_m(row, it, cap, rng):
    s = build(to_sample(row), Executor("out/cache"), vary=tuple(it["vary"]), offline=True, keep_empty=True)
    if s is None or s.belief is None:
        return None
    sl = s.topo.slots
    combos = list(product(*[range(len(sl[n].readings)) for n in s.topo.order]))
    order = s.topo.order
    rec = {"id": it["item_id"], "readings": len(combos), "areas": len(sl["area"].readings)}
    tags = [r.value for r in sl["tag"].readings]
    keys = []
    for f in tags + s.spec.secondary:
        if f.key not in keys:
            keys.append(f.key)
    sup = [timed(superset_query(a.value, [t.raw for t in tags], keys)) for a in sl["area"].readings]
    rec["sup"] = {"n": len(sup), "t": [t for _, t in sup], "bytes": [len(r.text) for r, _ in sup], "ok": [r.ok for r, _ in sup]}
    pick = rng.sample(combos, min(cap, len(combos)))
    per = []
    for idx in pick:
        a = dict(zip(order, idx))
        q = per_reading_m(s.spec, sl["area"].readings[a["area"]].value, sl["type"].readings[a["type"]].value,
                          sl["tag"].readings[a["tag"]].value, sl["filter"].readings[a["filter"]].value)
        r, t = timed(q)
        per.append((t, len(r.text), r.ok))
    rec["per"] = {"n": len(combos), "t": [x[0] for x in per], "bytes": [x[1] for x in per], "ok": [x[2] for x in per]}
    return rec


def item_m3(row, it, cap, rng):
    """Superset = one exact-around request per area (all anchor readings of the kind); per reading = one `around` request."""
    s = to_sample(row)
    ex = Executor("out/cache")
    areas = lookup_areas(s.spec.area_name, ex, offline=True)
    tags, main, anchors, data = fetch_exact(s.spec, s.tag_candidates, areas, it["anchor_kind"], ex, True)
    if not data:
        return None
    types = [s.spec.gold_type] + [t for t in ELEMENT_TYPES if t != s.spec.gold_type]
    keys = []
    for f in tags:
        if f.key not in keys:
            keys.append(f.key)
    combos = [c for c in product(range(len(areas)), range(len(tags)), range(len(anchors)), range(len(RADII)), range(len(types)))
              if c[0] in data]
    rec = {"id": it["item_id"], "readings": len(combos), "areas": len(areas)}
    sup = [timed(exact_query(a["id"], [f.raw for f in tags], [an.raw for an in anchors], keys)) for ai, a in enumerate(areas) if ai in data]
    rec["sup"] = {"n": len(sup), "t": [t for _, t in sup], "bytes": [len(r.text) for r, _ in sup], "ok": [r.ok for r, _ in sup]}

    def local(c):
        ai, ti, ni, di, yi = c
        rows, around = data[ai]
        return _targets(rows, tags[ti], types[yi]) & around[(ni, RADII[di])]
    nonempty = [c for c in combos if local(c)]
    rec["nonempty"] = len(nonempty)
    pick = rng.sample(nonempty, min(cap, len(nonempty)))
    per, agree = [], []
    for c in pick:
        ai, ti, ni, di, yi = c
        r, t = timed(per_reading_m3(areas[ai]["id"], tags[ti], anchors[ni], RADII[di], types[yi]))
        per.append((t, len(r.text), r.ok))
        if r.ok:
            loc, ovp = local(c), csv_ids(r.text)
            agree.append({"j": jaccard(loc, ovp), "loc": len(loc), "ovp": len(ovp), "eq": loc == ovp})
    rec["per"] = {"n": len(combos), "t": [x[0] for x in per], "bytes": [x[1] for x in per], "ok": [x[2] for x in per]}
    rec["agree"] = agree
    return rec


def main():
    bench, n = sys.argv[1], int(sys.argv[2])
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    rows = {r["sample_id"]: r for r in load_rows()}
    rng = random.Random(0)
    if bench == "M":
        its = [json.loads(l) for l in open("out/ambioverpass_m2b.jsonl")]
        its = [it for it in its if not it["flags"] and it.get("split") == "test"
               and isinstance(to_sample(rows[it["sample_id"]]).spec, QuerySpec)]
        fn = item_m
    else:
        its = [json.loads(l) for l in open("out/ambioverpass_m3.jsonl")]
        fn = item_m3
    rng.shuffle(its)
    out = open("out/exec_cost_%s.jsonl" % bench, "a")
    done = 0
    for it in its:
        if done >= n:
            break
        try:
            rec = fn(rows[it["sample_id"]], it, cap, rng)
        except Exception as e:
            print("skip", it["item_id"], repr(e)[:80], flush=True)
            continue
        if rec is None:
            continue
        out.write(json.dumps(rec) + "\n")
        out.flush()
        done += 1
        print(done, rec["id"], "readings", rec["readings"], "sup t", ["%.1f" % t for t in rec["sup"]["t"]],
              "per t mean %.1f" % (sum(rec["per"]["t"]) / max(1, len(rec["per"]["t"]))), flush=True)


if __name__ == "__main__":
    main()
