"""AmbiOverpass-M3: "target near anchor" requests (RESULTS.md, "Harder data").

Slots: area (same-name areas), tag (the target's candidate tags), anchor (competing OSM tags for one anchor kind),
dist (100 / 200 / 500 m), type (element type of the target).

Execution (exact, default): one Overpass request per area and anchor kind (per-anchor requests are read from the cache when present). It returns the union T of all target
readings with their tags, and, for every anchor reading a and radius r, the ids of T that Overpass's own
`around` selects (block markers made with `make`). A combination (t, e, a, r) is then {x in T : t, e} ∩ around(a, r)
— identical to executing `e[t](area)(around.A_a:r)` on its own, because `around` is a per-object predicate.
The earlier centre-point join (`exact=False`) is kept for reference only: it disagrees with Overpass for lines
and areas (Jaccard as low as 0.01 on long ways and relations).
"""
from __future__ import annotations

import math
import random
from itertools import product

from .belief import Belief, Hypothesis
from .overpass import Executor, csv_rows
from .query import ELEMENT_TYPES, Filter
from .topology import Reading, Slot, Topology

ANCHORS = {
    "station": ['["railway"="station"]', '["public_transport"="station"]', '["amenity"="bus_station"]'],
    "school": ['["amenity"="school"]', '["building"="school"]'],
    "park": ['["leisure"="park"]', '["landuse"="recreation_ground"]', '["leisure"="garden"]'],
    "hospital": ['["amenity"="hospital"]', '["healthcare"="hospital"]', '["amenity"="clinic"]'],
    "supermarket": ['["shop"="supermarket"]', '["shop"="convenience"]'],
}
ANCHOR_WORDS = {"station": "a station", "school": "a school", "park": "a park", "hospital": "a hospital",
                "supermarket": "a grocery shop"}
RADII = ["100", "200", "500"]
BLK = "ambitopo_blk"


def exact_query(area_id: str, target_raws: list[str], anchor_raws, keys: list[str]) -> str:
    """T with tags, then for every anchor reading the ids of T within 500 / 200 / 100 m of its objects (each radius
    searched inside the previous, larger one). `anchor_raws`: one raw filter or a list (one request per anchor kind)."""
    if isinstance(anchor_raws, str):
        anchor_raws = [anchor_raws]
    cols = ",".join(["::type", "::id", '"%s"' % BLK] + ['"%s"' % k for k in keys])
    q = ['[out:csv(%s;false;"\t")][timeout:300];area(id:%s)->.a;' % (cols, area_id),
         "(%s)->.T;" % "".join("nwr%s(area.a);" % t for t in target_raws),
         'make m %s="T";out;.T out;' % BLK]
    for ai, a in enumerate(anchor_raws):
        q.append("nwr%s(area.a)->.A;" % a)
        prev = "T"
        for r in sorted(RADII, key=float, reverse=True):
            q.append('nwr.%s(around.A:%s)->.R%s;make m %s="%d|%s";out;.R%s out ids;' % (prev, r, r, BLK, ai, r, r))
            prev = "R" + r
    return "".join(q)


def parse_exact(text: str, keys: list[str]):
    """-> rows of T (type, eid, tags) and {(anchor index, radius): frozenset of eids}"""
    rows, around, cur, buf = [], {}, None, []
    for row in csv_rows(text, 3 + len(keys)):
        if row[0] == "m":
            if cur is not None and cur != "T":
                around[cur] = frozenset(buf)
            cur, buf = row[2], []
            if cur != "T":
                ai, r = cur.split("|")
                cur = (int(ai), r)
            continue
        eid = row[0][0] + row[1]
        if cur == "T":
            rows.append((row[0], eid, dict(zip(keys, row[3:]))))
        else:
            buf.append(eid)
    if cur is not None and cur != "T":
        around[cur] = frozenset(buf)
    return rows, around


def fetch_exact(spec, tag_candidates, areas, anchor_kind, executor: Executor, offline=True):
    main = spec.filters[spec.main]
    tags = []
    for t in tag_candidates:
        f = Filter.from_tag(t)
        if f and f not in tags:
            tags.append(f)
    if main not in tags:
        tags.append(main)
    anchors = [Filter.parse(a) for a in ANCHORS[anchor_kind]]
    keys = []
    for f in tags:
        if f.key not in keys:
            keys.append(f.key)
    T = [f.raw for f in tags]

    def combined(a, off):
        res = executor.run(exact_query(a["id"], T, [f.raw for f in anchors], keys), offline=off)
        if res.ok:
            rows, around = parse_exact(res.text, keys)
            if len(around) == len(anchors) * len(RADII):
                return rows, around
        return None

    def per_anchor(a, off):
        rows, around = None, {}
        for ni, an in enumerate(anchors):
            res = executor.run(exact_query(a["id"], T, an.raw, keys), offline=off)
            if not res.ok:
                return None
            r, ar = parse_exact(res.text, keys)
            if len(ar) != len(RADII):
                return None
            rows = r
            around.update({(ni, rad): ids for (_, rad), ids in ar.items()})
        return rows, around

    data = {}
    for ai, a in enumerate(areas):
        got = combined(a, True) or per_anchor(a, True)        # cache first (either request form)
        if got is None and not offline:
            got = combined(a, False) or per_anchor(a, False)
        if got is not None:
            data[ai] = got
    return tags, main, anchors, data


def _gold(hyps, order, tags, main, rng, hint):
    ok = [h.assign for h in hyps if h.assign[order.index("area")] == 0 and h.assign[order.index("tag")] == tags.index(main)
          and h.assign[order.index("type")] == 0 and h.result]
    if not ok:
        return None
    if hint is not None:
        return tuple(hint) if tuple(hint) in set(ok) else None
    return rng.choice(sorted(ok))


def near_query(area_id: str, raws: list[str], keys: list[str]) -> str:
    cols = ",".join(["::type", "::id", "::lat", "::lon"] + ['"%s"' % k for k in keys])
    union = "".join("nwr%s(area.a);" % t for t in raws)
    return '[out:csv(%s;false;"\t")][timeout:120];area(id:%s)->.a;(%s);out center;' % (cols, area_id, union)


def _close(targets, anchors, r):
    """ids of targets within r metres of any anchor (grid hashing, haversine check)."""
    if not anchors or not targets:
        return frozenset()
    cell = r / 111000.0
    grid = {}
    for lat, lon in anchors:
        grid.setdefault((int(lat // cell), int(lon // cell)), []).append((lat, lon))
    out = set()
    for eid, lat, lon in targets:
        ci, cj = int(lat // cell), int(lon // cell)
        hit = False
        for di in (-1, 0, 1):
            for dj in range(-2, 3):           # longitude cells are narrower than latitude cells away from the equator
                for alat, alon in grid.get((ci + di, cj + dj), ()):
                    dl = math.radians(alat - lat)
                    dn = math.radians(alon - lon)
                    a = math.sin(dl / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(alat)) * math.sin(dn / 2) ** 2
                    if 2 * 6371000 * math.asin(math.sqrt(min(1.0, a))) <= r:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            out.add(eid)
    return frozenset(out)


def build_near(spec, tag_candidates, areas, anchor_kind, executor: Executor, offline=False, seed=0, exact=True, gold=None,
               keep_empty=False):
    """Returns (topology, belief, gold tuple) or None. `gold`: keep this gold assignment (None if its result is empty)."""
    if exact:
        return _build_near_exact(spec, tag_candidates, areas, anchor_kind, executor, offline, seed, gold, keep_empty)
    main = spec.filters[spec.main]
    tags = []
    for t in tag_candidates:
        f = Filter.from_tag(t)
        if f and f not in tags:
            tags.append(f)
    if main not in tags:
        tags.append(main)
    anchors = [Filter.parse(a) for a in ANCHORS[anchor_kind]]
    gold_type = spec.gold_type
    types = [gold_type] + [t for t in ELEMENT_TYPES if t != gold_type]
    slots = [
        Slot("area", "area", [Reading(a["id"], "%s [%s]" % (a["name"], a.get("admin_level") or a.get("place") or "area"), "align")
                              for a in areas], gold=0),
        Slot("tag", "tag", [Reading(f, f.raw, "gold" if f == main else "align") for f in tags], gold=tags.index(main),
             parents=[("area", "validity")]),
        Slot("anchor", "anchor", [Reading(f, f.raw, "align") for f in anchors], gold=0, parents=[("area", "validity")]),
        Slot("dist", "distance", [Reading(r, r + " m", "complete") for r in RADII], gold=0, parents=[("anchor", "exists")]),
        Slot("type", "element_type", [Reading(t, t, "complete") for t in types], gold=0, parents=[("tag", "candidates")]),
    ]
    topo = Topology(slots)
    keys = []
    for f in tags + anchors:
        if f.key not in keys:
            keys.append(f.key)
    data = {}
    for ai, a in enumerate(areas):
        res = executor.run(near_query(a["id"], [f.raw for f in tags + anchors], keys), offline=offline)
        if not res.ok:
            continue
        rows = []
        for row in csv_rows(res.text, 4 + len(keys)):
            try:
                lat, lon = float(row[2]), float(row[3])
            except ValueError:
                continue
            rows.append((row[0], row[0][0] + row[1], lat, lon, dict(zip(keys, row[4:]))))
        data[ai] = rows
    if not data:
        return None
    order = topo.order
    hyps = []
    for idx in product(*[range(len(topo.slots[n].readings)) for n in order]):
        a = dict(zip(order, idx))
        if a["area"] not in data:
            continue
        rows = data[a["area"]]
        t = tags[a["tag"]]
        et = types[a["type"]]
        tg = [(eid, lat, lon) for typ, eid, lat, lon, kv in rows if t.match(kv) and (et == "nwr" or typ == et)]
        an = [(lat, lon) for typ, eid, lat, lon, kv in rows if anchors[a["anchor"]].match(kv)]
        s = _close(tg, an, float(RADII[a["dist"]]))
        if s:
            hyps.append(Hypothesis(idx, s))
    if not hyps:
        return None
    b = Belief(topo, hyps)
    # gold: area 0, the sample's tag, its element type; anchor reading and radius at random among non-empty ones
    rng = random.Random(seed)
    ok = [h.assign for h in hyps if h.assign[order.index("area")] == 0 and h.assign[order.index("tag")] == tags.index(main)
          and h.assign[order.index("type")] == 0]
    if not ok:
        return None
    gold = rng.choice(sorted(ok))
    return topo, b, gold


def _fetch(spec, tag_candidates, areas, anchor_kind, executor, offline):
    """The cached single-anchor request of build_near, per area: rows (type, eid, lat, lon, tags)."""
    main = spec.filters[spec.main]
    tags = []
    for t in tag_candidates:
        f = Filter.from_tag(t)
        if f and f not in tags:
            tags.append(f)
    if main not in tags:
        tags.append(main)
    anchors = [Filter.parse(a) for a in ANCHORS[anchor_kind]]
    keys = []
    for f in tags + anchors:
        if f.key not in keys:
            keys.append(f.key)
    data = {}
    for ai, a in enumerate(areas):
        res = executor.run(near_query(a["id"], [f.raw for f in tags + anchors], keys), offline=offline)
        if not res.ok:
            continue
        rows = []
        for row in csv_rows(res.text, 4 + len(keys)):
            try:
                lat, lon = float(row[2]), float(row[3])
            except ValueError:
                continue
            rows.append((row[0], row[0][0] + row[1], lat, lon, dict(zip(keys, row[4:]))))
        data[ai] = rows
    return tags, main, anchors, data


def _slots1(areas, tags, main, anchors, types):
    return [
        Slot("area", "area", [Reading(a["id"], "%s [%s]" % (a["name"], a.get("admin_level") or a.get("place") or "area"), "align")
                              for a in areas], gold=0),
        Slot("tag", "tag", [Reading(f, f.raw, "gold" if f == main else "align") for f in tags], gold=tags.index(main),
             parents=[("area", "validity")]),
        Slot("anchor", "anchor", [Reading(f, f.raw, "align") for f in anchors], gold=0, parents=[("area", "validity")]),
        Slot("dist", "distance", [Reading(r, r + " m", "complete") for r in RADII], gold=0, parents=[("anchor", "exists")]),
        Slot("type", "element_type", [Reading(t, t, "complete") for t in types], gold=0, parents=[("tag", "candidates")]),
    ]


def _targets(rows, t, et):
    return frozenset(eid for typ, eid, kv in rows if t.match(kv) and (et == "nwr" or typ == et))


def _build_near_exact(spec, tag_candidates, areas, anchor_kind, executor, offline, seed, gold, keep_empty=False):
    tags, main, anchors, data = fetch_exact(spec, tag_candidates, areas, anchor_kind, executor, offline)
    if len(data) < len(areas):          # an area whose request never succeeded: the readings would be incomplete
        return None
    types = [spec.gold_type] + [t for t in ELEMENT_TYPES if t != spec.gold_type]
    topo = Topology(_slots1(areas, tags, main, anchors, types))
    order = topo.order
    hyps, tcache = [], {}
    for idx in product(*[range(len(topo.slots[n].readings)) for n in order]):
        a = dict(zip(order, idx))
        if a["area"] not in data:
            continue
        rows, around = data[a["area"]]
        key = (a["area"], a["tag"], a["type"])
        if key not in tcache:
            tcache[key] = _targets(rows, tags[a["tag"]], types[a["type"]])
        s = tcache[key] & around[(a["anchor"], RADII[a["dist"]])]
        if s or keep_empty:          # keep_empty: ablation of validity pruning
            hyps.append(Hypothesis(idx, s))
    if not hyps:
        return None
    g = _gold(hyps, order, tags, main, random.Random(seed), gold)
    if g is None:
        return None
    return topo, Belief(topo, hyps), g


def build_near2(spec, tag_candidates, areas, kinds, executor: Executor, offline=True, seed=0, max_hyps=60000, exact=True, gold=None,
                keep_empty=False):
    if exact:
        return _build_near2_exact(spec, tag_candidates, areas, kinds, executor, offline, seed, max_hyps, gold, keep_empty)
    return _build_near2_centre(spec, tag_candidates, areas, kinds, executor, offline, seed, max_hyps)


def _slots2(areas, tags, main, an1, an2, types):
    return [
        Slot("area", "area", [Reading(a["id"], "%s [%s]" % (a["name"], a.get("admin_level") or a.get("place") or "area"), "align")
                              for a in areas], gold=0),
        Slot("tag", "tag", [Reading(f, f.raw, "gold" if f == main else "align") for f in tags], gold=tags.index(main),
             parents=[("area", "validity")]),
        Slot("anchor1", "anchor", [Reading(f, f.raw, "align") for f in an1], gold=0, parents=[("area", "validity")]),
        Slot("dist1", "distance", [Reading(r, r + " m", "complete") for r in RADII], gold=0, parents=[("anchor1", "exists")]),
        Slot("anchor2", "anchor", [Reading(f, f.raw, "align") for f in an2], gold=0, parents=[("area", "validity")]),
        Slot("dist2", "distance", [Reading(r, r + " m", "complete") for r in RADII], gold=0, parents=[("anchor2", "exists")]),
        Slot("type", "element_type", [Reading(t, t, "complete") for t in types], gold=0, parents=[("tag", "candidates")]),
    ]


def _build_near2_exact(spec, tag_candidates, areas, kinds, executor, offline, seed, max_hyps, gold, keep_empty=False):
    k1, k2 = kinds
    tags, main, an1, d1 = fetch_exact(spec, tag_candidates, areas, k1, executor, offline)
    _, _, an2, d2 = fetch_exact(spec, tag_candidates, areas, k2, executor, offline)
    ok_areas = [ai for ai in d1 if ai in d2]
    if len(ok_areas) < len(areas):      # incomplete readings: drop the item
        return None
    types = [spec.gold_type] + [t for t in ELEMENT_TYPES if t != spec.gold_type]
    topo = Topology(_slots2(areas, tags, main, an1, an2, types))
    order = topo.order
    hyps, tcache = [], {}
    for idx in product(*[range(len(topo.slots[n].readings)) for n in order]):
        a = dict(zip(order, idx))
        if a["area"] not in ok_areas:
            continue
        key = (a["area"], a["tag"], a["type"])
        if key not in tcache:
            tcache[key] = _targets(d1[a["area"]][0], tags[a["tag"]], types[a["type"]])
        s = tcache[key] & d1[a["area"]][1][(a["anchor1"], RADII[a["dist1"]])] & d2[a["area"]][1][(a["anchor2"], RADII[a["dist2"]])]
        if s or keep_empty:
            hyps.append(Hypothesis(idx, s))
    if not hyps or len(hyps) > max_hyps * (5 if keep_empty else 1):
        return None
    g = _gold(hyps, order, tags, main, random.Random(seed), gold)
    if g is None:
        return None
    return topo, Belief(topo, hyps), g


def _build_near2_centre(spec, tag_candidates, areas, kinds, executor: Executor, offline=True, seed=0, max_hyps=60000):
    """AmbiOverpass-M4: targets within r1 of an anchor of kind 1 AND within r2 of an anchor of kind 2.
    Reuses the two single-anchor requests (one per kind and area); the join is local.
    Slots: area, tag, anchor1, dist1, anchor2, dist2, type. Returns (topology, belief, gold) or None."""
    k1, k2 = kinds
    tags, main, an1, d1 = _fetch(spec, tag_candidates, areas, k1, executor, offline)
    _, _, an2, d2 = _fetch(spec, tag_candidates, areas, k2, executor, offline)
    gold_type = spec.gold_type
    types = [gold_type] + [t for t in ELEMENT_TYPES if t != gold_type]
    slots = [
        Slot("area", "area", [Reading(a["id"], "%s [%s]" % (a["name"], a.get("admin_level") or a.get("place") or "area"), "align")
                              for a in areas], gold=0),
        Slot("tag", "tag", [Reading(f, f.raw, "gold" if f == main else "align") for f in tags], gold=tags.index(main),
             parents=[("area", "validity")]),
        Slot("anchor1", "anchor", [Reading(f, f.raw, "align") for f in an1], gold=0, parents=[("area", "validity")]),
        Slot("dist1", "distance", [Reading(r, r + " m", "complete") for r in RADII], gold=0, parents=[("anchor1", "exists")]),
        Slot("anchor2", "anchor", [Reading(f, f.raw, "align") for f in an2], gold=0, parents=[("area", "validity")]),
        Slot("dist2", "distance", [Reading(r, r + " m", "complete") for r in RADII], gold=0, parents=[("anchor2", "exists")]),
        Slot("type", "element_type", [Reading(t, t, "complete") for t in types], gold=0, parents=[("tag", "candidates")]),
    ]
    topo = Topology(slots)
    areas_ok = [ai for ai in d1 if ai in d2]
    if not areas_ok:
        return None
    order = topo.order
    sizes = [len(topo.slots[n].readings) for n in order]
    hyps = []
    cache = {}
    for idx in product(*[range(k) for k in sizes]):
        a = dict(zip(order, idx))
        if a["area"] not in areas_ok:
            continue
        key = (a["area"], a["tag"], a["type"])
        if key not in cache:
            t, et = tags[a["tag"]], types[a["type"]]
            cache[key] = [(eid, lat, lon) for typ, eid, lat, lon, kv in d1[a["area"]] if t.match(kv) and (et == "nwr" or typ == et)]
        tg = cache[key]
        if not tg:
            continue
        k_1 = (1, a["area"], a["tag"], a["type"], a["anchor1"], a["dist1"])
        if k_1 not in cache:
            anc = [(lat, lon) for typ, eid, lat, lon, kv in d1[a["area"]] if an1[a["anchor1"]].match(kv)]
            cache[k_1] = _close(tg, anc, float(RADII[a["dist1"]]))
        s1 = cache[k_1]
        if not s1:
            continue
        k_2 = (2, a["area"], a["tag"], a["type"], a["anchor2"], a["dist2"])
        if k_2 not in cache:
            anc = [(lat, lon) for typ, eid, lat, lon, kv in d2[a["area"]] if an2[a["anchor2"]].match(kv)]
            cache[k_2] = _close(tg, anc, float(RADII[a["dist2"]]))
        s = s1 & cache[k_2]
        if s:
            hyps.append(Hypothesis(idx, s))
    if not hyps or len(hyps) > max_hyps:
        return None
    b = Belief(topo, hyps)
    rng = random.Random(seed)
    ok = [h.assign for h in hyps if h.assign[order.index("area")] == 0 and h.assign[order.index("tag")] == tags.index(main)
          and h.assign[order.index("type")] == 0]
    if not ok:
        return None
    gold = rng.choice(sorted(ok))
    return topo, b, gold
