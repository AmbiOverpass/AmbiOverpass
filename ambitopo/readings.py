"""Build the ambiguity topology of one sample: candidate readings per slot.

Candidate sources follow the old AmbiOverpass generator (rules first, no invented values):
  area   : every OSM area carrying the requested name                      (op: align)
  tag    : the rule-built, LLM-filtered tag candidates stored in the row   (op: align)
  type   : node / way / relation / nwr                                     (op: complete)
  filter : keep all secondary conditions, or drop one of them              (op: relax)
"""
from __future__ import annotations

from .overpass import Executor, area_lookup_query, csv_rows
from .query import ELEMENT_TYPES, AroundSpec, Filter, QuerySpec
from .topology import Reading, Slot, Topology

SLOT_NAMES = ("area", "tag", "type", "filter")


def lookup_areas(name: str, executor: Executor, max_areas: int = 4, offline: bool = False) -> list[dict] | None:
    r = executor.run(area_lookup_query(name), offline=offline)
    if not r.ok:
        return None
    areas = [dict(zip(["id", "name", "admin_level", "boundary", "place"], row)) for row in csv_rows(r.text, 5)]
    # Administrative boundaries first: that is what geocodeArea (Nominatim) usually returns, so the
    # first area stands in as the gold reading. This is a convention, flagged in the gold field.
    areas.sort(key=lambda a: (a["boundary"] != "administrative", int(a["id"])))
    return areas[:max_areas]


def build_topology(spec: QuerySpec, tag_candidates: list[str], areas: list[dict],
                   vary=SLOT_NAMES, max_drop: int = 3) -> Topology | None:
    if spec.main is None or not areas:
        return None
    main = spec.filters[spec.main]

    area_r = [Reading(a["id"], "%s [%s %s]" % (a["name"], a.get("boundary") or a.get("place") or "area", a["id"]),
                      "align") for a in areas]
    if "area" not in vary:
        area_r = area_r[:1]

    # Keep the dataset's candidate order: the superset query string (and so the cache key) follows it.
    tags = []
    for t in tag_candidates:
        f = Filter.from_tag(t)
        if f and f not in tags:
            tags.append(f)
    if main not in tags:
        tags.append(main)
    if "tag" not in vary:
        tags = [main]
    tag_r = [Reading(f, f.raw, "gold" if f == main else "align") for f in tags]
    tag_gold = tags.index(main)

    gold_type = spec.gold_type
    type_r = [Reading(t, t, "gold" if t == gold_type else "complete")
              for t in ([gold_type] + [t for t in ELEMENT_TYPES if t != gold_type])]
    if "type" not in vary:
        type_r = type_r[:1]

    filt_r = [Reading(None, "keep all conditions", "gold")]
    filt_r += [Reading(j, "drop " + f.raw, "relax") for j, f in enumerate(spec.secondary[:max_drop])]
    if "filter" not in vary:
        filt_r = filt_r[:1]

    return Topology([
        Slot("area", "toponym", area_r, gold=0),
        Slot("tag", "tag", tag_r, gold=tag_gold, parents=[("area", "validity")]),
        Slot("type", "element_type", type_r, gold=0, parents=[("tag", "candidates")]),
        Slot("filter", "dropped_condition", filt_r, gold=0, parents=[("tag", "candidates")]),
    ])


def build_topology_around(spec: AroundSpec, radius_candidates: list[str],
                          vary=("dist", "type", "filter"), max_drop: int = 3) -> Topology:
    radii = []
    for r in [spec.radius] + list(radius_candidates):
        try:
            v = str(int(float(r)))
        except ValueError:
            continue
        if v not in radii and float(v) > 0:
            radii.append(v)
    gold = radii.index(str(int(float(spec.radius))))
    if "dist" not in vary:
        radii, gold = [radii[gold]], 0
    dist_r = [Reading(v, "within %s m" % v, "gold" if j == gold else "complete") for j, v in enumerate(radii)]
    gold_type = spec.gold_type
    type_r = [Reading(t, t, "gold" if t == gold_type else "complete")
              for t in ([gold_type] + [t for t in ELEMENT_TYPES if t != gold_type])]
    if "type" not in vary:
        type_r = type_r[:1]
    filt_r = [Reading(None, "keep all conditions", "gold")]
    filt_r += [Reading(j, "drop " + f.raw, "relax") for j, f in enumerate(spec.secondary[:max_drop])]
    if "filter" not in vary:
        filt_r = filt_r[:1]
    return Topology([
        Slot("dist", "distance", dist_r, gold=gold),
        Slot("type", "element_type", type_r, gold=0, parents=[("dist", "validity")]),
        Slot("filter", "dropped_condition", filt_r, gold=0, parents=[("dist", "validity")]),
    ])
