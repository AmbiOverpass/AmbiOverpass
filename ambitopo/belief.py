"""Belief over joint readings.

A hypothesis is one joint reading (slot -> reading index) together with the set of OSM elements it
returns. Hypotheses whose result sets are near-identical (Jaccard >= 0.8, complete linkage) form one
result class: the user is satisfied by any member of the right class. Everything downstream
(question value, planning, success) is defined on classes, not on raw readings.

All methods take `alive`, a sorted tuple of hypothesis indices still consistent with what has been
observed, so states can be cached and merged (two question orders that leave the same alive set are
the same state).

W(alive)       = total prior weight of pairs that would give different results (EC2 edge weight)
gain(alive, v) = expected drop of W from learning slot v
"""
from __future__ import annotations

import collections
import os
import random
from dataclasses import dataclass
from itertools import product

from .overpass import Executor, around_query, csv_rows, superset_query
from .query import QuerySpec
from .topology import Topology


@dataclass(frozen=True)
class Hypothesis:
    assign: tuple            # reading index per slot, in topology order
    result: frozenset        # element ids, e.g. "n123", "w45"


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def cluster(results: list[frozenset], threshold: float = 0.8) -> list[int]:
    """Greedy complete linkage: join the first class whose every member is within the threshold."""
    labels, members = [], []
    for s in results:
        for k, mem in enumerate(members):
            if all(jaccard(s, m) >= threshold for m in mem):
                mem.append(s)
                labels.append(k)
                break
        else:
            members.append([s])
            labels.append(len(members) - 1)
    return labels


def execute_hypotheses(spec: QuerySpec, topo: Topology, executor: Executor, offline: bool = False,
                       keep_empty: bool = False):
    """One superset request per area reading; the rest is split locally. Empty results are dropped
    (validity edge: a reading that returns nothing in this area is not a reading); keep_empty=True keeps
    them (ablation of the validity edge)."""
    slots = topo.slots
    tags = [r.value for r in slots["tag"].readings]
    keys = []
    for f in tags + spec.secondary:
        if f.key not in keys:
            keys.append(f.key)
    stats = {"areas_ok": 0, "areas_failed": 0, "errors": []}
    elems = {}
    for ai, ar in enumerate(slots["area"].readings):
        res = executor.run(superset_query(ar.value, [t.raw for t in tags], keys), offline=offline)
        if not res.ok:
            stats["areas_failed"] += 1
            stats["errors"].append(res.error)
            continue
        stats["areas_ok"] += 1
        elems[ai] = [(row[0], row[0][0] + row[1], dict(zip(keys, row[2:])))
                     for row in csv_rows(res.text, 2 + len(keys))]
    order = topo.order
    hyps = []
    for idx in product(*[range(len(slots[n].readings)) for n in order]):
        a = dict(zip(order, idx))
        if a["area"] not in elems:
            continue
        tag = slots["tag"].readings[a["tag"]].value
        etype = slots["type"].readings[a["type"]].value
        drop = slots["filter"].readings[a["filter"]].value
        sec = [f for j, f in enumerate(spec.secondary) if j != drop]
        s = frozenset(eid for t, eid, tv in elems[a["area"]]
                      if (etype == "nwr" or t == etype) and tag.match(tv) and all(f.match(tv) for f in sec))
        if s or keep_empty:
            hyps.append(Hypothesis(idx, s))
    return hyps, stats


def execute_around(spec, topo: Topology, executor: Executor, offline: bool = False, keep_empty: bool = False):
    """Distance samples: one request per candidate radius; type and dropped conditions split locally."""
    slots = topo.slots
    keys = []
    for f in spec.filters:
        if f.key not in keys:
            keys.append(f.key)
    stats = {"areas_ok": 0, "areas_failed": 0, "errors": []}
    elems = {}
    for di, dr in enumerate(slots["dist"].readings):
        res = executor.run(around_query(dr.value, spec.lat, spec.lon, spec.filters[0].raw, keys), offline=offline)
        if not res.ok:
            stats["areas_failed"] += 1
            stats["errors"].append(res.error)
            continue
        stats["areas_ok"] += 1
        elems[di] = [(row[0], row[0][0] + row[1], dict(zip(keys, row[2:])))
                     for row in csv_rows(res.text, 2 + len(keys))]
    order = topo.order
    hyps = []
    for idx in product(*[range(len(slots[n].readings)) for n in order]):
        a = dict(zip(order, idx))
        if a["dist"] not in elems:
            continue
        etype = slots["type"].readings[a["type"]].value
        drop = slots["filter"].readings[a["filter"]].value
        sec = [f for j, f in enumerate(spec.secondary) if j != drop]
        s = frozenset(eid for t, eid, tv in elems[a["dist"]]
                      if (etype == "nwr" or t == etype) and all(f.match(tv) for f in sec))
        if s or keep_empty:
            hyps.append(Hypothesis(idx, s))
    return hyps, stats


class Belief:
    def __init__(self, topo: Topology, hyps: list[Hypothesis], prior: list[float] | None = None,
                 threshold: float | None = None):
        if threshold is None:
            threshold = float(os.environ.get("AMBITOPO_JACCARD", "1.0"))   # 1.0 = identical result sets, the EX criterion
        if not hyps:
            raise ValueError("empty belief")
        self.topo = topo
        self.slots = list(topo.order)
        self.hyps = hyps
        w = prior or [1.0] * len(hyps)
        z = sum(w)
        self.w = [x / z for x in w]
        self.cls = cluster([h.result for h in hyps], threshold)
        self.n_classes = len(set(self.cls))
        # ties between equally likely classes are broken in a fixed pseudo-random order: class ids follow
        # the reading order, and the gold reading tends to come first, so "lowest id" would favour it
        perm = list(range(self.n_classes))
        random.Random(len(hyps) * 1000003 + self.n_classes).shuffle(perm)
        self._tiebreak = perm
        self.root = tuple(range(len(hyps)))
        self._cache = {}

    # -- basic quantities -------------------------------------------------------------------
    def value(self, i: int, slot: str) -> int:
        return self.hyps[i].assign[self.slots.index(slot)]

    def mass(self, alive) -> float:
        return sum(self.w[i] for i in alive)

    def class_mass(self, alive) -> dict:
        key = ("cm", alive)
        if key not in self._cache:
            m = collections.defaultdict(float)
            for i in alive:
                m[self.cls[i]] += self.w[i]
            self._cache[key] = dict(m)
        return self._cache[key]

    def W(self, alive) -> float:
        cm = self.class_mass(alive)
        tot = sum(cm.values())
        return max(0.0, tot * tot - sum(x * x for x in cm.values()))

    def certain(self, alive) -> bool:
        return len(self.class_mass(alive)) == 1

    def map_class(self, alive) -> int:
        cm = self.class_mass(alive)
        return max(cm, key=lambda c: (cm[c], self._tiebreak[c]))

    def acc_stop(self, alive) -> float:
        """Probability that submitting the most likely class now is correct."""
        cm = self.class_mass(alive)
        return max(cm.values()) / sum(cm.values())

    def representative(self, alive, cls: int) -> int:
        """Most probable reading of the class; ties in a fixed pseudo-random order (not the reading order,
        in which the gold reading tends to come first)."""
        if not hasattr(self, "_rep_order"):
            perm = list(range(len(self.hyps)))
            random.Random(len(self.hyps) * 7919 + 17).shuffle(perm)
            self._rep_order = {i: r for r, i in enumerate(perm)}
        return max((i for i in alive if self.cls[i] == cls), key=lambda i: (self.w[i], self._rep_order[i]))

    # -- questions --------------------------------------------------------------------------
    def split(self, alive, slot: str) -> dict:
        key = ("sp", alive, slot)
        if key not in self._cache:
            g = collections.defaultdict(list)
            k = self.slots.index(slot)
            for i in alive:
                g[self.hyps[i].assign[k]].append(i)
            self._cache[key] = {v: tuple(x) for v, x in g.items()}
        return self._cache[key]

    def splits(self, alive, slot: str) -> bool:
        return len(self.split(alive, slot)) > 1

    def gain(self, alive, slot: str) -> float:
        tot = self.mass(alive)
        return self.W(alive) - sum(self.mass(h) / tot * self.W(h) for h in self.split(alive, slot).values())

    def matters(self, alive, slot: str) -> bool:
        """Two readings in `alive` that differ only in `slot` fall in different result classes."""
        key = ("mt", alive, slot)
        if key not in self._cache:
            k = self.slots.index(slot)
            groups = {}
            for i in alive:
                a = self.hyps[i].assign
                groups.setdefault(a[:k] + a[k + 1:], set()).add(self.cls[i])
            self._cache[key] = any(len(c) > 1 for c in groups.values())
        return self._cache[key]

    def closing_roots(self, alive, slots):
        """Slots that still matter and that no other undecided, still-mattering slot can make irrelevant
        (the roots of the closing graph at `alive`); all mattering slots if the graph has no root."""
        amb = [v for v in slots if self.matters(alive, v)]
        closed = set()
        for u in amb:
            for h in self.split(alive, u).values():
                for v in amb:
                    if v != u and v not in closed and not self.matters(h, v):
                        closed.add(v)
        return [v for v in amb if v not in closed] or amb

    def canon(self, alive):
        """Canonical form of a state: hypotheses projected onto the slots still undecided in `alive`,
        readings and classes relabelled in order of appearance, masses normalised. Two states with the
        same canonical form are the same decision problem up to renaming, so they have equal value."""
        key = ("cn", alive)
        if key in self._cache:
            return self._cache[key]
        idx = [k for k, v in enumerate(self.slots) if len({self.hyps[i].assign[k] for i in alive}) > 1]
        relab = []
        for k in idx:
            vals = sorted({self.hyps[i].assign[k] for i in alive})
            relab.append({x: j for j, x in enumerate(vals)})
        tot = self.mass(alive)
        rows = sorted((tuple(relab[j][self.hyps[i].assign[k]] for j, k in enumerate(idx)), self.cls[i],
                       round(self.w[i] / tot, 9)) for i in alive)
        cmap = {}
        out = tuple((p, cmap.setdefault(c, len(cmap)), w) for p, c, w in rows)
        val = (tuple(self.slots[k] for k in idx), out)
        self._cache[key] = val
        return val

    def observe(self, alive, slot: str, reading: int):
        return self.split(alive, slot).get(reading, ())

    def after_show(self, alive, cls: int, accepted: bool):
        """The user saw the result of class `cls` and accepted or rejected it."""
        return tuple(i for i in alive if (self.cls[i] == cls) == accepted)

    def summary(self) -> dict:
        return {"hypotheses": len(self.hyps), "classes": self.n_classes,
                "exact_sets": len({h.result for h in self.hyps}),
                "ambiguous_slots": [s for s in self.slots if self.splits(self.root, s)]}
