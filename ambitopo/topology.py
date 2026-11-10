"""Ambiguity topology: slots (nodes), their candidate readings, and the edges between slots.

A slot is one open decision of the query. Each slot carries:
  - readings     : the candidate values, each tagged with the rewrite operation that produced it
  - handling     : how the old AmbiOverpass taxonomy says this ambiguity type should be resolved
                   ask      -> the planner may ask the user about it
                   default  -> resolve with a default value, never ask
                   infer    -> the system infers it, never ask (users cannot answer it)
                   feedback -> only execution feedback / showing results can resolve it
  - parents      : edges (parent slot, edge type), edge type in {exists, candidates, validity}

Edges are recorded for two uses: legality (a slot whose candidates depend on an unresolved parent
cannot be asked on its own) and reporting. Validity edges act through pruning: a joint reading
whose result is empty is dropped when the belief is built.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

REWRITE_OPS = ("gold", "complete", "align", "correct", "split", "relax")
EDGE_TYPES = ("exists", "candidates", "validity")

# Default handling per slot, from the old taxonomy note (09-ambiguity-taxonomy). Each type there had
# only 4-13 samples, so these are defaults to re-validate, not findings.
DEFAULT_HANDLING = {"area": "ask", "tag": "ask", "dist": "ask", "type": "infer", "filter": "feedback"}


@dataclass(frozen=True)
class Reading:
    value: object          # area id | Filter | element type | dropped-condition index | distance
    label: str             # short human-readable form, used in questions and logs
    op: str = "gold"       # rewrite operation that produced it

    def __post_init__(self):
        assert self.op in REWRITE_OPS, self.op


@dataclass
class Slot:
    name: str
    kind: str                                   # ambiguity type, e.g. toponym / tag / element_type
    readings: list[Reading]
    gold: int | None = None                     # index of the gold reading, if known
    handling: str = ""
    parents: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if not self.handling:
            self.handling = DEFAULT_HANDLING.get(self.name, "ask")

    @property
    def ambiguous(self) -> bool:
        return len(self.readings) > 1


class Topology:
    def __init__(self, slots: list[Slot]):
        self.slots = {s.name: s for s in slots}
        self.order = self._toposort(slots)

    @staticmethod
    def _toposort(slots):
        names = [s.name for s in slots]
        deps = {s.name: {p for p, _ in s.parents if p in names} for s in slots}
        order = []
        while deps:
            ready = [n for n in names if n in deps and not deps[n]]
            if not ready:
                raise ValueError("ambiguity topology has a cycle: %s" % deps)
            for n in ready:
                order.append(n)
                del deps[n]
            for d in deps.values():
                d.difference_update(ready)
        return order

    def assignments(self):
        """All joint readings as dicts slot -> reading index (existence edges are not used yet:
        every slot in the current query shape always exists)."""
        names = self.order
        for idx in product(*[range(len(self.slots[n].readings)) for n in names]):
            yield dict(zip(names, idx))

    def askable(self, allow_all: bool = False) -> list[str]:
        return [n for n in self.order if self.slots[n].ambiguous
                and (allow_all or self.slots[n].handling == "ask")]

    def gold_assignment(self) -> dict | None:
        if any(self.slots[n].gold is None for n in self.order):
            return None
        return {n: self.slots[n].gold for n in self.order}

    def describe(self) -> str:
        lines = []
        for n in self.order:
            s = self.slots[n]
            par = ", ".join("%s(%s)" % p for p in s.parents) or "-"
            lines.append("%-6s %-13s %-8s parents=%s readings=[%s]" % (
                n, s.kind, s.handling, par, ", ".join(r.label for r in s.readings)))
        return "\n".join(lines)
