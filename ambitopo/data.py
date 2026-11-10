"""AmbiOverpass rows -> samples with topology and belief.

Rows come from the old benchmark (ambiov_full_verified.jsonl / ambi_train_verified.jsonl). Each has
the clear request, the vague rewrite (one weakened slot), the gold query, the gold tag c and the
candidate tags Q (rule-built, LLM-filtered, executed once).

What is still missing for the paper (needs an LLM, see llm.py): natural-language requests in which
several slots are vague at the same time. `compound_request` is the hook for that.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .belief import Belief, execute_around, execute_hypotheses
from .overpass import Executor
from .query import AroundSpec, QuerySpec
from .readings import SLOT_NAMES, build_topology, build_topology_around, lookup_areas
from .topology import Topology

DEFAULT_SRC = os.environ.get("AMBI_SRC", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "source"))
FILES = ("ambiov_full_verified.jsonl", "ambi_train_verified.jsonl")


@dataclass
class Sample:
    sample_id: str
    x_clear: str
    x_amb: str
    verify: str                  # genuine / spurious / unusable label (Chinese strings) in the old single-slot benchmark
    spec: QuerySpec | AroundSpec
    tag_candidates: list[str]          # tag candidates (area samples) or radius candidates (around samples)
    topo: Topology | None = None
    belief: Belief | None = None
    exec_stats: dict | None = None


def load_rows(src: str = DEFAULT_SRC) -> list[dict]:
    rows = [json.loads(l) for f in FILES for l in open(os.path.join(src, f))]
    # keep rows whose gold reading executed to a non-empty result in the old verification
    rows = [r for r in rows if any(x["value"] == r["c"] and str(x.get("exec", "")).startswith("ok")
                                   for x in r["Q"])]
    extra = os.environ.get("AMBI_EXTRA_ROWS")          # test2 expansion: new OverpassNL source rows (scripts/onl_sources.py)
    if extra and os.path.exists(extra):
        rows += [json.loads(l) for l in open(extra)]
    return rows


def to_sample(row: dict) -> Sample | None:
    if str(row.get("t", "")).startswith("t2"):
        spec = AroundSpec.parse(row["gold_query"])
        if spec is None:
            return None
        return Sample(row["sample_id"], row["x_clear"], row["x_amb"], row["verify"], spec,
                      [x["value"] for x in row["Q"]])
    spec = QuerySpec.parse(row["gold_query"], main_tag=row["c"] if isinstance(row["c"], str) else None)
    if spec is None or spec.main is None:
        return None
    return Sample(row["sample_id"], row["x_clear"], row["x_amb"], row["verify"], spec,
                  [x["value"] for x in row["Q"]])


def build(sample: Sample, executor: Executor, vary=SLOT_NAMES, offline: bool = False,
          max_areas: int = 4, keep_empty: bool = False) -> Sample:
    if isinstance(sample.spec, AroundSpec):
        sample.topo = build_topology_around(sample.spec, sample.tag_candidates,
                                            vary=tuple(v if v != "area" else "dist" for v in vary))
        hyps, stats = execute_around(sample.spec, sample.topo, executor, offline=offline, keep_empty=keep_empty)
        sample.exec_stats = stats
        if hyps:
            sample.belief = Belief(sample.topo, hyps)
        return sample
    areas = lookup_areas(sample.spec.area_name, executor, max_areas=max_areas, offline=offline)
    sample.topo = build_topology(sample.spec, sample.tag_candidates, areas or [], vary=vary)
    if sample.topo is None:
        err = "area_lookup_failed" if areas is None else ("no_area_match" if not areas else "no_topology")
        sample.exec_stats = {"error": err}
        return sample
    hyps, stats = execute_hypotheses(sample.spec, sample.topo, executor, offline=offline, keep_empty=keep_empty)
    sample.exec_stats = stats
    if hyps:
        sample.belief = Belief(sample.topo, hyps)
    return sample


def compound_request(sample: Sample, slots: list[str], llm) -> str:
    """Rewrite x_clear so that every slot in `slots` becomes vague at once (AST-grounded: the LLM only
    erases the named values). Not implemented until an LLM endpoint is configured."""
    raise NotImplementedError("needs an LLM endpoint; see ambitopo/llm.py")
