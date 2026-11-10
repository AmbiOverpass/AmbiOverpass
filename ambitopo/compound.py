"""Multi-slot vague requests (the benchmark extension): rewrite a clear request so that several
chosen slots become vague at once.

AST-grounded, as in the old AmbiOverpass generator: we know exactly which slots are weakened and
what their gold values are; the LLM only erases those values from the sentence. Slots that are not
weakened stay as specific as in the clear request, so a compound sample's belief varies only the
weakened slots (build(..., vary=weakened)).
"""
from __future__ import annotations

import re

from .llm import LLM

SYSTEM = """You simulate a non-expert user asking for map data.
You get a precise request and a list of things to make vague. Produce ONE rewritten request that a
real person might type.

Hard rules:
- Same subject as the precise request. Examples in the instructions show the STYLE of change only.
- Make vague ONLY the listed parts; everything else stays exactly as specific as before.
- Never mention OSM tags, keys, or query syntax (no 'amenity=', no 'tagged as', no 'node', no 'way',
  no 'relation'), never copy ids or coordinate lists.
- One sentence, natural and casual. Output only the rewritten request."""

INSTRUCTIONS = {
    "area": "Mention the place only by its bare name, dropping any country, region or 'city of' "
            "qualifier that would tell it apart from other places with the same name.",
    "tag": "Describe the KIND OF OBJECT ({value}) in vague everyday words that could fit several "
           "related kinds of objects, e.g. 'charging stations' -> 'places to power up my car'.",
    "type": "Make it unclear whether the user wants points, outlines/areas or larger groupings: drop "
            "words that signal the geometry (e.g. 'buildings' -> 'these places', 'roads' -> 'them').",
    "filter": "Silently leave out this one condition: {value}. Do not signal that anything is missing.",
    "dist": "Replace the exact distance ({value} m) with a vague word such as 'near' or 'close to'.",
}

_STOP = set("a an the of in on at to for and or with within around near by is are how many what where "
            "which show me all any there some that this these those it its from about more most best top "
            "new".split())


def _words(t):
    return {w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in _STOP}


def instructions(sample, slots: list[str]) -> list[str]:
    topo = sample.topo
    out = []
    for s in slots:
        sl = topo.slots[s]
        gold = sl.readings[sl.gold].value if sl.gold is not None else None
        if s == "tag":
            value = "%s = %s" % (gold.key, gold.val)
        elif s == "filter":
            sec = sample.spec.secondary
            value = sec[0].raw if sec else ""
        elif s == "dist":
            value = gold
        else:
            value = ""
        out.append(INSTRUCTIONS[s].format(value=value))
    return out


def quality_flags(clear: str, vague: str, sample, slots: list[str]) -> list[str]:
    bad = []
    t = vague
    if re.search(r"\b(amenity|highway|building|landuse|shop|tourism|leisure|craft|historic|railway)\s*[=\[]", t, re.I):
        bad.append("tag syntax left")
    if re.search(r"\btagged\s+as\b|\btags?\s*[:=]", t, re.I):
        bad.append("tag jargon left")
    if re.search(r"(?<![-\w])(nodes?|ways?|relations?|nwr)(?![-\w])", t, re.I):
        bad.append("geometry jargon left")
    if re.search(r"\[out:|out\s+count|;", t):
        bad.append("query syntax left")
    if t.strip().lower() == clear.strip().lower():
        bad.append("same as clear request")
    if len(t) < 10:
        bad.append("too short")
    topo = sample.topo
    if "tag" in slots:
        g = topo.slots["tag"].readings[topo.slots["tag"].gold].value
        if g.val and g.val.replace("_", " ").lower() in t.lower():
            bad.append("tag value still named")
    if "dist" in slots:
        g = topo.slots["dist"].readings[topo.slots["dist"].gold].value
        if re.search(r"\b%s\s*(m|meters?|metres?)\b" % re.escape(str(g)), t, re.I):
            bad.append("distance still given")
    if "tag" not in slots and "type" not in slots and _words(clear) and not (_words(t) & _words(clear)):
        bad.append("topic drift")
    return bad


def rewrite(sample, slots: list[str], llm: LLM) -> dict:
    ins = instructions(sample, slots)
    user = "Precise request: %s\nMake vague:\n%s\nRewritten request:" % (
        sample.x_clear, "\n".join("- " + i for i in ins))
    vague = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], temperature=0.7)
    vague = vague.strip().strip('"').splitlines()[0] if vague.strip() else ""
    return {"sample_id": sample.sample_id, "slots": slots, "x_clear": sample.x_clear,
            "x_vague": vague, "flags": quality_flags(sample.x_clear, vague, sample, slots)}
