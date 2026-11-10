"""OverpassQL in the one shape we vary: a single statement scoped by a named area.

    {{geocodeArea:"X"}}->.searchArea; (TYPE[f1][f2]...(area.searchArea);); out ...;

QuerySpec keeps the parts that can be ambiguous (area name, element type, filters, which filter is
the main tag) and renders a concrete query for any choice of readings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

GEO = re.compile(r'\{\{\s*geocodeArea\s*:\s*"?([^"}\)]+?)"?\s*\}\}')
STMT = re.compile(r'\b(node|way|relation|nwr|nw|nr|wr)((?:\[[^\]]*\])+)\(area\.searchArea\)')
UNSUPPORTED = re.compile(r'foreach|is_in|map_to_area|make|convert|around|\(\s*\.\w+\s*;\s*-')
_FQ = re.compile(r'^\[\s*(!)?\s*"?([^"!~=\[\]]+?)"?\s*(?:(!=|=|!~|~)\s*"(.*)"\s*(,\s*i)?)?\s*\]$')
_FU = re.compile(r'^\[\s*(!)?\s*([\w:.-]+)\s*(?:(!=|=|!~|~)\s*([^\],"]*?)\s*(,\s*i)?)?\s*\]$')

ELEMENT_TYPES = ("nwr", "node", "way", "relation")


@dataclass(frozen=True)
class Filter:
    """One tag condition, e.g. ["amenity"="cafe"], ["name"], ["access"!~"private"]."""
    key: str
    op: str                  # has | !has | = | != | ~ | !~
    val: str | None = None
    ci: bool = False

    @staticmethod
    def parse(text: str) -> Filter | None:
        m = _FQ.match(text) or _FU.match(text)
        if not m or m.group(2).strip().startswith("~"):
            return None
        neg, key, op, val, ci = m.groups()
        if op is None:
            op = "!has" if neg else "has"
        return Filter(key.strip(), op, val, bool(ci))

    @staticmethod
    def from_tag(tag: str) -> Filter | None:
        """'k=v' -> ["k"="v"]; other shapes are not used as tag readings."""
        if "=" not in tag or any(c in tag for c in "~!"):
            return None
        k, v = tag.split("=", 1)
        return Filter(k.strip(), "=", v.strip())

    @property
    def raw(self) -> str:
        k = self.key.replace('"', '\\"')
        if self.op == "has":
            return '["%s"]' % k
        if self.op == "!has":
            return '[!"%s"]' % k
        v = (self.val or "").replace('"', '\\"')
        return '["%s"%s"%s"%s]' % (k, self.op, v, ",i" if self.ci else "")

    def match(self, tags: dict) -> bool:
        v = tags.get(self.key, "") or ""
        if self.op == "has":
            return v != ""
        if self.op == "!has":
            return v == ""
        if self.op == "=":
            return v == self.val
        if self.op == "!=":
            return v != self.val
        try:
            hit = v != "" and re.search(self.val or "", v, re.I if self.ci else 0) is not None
        except re.error:
            return False
        return hit if self.op == "~" else not hit


@dataclass
class QuerySpec:
    area_name: str
    types: tuple[str, ...]
    filters: list[Filter]
    main: int | None = None          # index of the main tag filter in `filters`
    source: str = ""

    @staticmethod
    def parse(query: str, main_tag: str | None = None) -> QuerySpec | None:
        area = GEO.search(query)
        if not area or UNSUPPORTED.search(query):
            return None
        stmts = STMT.findall(query)
        if not stmts or len({s[1] for s in stmts}) != 1:
            return None
        filters = [Filter.parse(f) for f in re.findall(r'\[[^\]]*\]', stmts[0][1])]
        if any(f is None for f in filters):
            return None
        spec = QuerySpec(area.group(1).split(",")[0].strip(), tuple(sorted({s[0] for s in stmts})),
                         filters, source=query)
        if main_tag:
            mt = Filter.from_tag(main_tag)
            spec.main = next((i for i, f in enumerate(filters) if mt and f == mt), None)
        return spec

    @property
    def gold_type(self) -> str:
        return self.types[0] if len(self.types) == 1 and self.types[0] in ELEMENT_TYPES else "nwr"

    @property
    def secondary(self) -> list[Filter]:
        return [f for i, f in enumerate(self.filters) if i != self.main]

    def render(self, area_id: str, etype: str = "nwr", tag: Filter | None = None,
               drop: int | None = None, out: str = "out ids;") -> str:
        """Concrete query for one choice of readings. `drop` indexes into `secondary`."""
        main = tag if tag is not None else (self.filters[self.main] if self.main is not None else None)
        conds = ([main] if main else []) + [f for j, f in enumerate(self.secondary) if j != drop]
        body = "".join(f.raw for f in conds)
        return ('[out:json][timeout:90];area(id:%s)->.searchArea;(%s%s(area.searchArea););%s'
                % (area_id, etype, body, out))


AROUND = re.compile(r'\b(node|way|relation|nwr|nw|nr|wr)((?:\[[^\]]*\])*)'
                    r'\(around:(\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)')


@dataclass
class AroundSpec:
    """'TYPE[filters](around:R,lat,lon)' - distance ambiguity around a coordinate. The first filter
    is the object itself (fixed); the remaining ones are secondary conditions that may be dropped."""
    lat: str
    lon: str
    radius: str
    types: tuple[str, ...]
    filters: list[Filter]
    source: str = ""
    main: int | None = 0

    @staticmethod
    def parse(query: str) -> AroundSpec | None:
        st = AROUND.findall(query)
        if not st or len({s[1] for s in st}) != 1 or len({s[2:] for s in st}) != 1:
            return None
        filters = [Filter.parse(f) for f in re.findall(r'\[[^\]]*\]', st[0][1])]
        if not filters or any(f is None for f in filters):
            return None
        return AroundSpec(st[0][3], st[0][4], st[0][2], tuple(sorted({s[0] for s in st})), filters, query)

    gold_type = QuerySpec.gold_type
    secondary = QuerySpec.secondary

    def render(self, radius: str, etype: str = "nwr", drop: int | None = None, out: str = "out ids;") -> str:
        conds = [self.filters[0]] + [f for j, f in enumerate(self.secondary) if j != drop]
        return ('[out:json][timeout:90];(%s%s(around:%s,%s,%s););%s'
                % (etype, "".join(f.raw for f in conds), radius, self.lat, self.lon, out))
