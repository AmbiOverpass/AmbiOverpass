"""Building the object slot of a topology by searching over LLM rewrite operations (TopoMCTS-Build).

A hard request names the wanted objects indirectly: through properties a map does not record, vague
everyday words, or requirements a map cannot check. Readings are OSM tags. One LLM call = one expansion:
at a node (the request, or one reading of it) the LLM proposes K new readings that differ from the
node's existing children, each with up to 5 tags. Every method pays one unit per call and differs only
in WHERE it expands:

  resample     always expand the request, no memory (temperature 1.0)
  iterative    always expand the request, told what was already proposed (flat breadth)
  beam         fixed schedule: request, then the B_W most plausible readings of each level, depth <= D_MAX
  mcts         PUCT over "expand here" vs "descend into a reading", progressive widening
  dpw          MCTS with sampled widening: an expansion SAMPLES readings (no "give different ones" list);
               a sampled reading that repeats an existing one is merged into it and counts as evidence.
               Prior of a reading = its share of the samples at the node; prior of expanding again =
               ALPHA / (samples + ALPHA) (Chinese-restaurant style). Mass flows down the DAG by these shares.

What the topology adds to the search (mcts vs mcts_plain):
  grounding   a tag no OSM object carries (taginfo count 0) is void: no reading, no reward
  merging     a proposed reading whose valid tags repeat an existing node's is merged into it (DAG),
              so paraphrases neither create nodes nor earn reward
  reward      yield of an expansion = share of its rank-weighted VALID tag mass that is new
mcts_plain counts every new tag string as progress and merges only identical reading texts.

Prior: readings proposed at a node are ranked; reading r gets weight (1-RHO) RHO^r of its parent, so
"readings not yet proposed" keep the geometric tail RHO^n/(1-RHO) -- the prior of expanding again.
Final candidate ranking (all methods, so the comparison isolates the search): valid tags by mass,
mass(t) = sum over readings proposing t of weight(reading) * RHO_T^rank.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field

K = 3            # readings per expansion
D_MAX = 3        # readings of readings of readings
RHO = 0.6        # prior decay over reading rank
RHO_T = 0.8      # decay over tag rank within a reading
C_PUCT = 1.0
B_W = 2          # beam width
ALPHA = 3.0      # dpw: pseudo-count of "a reading not sampled yet" (one call samples K readings)

SYS = "You map natural-language map requests to OpenStreetMap. Answer with JSON only."
TAG_RE = re.compile(r'([a-z][a-z0-9_:.\-]*)\s*=\s*([^,"\]\n]+)')
# keys that describe an object's attributes, not what kind of object it is; placeholder values
NOT_KIND = ("name", "addr:", "contact:", "website", "phone", "email", "url", "opening_hours", "wikipedia",
            "wikidata", "description", "note", "population", "ele", "source", "fixme", "image", "operator",
            "brand", "check_date", "start_date")
PLACEHOLDER = re.compile(r"\.\.\.|[*<>|?]")


def kind_tag(t: str) -> bool:
    k, v = t.split("=", 1)
    return not k.startswith(NOT_KIND) and not PLACEHOLDER.search(v)


def norm_tag(t: str) -> str | None:
    m = TAG_RE.search(t.strip().strip('"[]').lower())
    if not m:
        return None
    return "%s=%s" % (m.group(1).strip(), m.group(2).strip().strip("'\"").replace(" ", "_"))


def expansion_prompt(request: str, node_text: str | None, avoid: list[str], k: int = K) -> str:
    head = ('Request: "%s"\nA non-expert wrote this request for map data. It may describe the wanted objects '
            'through properties a map does not record, in vague everyday words, or with requirements a map '
            'cannot check (ignore those).\n' % request)
    if node_text is None:
        task = "Give %d different readings of what kind of map object the user wants. Each reading names one concrete kind of object." % k
    else:
        task = ('One reading of it: "%s".\nGive %d more precise readings within this one: specific sub-kinds, or '
                'the exact kind of object a mapper would record for it.' % (node_text, k))
    av = ""
    if avoid:
        av = "\nAlready considered (give different ones):\n" + "\n".join("- " + a for a in avoid)
    return (head + task + av + "\nFor each reading list up to 5 OpenStreetMap tags (key=value) for that kind of "
            'object, most likely first.\nOutput JSON only: [{"reading": "...", "tags": ["key=value", ...]}, ...]')


def parse_readings(text: str) -> list[tuple[str, list[str]]]:
    out = []
    m = re.search(r"\[.*\]", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else []
    except Exception:
        data = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        tags = []
        for t in d.get("tags", []) if isinstance(d.get("tags"), list) else []:
            n = norm_tag(t) if isinstance(t, str) else None
            if n and n not in tags:
                tags.append(n)
        txt = str(d.get("reading", "")).strip()
        if txt and tags:
            out.append((txt, tags[:5]))
    return out


class CachedLLM:
    """Deterministic replays: response cached by (prompt, temperature, nonce); nonce = how many times this
    exact prompt was already sent within one method run on one item."""

    def __init__(self, llm, path: str = "out/build_llm_cache.jsonl"):
        self.llm, self.path = llm, path
        self.mem: dict[str, str] = {}
        self.lock = threading.Lock()
        if os.path.exists(path):
            for l in open(path):
                d = json.loads(l)
                self.mem[d["k"]] = d["v"]

    def __call__(self, prompt: str, temperature: float, nonce: int, max_tokens: int = 500) -> str:
        k = hashlib.sha1(json.dumps([prompt, temperature, nonce]).encode()).hexdigest()
        if k in self.mem:
            return self.mem[k]
        v = self.llm.chat([{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                          temperature=temperature, max_tokens=max_tokens)
        with self.lock:
            self.mem[k] = v
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
        return v


@dataclass(eq=False)
class Node:
    text: str | None                 # None = the request itself
    depth: int
    w: float                         # prior plausibility
    tags: list[str] = field(default_factory=list)
    children: list = field(default_factory=list)
    proposed: list = field(default_factory=list)   # texts of every reading proposed here (incl. merged)
    n: int = 0                       # expansions in the subtree
    v: float = 0.0                   # summed yield of those
    nw: int = 0                      # expansions at this node
    vw: float = 0.0
    own: float = 0.0                 # share of this reading's valid tag mass that was new when proposed
    edges: dict = field(default_factory=dict)      # dpw: child -> #samples at this node that landed on it
    props: int = 0                   # dpw: readings sampled at this node
    tagw: dict = field(default_factory=dict)       # dpw: tag -> summed normalised rank weight over samples


class Builder:
    def __init__(self, request: str, llm: CachedLLM, taginfo, method: str, grounded: bool, temperature: float = 0.7):
        self.x, self.llm, self.ti, self.method = request, llm, taginfo, method
        self.t = temperature
        self.grounded = grounded
        self.root = Node(None, 0, 1.0)
        self.nodes = [self.root]
        self.mass: dict[str, float] = {}
        self.seen_strings: set[str] = set()
        self.calls = 0
        self.nonces: dict[str, int] = {}
        self.proposed_tags = 0
        self.void_tags = 0

    # -- grounding -------------------------------------------------------------------------------
    def valid(self, t: str) -> bool:
        return kind_tag(t) and self.ti.count(t) > 0

    # -- one LLM call ------------------------------------------------------------------------------
    def expand(self, node: Node, temperature: float | None = None, memory: bool = True) -> float:
        temperature = self.t if temperature is None else temperature
        avoid = node.proposed if memory else []
        prompt = expansion_prompt(self.x, node.text, avoid)
        nonce = self.nonces.get(prompt, 0)
        self.nonces[prompt] = nonce + 1
        readings = parse_readings(self.llm(prompt, temperature, nonce))
        self.calls += 1
        new_mass = tot_mass = 0.0
        for txt, tags in readings:
            node.proposed.append(txt)
            rank = len(node.children)
            w = node.w * (1 - RHO) * RHO ** rank
            valid = [t for t in tags if self.valid(t)]
            self.proposed_tags += len(tags)
            self.void_tags += len(tags) - len(valid)
            # novelty as the search sees it
            new = tot = 0.0
            if self.grounded:
                for r, t in enumerate(valid):
                    tot += RHO_T ** r
                    new += RHO_T ** r if t not in self.mass else 0.0
            else:
                for r, t in enumerate(tags):
                    tot += RHO_T ** r
                    new += RHO_T ** r if t not in self.seen_strings else 0.0
            # evidence: every proposal adds mass to its valid tags (merged ones too)
            for r, t in enumerate(valid):
                self.mass[t] = self.mass.get(t, 0.0) + w * RHO_T ** r
            self.seen_strings.update(tags)
            new_mass += new
            tot_mass += tot
            if self.merged(node, txt, valid):
                continue
            child = Node(txt, node.depth + 1, w, tags=tags, own=new / tot if tot else 0.0)
            node.children.append(child)
            self.nodes.append(child)
        return new_mass / tot_mass if tot_mass else 0.0

    def merge_target(self, txt: str, valid: list[str]):
        """dpw: the existing node a sampled reading repeats, "void" if it has no valid tag, else None."""
        if not self.grounded:
            return next((c for c in self.nodes if c.text and c.text.lower() == txt.lower()), None)
        if not valid:
            return "void"
        for c in self.nodes:
            if c.text is None:
                continue
            cv = [t for t in c.tags if self.valid(t)]
            if cv and cv[0] == valid[0] and len(set(cv) & set(valid)) * 2 >= min(len(cv), len(valid)):
                return c
        return None

    def ancestors(self, node: Node) -> set:
        out, todo = set(), [node]
        while todo:
            n = todo.pop()
            for p in self.nodes:
                if n in p.edges and p not in out:
                    out.add(p)
                    todo.append(p)
        return out

    def expand_dpw(self, node: Node) -> float:
        prompt = expansion_prompt(self.x, node.text, [])
        nonce = self.nonces.get(prompt, 0)
        self.nonces[prompt] = nonce + 1
        readings = parse_readings(self.llm(prompt, self.t, nonce))
        self.calls += 1
        new_mass = tot_mass = 0.0
        for txt, tags in readings:
            valid = [t for t in tags if self.valid(t)]
            self.proposed_tags += len(tags)
            self.void_tags += len(tags) - len(valid)
            seen = self.mass if self.grounded else self.seen_strings
            counted = valid if self.grounded else tags
            new = sum(RHO_T ** r for r, t in enumerate(counted) if t not in seen)
            tot = sum(RHO_T ** r for r in range(len(counted)))
            for t in valid:
                self.mass.setdefault(t, 0.0)                # "seen" for novelty; ranking() recomputes mass
            self.seen_strings.update(tags)
            new_mass += new
            tot_mass += tot
            m = self.merge_target(txt, valid)
            if m == "void":
                continue
            node.props += 1
            if m is None:
                m = Node(txt, node.depth + 1, 0.0, tags=tags, own=new / tot if tot else 0.0)
                self.nodes.append(m)
            if m is node or m in self.ancestors(node):
                node.props -= 1                             # proposing an ancestor again: no edge, no cycle
                continue
            node.edges[m] = node.edges.get(m, 0) + 1
            if m not in node.children:
                node.children.append(m)
            z = sum(RHO_T ** r for r in range(len(valid))) or 1.0
            for r, t in enumerate(valid):
                m.tagw[t] = m.tagw.get(t, 0.0) + RHO_T ** r / z
        return new_mass / tot_mass if tot_mass else 0.0

    def dpw_step(self):
        n, path = self.root, [self.root]
        while n.children:
            sq = math.sqrt(n.n + 1)
            zz = n.props + ALPHA
            qw = n.vw / n.nw if n.nw else 1.0
            best, score = None, qw + C_PUCT * (ALPHA / zz) * sq / (1 + n.nw)
            for c in n.children:
                if c.depth >= D_MAX or c in path:
                    continue
                q = c.v / c.n if c.n else c.own
                s = q + C_PUCT * (n.edges[c] / zz) * sq / (1 + c.n)
                if s > score:
                    best, score = c, s
            if best is None:
                break
            n = best
            path.append(n)
        y = self.expand_dpw(n)
        n.nw += 1
        n.vw += y
        for p in path:
            p.n += 1
            p.v += y

    def dpw_mass(self) -> dict:
        w = {self.root: 1.0}

        def weight(c, stack=()):
            if c in w:
                return w[c]
            tot = 0.0
            for p in self.nodes:
                if c in p.edges and p not in stack and p.props:
                    tot += weight(p, stack + (c,)) * p.edges[c] / p.props
            w[c] = tot
            return tot

        mass = {}
        for c in self.nodes[1:]:
            k = sum(c.tagw.values())
            if not k:
                continue
            wc = weight(c)
            for t, x in c.tagw.items():
                mass[t] = mass.get(t, 0.0) + wc * x / k
        return mass

    def merged(self, parent: Node, txt: str, valid: list[str]) -> bool:
        if not self.grounded:
            return any(c.text and c.text.lower() == txt.lower() for c in self.nodes)
        if not valid:
            return True                                    # nothing in OSM: not a reading at all
        for c in self.nodes:
            if c.text is None:
                continue
            cv = [t for t in c.tags if self.valid(t)]
            if cv and cv[0] == valid[0] and len(set(cv) & set(valid)) * 2 >= min(len(cv), len(valid)):
                return True
        return False

    # -- search policies ---------------------------------------------------------------------------
    def step(self, state: dict):
        m = self.method
        if m == "resample":
            self.expand(self.root, temperature=1.0, memory=False)
        elif m == "iterative":
            self.expand(self.root)
        elif m == "beam":
            self.beam_step(state)
        elif m in ("dpw", "dpw_plain"):
            self.dpw_step()
        else:
            self.mcts_step()

    def beam_step(self, state: dict):
        # expand the request, then the B_W most plausible unexpanded readings of each level down to
        # depth D_MAX - 1, then the request again
        q = state.setdefault("queue", [])
        if not q:
            lvl = state.get("level", 0)
            cands = [n for n in self.nodes if n.depth == lvl and n.nw == 0] if 0 < lvl < D_MAX else []
            if cands:
                q.extend(sorted(cands, key=lambda n: -n.w)[:B_W])
                state["level"] = lvl + 1
            else:
                q.append(self.root)
                state["level"] = 1
        node = q.pop(0)
        self.expand(node)
        node.nw += 1

    def mcts_step(self):
        n, path = self.root, [self.root]
        while True:
            kids = [c for c in n.children if c.depth < D_MAX]
            if not n.children:
                break
            nb = len(n.children)
            pri = [(1 - RHO) * RHO ** i for i in range(nb)]
            pw = RHO ** nb                                   # tail: readings not yet proposed here
            z = sum(pri) + pw
            sq = math.sqrt(n.n + 1)
            qw = n.vw / n.nw if n.nw else 1.0
            best, score = None, qw + C_PUCT * (pw / z) * sq / (1 + n.nw)
            for i, c in enumerate(n.children):
                if c not in kids:
                    continue
                q = c.v / c.n if c.n else c.own
                s = q + C_PUCT * (pri[i] / z) * sq / (1 + c.n)
                if s > score:
                    best, score = c, s
            if best is None:
                break
            n = best
            path.append(n)
        y = self.expand(n)
        n.nw += 1
        n.vw += y
        for p in path:
            p.n += 1
            p.v += y

    # -- output ------------------------------------------------------------------------------------
    def ranking(self) -> list[str]:
        mass = self.dpw_mass() if self.method in ("dpw", "dpw_plain") else self.mass
        return sorted(mass, key=lambda t: -mass[t])


def oneshot(request: str, llm: CachedLLM, n: int = 50) -> list[str]:
    prompt = ("Request: %s\nList %d different OpenStreetMap tags (key=value) that could describe the MAIN kind of "
              "object the user wants, most likely first. Output a JSON list of strings like [\"amenity=cafe\", ...]."
              % (request, n))
    out = llm(prompt, 0.3, 0, max_tokens=60 + 14 * n)
    tags = []
    m = re.search(r"\[.*\]", out, re.S)
    try:
        raw = json.loads(m.group(0)) if m else []
    except Exception:
        raw = TAG_RE.findall(out)
        raw = ["%s=%s" % kv for kv in raw]
    for t in raw:
        t = norm_tag(t) if isinstance(t, str) else None
        if t and t not in tags:
            tags.append(t)
    return tags[:n]
