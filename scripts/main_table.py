"""Exp.2 main table (and Exp.3 / Exp.4 from the same runs) on AmbiOverpass-M, following
~/Desktop/Ambiguous_Text_to_OverpassQL_Experiment_Plan.md.

All planners share the candidates, prior (uniform), execution results, budget and the final output rule
(submit the most probable result class; its representative reading is rendered to OverpassQL). Only the
next-step decision differs. User = gold reading, answers "no preference" with prob NOISE.

Metrics (per item, then averaged): EM = rendered prediction equals the rendered gold reading (same
renderer, so formatting cannot differ); OQS = mean of key/value F1, structure F1 (element type, area,
radius, number of filters; a stand-in for OverpassNL's XML overlap) and chrF; EX = predicted result set
equals the gold result set; EXSOFT = Jaccard of the two result sets; EXERROR = prediction not executable
(empty or failed); #Questions.

  python scripts/main_table.py [K=2] [seeds=5]          # needs the local LLM for the LLM planner
"""
import collections
import hashlib
import json
import os
import random
import re
import statistics
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo import (AskAll, EIGReadings, EIGClasses, Greedy, Natural, NoAsk, Optimal, PlainMCTS, Setting, ShapleyGreedy, ShapleyRollout,
                      SimulatedUser, TopoMCTS, run_episode)
from ambitopo.belief import jaccard
from ambitopo.data import build, load_rows, to_sample
from ambitopo.overpass import Executor
from ambitopo.planners import SUBMIT, Action, Policy
from ambitopo.query import AroundSpec

K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
LAM, NOISE = 0.05, 0.2
BUDGETS = (0, 1, 2, 3)
PRIOR = json.load(open(os.environ.get("MT_PRIOR_FILE", "out/llm_prior_m2.json"))) if os.environ.get("MT_PRIOR", "").startswith("llm") else {}
CALIB = json.load(open(os.environ.get("MT_CALIB_FILE", "out/prior_calib.json"))) if os.environ.get("MT_PRIOR") == "llmcal" else {}
TRUTH = os.environ.get("MT_TRUTH", "gold")
POLS = [x for x in os.environ.get("POLS", "").split(";") if x]     # run only these policies (names separated by ";")          # gold | prior (truth drawn from the shared prior)
BENCH = {}
ITEMS = {}
VARIANT = os.environ.get("MT_VARIANT", "m2b")      # "full": every slot of every collected sample varies
for l in open("out/ambioverpass_m.jsonl") if VARIANT == "bench" else []:
    it = json.loads(l)
    if not it["flags"]:
        BENCH[it["sample_id"]] = tuple(v if v != "dist" else "area" for v in it["weakened"])
        ITEMS[it["sample_id"]] = it


# ---------------------------------------------------------------------------------------------- policies
class RandomAsk(Policy):
    name = "random"

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def act(self, st, alive, used):
        if st.belief.certain(alive) or st.left(used) == 0:
            return SUBMIT
        c = st.candidates(alive)
        return Action("ask", self.rng.choice(c)) if c else SUBMIT


SLOT_WORDS = {"area": "which place (area)", "tag": "what kind of object (OSM tag)", "type": "point, line or area (element type)",
              "filter": "whether an extra condition must hold", "dist": "how far (search radius)"}


class LLMPlanner(Policy):
    """Qwen3-14B decides which ambiguity to ask about next, or to stop (ReAct-style clarification agent)."""
    name = "llm_planner"
    cache = {}

    def __init__(self, request, topo):
        from ambitopo.llm import OpenAICompatible
        self.llm = OpenAICompatible()
        self.request, self.topo = request, topo

    def act(self, st, alive, used):
        b = st.belief
        k = st.left(used)
        if b.certain(alive) or k == 0:
            return SUBMIT
        cands = st.candidates(alive)
        if not cands:
            return SUBMIT
        lines = []
        for v in cands:
            opts = [self.topo.slots[v].readings[r].label for r in sorted(b.split(alive, v))]
            lines.append("- %s: %s. Options still possible: %s" % (v, SLOT_WORDS.get(v, self.topo.slots[v].kind), "; ".join(opts[:6])))
        prompt = ("A user asked a map database: \"%s\"\nSome parts of the request are unclear. You may ask the user at most %d "
                  "more question(s), each about ONE of these parts, and every question costs the user effort:\n%s\n"
                  "Which part should be asked about next? If asking is not worth it, answer STOP. "
                  "Answer with exactly one word: one of [%s] or STOP." % (self.request, k, "\n".join(lines), ", ".join(cands)))
        if prompt not in self.cache:
            self.cache[prompt] = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=8)
        ans = self.cache[prompt].strip().lower()
        for v in cands:
            if re.search(r"\b%s\b" % v, ans):
                return Action("ask", v)
        return SUBMIT


REACT_WORDS = dict(SLOT_WORDS, anchor="which kind of landmark is meant", anchor1="which kind of landmark the first landmark word means",
                   anchor2="which kind of landmark the second landmark word means", dist1="how close to the first landmark (search radius)",
                   dist2="how close to the second landmark (search radius)")


class ReActPlanner(Policy):
    """ReAct-style clarification agent (Yao et al., 2023) with the same LLM: at every step it sees the request, the
    dialogue so far (question -> answer, or "no preference"), an observation of the current state (how many
    interpretations remain and how many distinct query results they give), and the open parts with their options;
    it writes a short thought and then an action, ASK <part> or STOP. Same information as the other planners; no gold."""
    name = "react"
    cache = {}

    def __init__(self, request, topo):
        from ambitopo.llm import OpenAICompatible
        self.llm = OpenAICompatible()
        self.request, self.topo = request, topo
        self.hist, self.last, self.prev = [], None, None

    def _record(self, b, alive):
        if self.last is None:
            return
        v = self.last
        if alive == self.prev:
            ans = "no preference"
        else:
            vals = {b.value(i, v) for i in alive}
            ans = self.topo.slots[v].readings[vals.pop()].label if len(vals) == 1 else "partly answered"
        self.hist.append("Q: %s? -> A: %s" % (REACT_WORDS.get(v, v), ans))
        self.last = None

    def act(self, st, alive, used):
        b = st.belief
        self._record(b, alive)
        k = st.left(used)
        if b.certain(alive) or k == 0:
            return SUBMIT
        cands = st.candidates(alive)
        if not cands:
            return SUBMIT
        n_cls = len({b.cls[i] for i in alive})
        lines = []
        for v in cands:
            opts = [self.topo.slots[v].readings[r].label for r in sorted(b.split(alive, v))]
            lines.append("- %s: %s. Options still possible: %s" % (v, REACT_WORDS.get(v, self.topo.slots[v].kind), "; ".join(opts[:6])))
        prompt = ("You help a user query a map database. The user asked: \"%s\"\n"
                  "Dialogue so far:\n%s\n"
                  "Observation: %d interpretations of the request are still possible; executed on the database they give %d "
                  "different results. If you stop now, the system returns the most likely result.\n"
                  "Open parts you may ask about (each question costs the user effort; at most %d more question(s)):\n%s\n"
                  "Think step by step in at most three sentences about which part, if any, is worth asking, then give your action.\n"
                  "Format:\nThought: ...\nAction: ASK <part> or Action: STOP, where <part> is one of [%s]."
                  % (self.request, "\n".join(self.hist) or "(none)", len(alive), n_cls, k, "\n".join(lines), ", ".join(cands)))
        if prompt not in self.cache:
            self.cache[prompt] = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=220)
        txt = self.cache[prompt]
        m = re.search(r"Action:\s*(ASK\s+(\w+)|STOP)", txt, re.I)
        if m and m.group(2) and m.group(2).lower() in cands:
            self.last, self.prev = m.group(2).lower(), alive
            return Action("ask", self.last)
        if m is None:                                     # no parsable action: take the first part named after "Action"
            for v in cands:
                if re.search(r"\b%s\b" % v, txt.split("Action")[-1], re.I):
                    self.last, self.prev = v, alive
                    return Action("ask", v)
        return SUBMIT


# ---------------------------------------------------------------------------------------------- metrics
KV = re.compile(r'\["([^"]+)"(?:(=|!=|~)"([^"]*)")?\]')



_ANSWERS = None


def make_user(b, g, item_id, noise, seed):
    """The simulated user; with LLM_USER set (V4 robustness check) answers come from that table of LLM answers."""
    global _ANSWERS
    if os.environ.get("LLM_USER"):
        if _ANSWERS is None:
            _ANSWERS = json.load(open(os.environ["LLM_USER"]))
        from ambitopo import AnswerTableUser
        return AnswerTableUser(b, g, _ANSWERS.get(item_id, {}), noise=noise, seed=seed)
    return SimulatedUser(b, g, noise=noise, seed=seed)

def kv(q):
    return {m.group(0) for m in KV.finditer(q)}


def structure(q):
    s = set()
    m = re.search(r"\((node|way|relation|nwr)\[", q) or re.search(r"\b(node|way|relation|nwr)\[", q)
    s.add("type:" + (m.group(1) if m else "?"))
    a = re.search(r"area\(id:(\d+)\)", q)
    s.add("area:" + (a.group(1) if a else "-"))
    r = re.search(r"around:([\d.]+)", q)
    s.add("around:" + (r.group(1) if r else "-"))
    s.add("nfilters:%d" % len(kv(q)))
    return s


def f1(a, b):
    if not a and not b:
        return 1.0
    i = len(a & b)
    return 2 * i / (len(a) + len(b)) if (a or b) else 0.0


def chrf(hyp, ref, n=6, beta=2.0):
    ps, rs = [], []
    for k in range(1, n + 1):
        h = collections.Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = collections.Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        m = sum((h & r).values())
        if sum(h.values()):
            ps.append(m / sum(h.values()))
        if sum(r.values()):
            rs.append(m / sum(r.values()))
    p, r = (statistics.mean(ps) if ps else 0), (statistics.mean(rs) if rs else 0)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if p + r else 0.0


def render(s, b, i):
    a = dict(zip(b.slots, b.hyps[i].assign))
    sl = s.topo.slots
    if isinstance(s.spec, AroundSpec):
        return s.spec.render(sl["dist"].readings[a["dist"]].value, sl["type"].readings[a["type"]].value,
                             sl["filter"].readings[a["filter"]].value)
    return s.spec.render(sl["area"].readings[a["area"]].value, sl["type"].readings[a["type"]].value,
                         sl["tag"].readings[a["tag"]].value, sl["filter"].readings[a["filter"]].value)


# ---------------------------------------------------------------------------------------------- runs
def policies(item, s, seed):
    req = item.get("x_vague") or item["x_clear"]
    return [("Direct", lambda: NoAsk()), ("Random", lambda: RandomAsk(seed * 7 + len(item["x_clear"]))),
            ("Ask every ambiguity (Sphinteract/AmbiSQL)", lambda: AskAll()),
            ("Ask until results agree (ClarifyGPT)", lambda: Natural()),
            ("Info gain over readings (Qiu et al.)", lambda: EIGReadings()),
            ("Greedy (topology importance)", lambda: Greedy()),
            ("LLM planner (Qwen3-14B)", lambda: LLMPlanner(req, s.topo)),
            ("ReAct (Qwen3-14B)", lambda: ReActPlanner(req, s.topo)),
            ("Info gain over result classes", lambda: EIGClasses()),
            ("Generic MCTS (VisInteract)", lambda: PlainMCTS(n_sim=30, seed=seed)),
            ("TopoMCTS", lambda: TopoMCTS(n_sim=30, seed=seed)),
            ("TopoMCTS (100 sims)", lambda: TopoMCTS(n_sim=100, seed=seed)),
            ("TopoMCTS-LA", lambda: TopoMCTS(n_sim=300, seed=seed, prior="lookahead")),
            ("ShapleyGreedy", lambda: ShapleyGreedy()),
            ("ShapleyRollout", lambda: ShapleyRollout()),
            ("TopoMCTS-SV", lambda: TopoMCTS(n_sim=300, seed=seed, prior="lookahead_shapley", rollout="shapley")),
            ("Exact planner", lambda: Optimal(canonical=True))]


def jobs():
    rows = {r["sample_id"]: r for r in load_rows()}
    if VARIANT in ("m2", "m2b"):
        its = [json.loads(l) for l in open("out/ambioverpass_%s.jsonl" % VARIANT)]
        split = os.environ.get("MT_SPLIT", "all")
        return [(rows[it["sample_id"]], it) for it in its if not it["flags"] and split in ("all", it.get("split"))]
    return [(r, None) for r in rows.values()]


def one(job):
    row, m2 = job
    sid = row["sample_id"]
    if m2 is not None:
        BENCH[m2["item_id"]] = tuple(m2["vary"])
        ITEMS[m2["item_id"]] = m2
        sid = m2["item_id"]
    if VARIANT == "full":
        from ambitopo.readings import SLOT_NAMES
        if to_sample(row) is None:
            return None
        BENCH[sid] = SLOT_NAMES
        ITEMS[sid] = {"x_clear": row.get("x_amb") or row["x_clear"]}
    if sid not in BENCH:
        return None
    s = build(to_sample(row), Executor("out/cache"), vary=BENCH[sid], offline=True)
    if s is None or s.belief is None:
        return None
    b = s.belief
    if m2 is not None and os.environ.get("MT_PRIOR", "").startswith("llm"):
        from ambitopo import Belief
        P = {}
        for v, p in PRIOR.get(m2.get("prior_from", sid), {}).items():
            a = CALIB.get(v, 1.0)
            q = [max(x, 1e-9) ** a for x in p]
            P[v] = [x / sum(q) for x in q]
        w = []
        for h in b.hyps:
            x = 1.0
            for v, r in zip(b.slots, h.assign):
                if v in P:
                    x *= P[v][r]
            w.append(x)
        b = s.belief = Belief(b.topo, b.hyps, prior=w)
    amb = [v for v in b.slots if b.splits(b.root, v)]
    ov = (m2 or {}).get("gold_override", {})
    gold = tuple(ov.get(n, s.topo.slots[n].gold) for n in s.topo.order)
    gi = [i for i, h in enumerate(b.hyps) if h.assign == gold]
    if not gi or not amb:
        return None
    g0 = gi[0]
    rec = {"id": sid, "n_amb": len(BENCH[sid]) if VARIANT == "bench" else (m2["n_matter"] if m2 else len(amb)),
           "B": m2["B"] if m2 else None, "res": {}}
    for budget in BUDGETS:
        st = Setting(b, askable=amb, budget=budget, lam=LAM, noise=NOISE, show_cost=None)
        for seed in range(int(os.environ.get("SEED_OFFSET", "0")), int(os.environ.get("SEED_OFFSET", "0")) + SEEDS):   # SEED_OFFSET: robustness check
            if TRUTH == "prior":
                rt = random.Random(seed * 7919 + int(hashlib.md5(sid.encode()).hexdigest()[:8], 16))
                g = rt.choices(b.root, weights=[b.w[i] for i in b.root])[0]
            else:
                g = g0
            gq, gres = render(s, b, g), b.hyps[g].result
            for name, fn in policies(ITEMS[sid], s, seed):
                if POLS and name not in POLS:
                    continue
                if budget != K and name.startswith(("Random", "Ask every", "Ask until", "Info gain", "LLM", "ReAct")):
                    continue
                user_seed = seed * 1000003 + int(hashlib.md5(sid.encode()).hexdigest()[:8], 16)
                e = run_episode(st, fn(), make_user(b, g, sid, noise=NOISE, seed=user_seed))
                pq, pres = render(s, b, e.submitted), b.hyps[e.submitted].result
                m = {"EM": float(pq == gq), "OQS": (f1(kv(pq), kv(gq)) + f1(structure(pq), structure(gq)) + chrf(pq, gq)) / 3,
                     "EX": float(pres == gres), "EXSOFT": jaccard(pres, gres), "EXERROR": float(not pres),
                     "Q": e.asks, "succ": float(e.success)}
                rec["res"].setdefault("%s|K=%d" % (name, budget), []).append(m)
    return rec


def agg(R, key, metric):
    """mean over items per seed, then mean and std over seeds"""
    per_seed = collections.defaultdict(list)
    for r in R:
        for j, m in enumerate(r["res"].get(key, [])):
            per_seed[j].append(m[metric])
    vals = [statistics.mean(v) for v in per_seed.values()]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    with Pool(32) as pool:
        R = [r for r in pool.imap_unordered(one, jobs(), chunksize=1) if r]
    json.dump(R, open("out/main_table_%s_%s_%s_%s_K%d%s.json" % (VARIANT, os.environ.get("MT_SPLIT", "all"),
                                                             os.environ.get("MT_PRIOR", "uniform"), TRUTH, K, os.environ.get("OUT_TAG", "")), "w"))
    names = [n for n, _ in policies({"x_clear": ""}, None, 0)]
    print("[%s split=%s prior=%s truth=%s]" % (VARIANT, os.environ.get("MT_SPLIT", "all"), os.environ.get("MT_PRIOR", "uniform"), TRUTH))
    print("Exp.2 main table: AmbiOverpass-M, %d items (gold reading non-empty), K=%d, cost %.2f, %d%% 'no preference', %d seeds"
          % (len(R), K, LAM, 100 * NOISE, SEEDS))
    print("| Method | EM | OQS | EX | EXSOFT | EXERROR | #Questions |\n|---|---|---|---|---|---|---|")
    for n in names:
        key = "%s|K=%d" % (n, K)
        cells = []
        for met in ("EM", "OQS", "EX", "EXSOFT", "EXERROR", "Q"):
            mu, sd = agg(R, key, met)
            cells.append("%.3f" % mu + (" ± %.3f" % sd if sd > 0.0005 else ""))
        print("| %s | %s |" % (n, " | ".join(cells)))
    print("\nExp.3 by number of ambiguous slots (K=%d): OQS / EX / EXSOFT" % K)
    for n in ("Direct", "Greedy (topology importance)", "Generic MCTS (VisInteract)", "TopoMCTS", "Exact planner"):
        out = []
        for na in sorted({r["n_amb"] for r in R}):
            rs = [r for r in R if r["n_amb"] == na]
            out.append("%d-amb (n=%d) %.3f / %.3f / %.3f" % (na, len(rs), *(agg(rs, "%s|K=%d" % (n, K), m)[0] for m in ("OQS", "EX", "EXSOFT"))))
        print("  %-32s %s" % (n, " | ".join(out)))
    if any(r.get("B") is not None for r in R):
        print("\nBy closing blind spot B: EX at K=1 / K=2")
        for n in ("Greedy (topology importance)", "Generic MCTS (VisInteract)", "TopoMCTS", "Exact planner"):
            out = []
            for lab, f in (("B>0", lambda r: r["B"] > 1e-9), ("B=0", lambda r: r["B"] <= 1e-9)):
                rs = [r for r in R if f(r)]
                out.append("%s (n=%d) %.3f / %.3f" % (lab, len(rs), agg(rs, "%s|K=1" % n, "EX")[0], agg(rs, "%s|K=2" % n, "EX")[0]))
            print("  %-32s %s" % (n, " | ".join(out)))
    print("\nExp.4 interaction budget: EX (EXSOFT) [#Q] for K = 0,1,2,3")
    for n in ("Greedy (topology importance)", "Generic MCTS (VisInteract)", "TopoMCTS", "Exact planner"):
        print("  %-32s %s" % (n, " | ".join("%.3f (%.3f) [%.2f]" % tuple(agg(R, "%s|K=%d" % (n, k), m)[0] for m in ("EX", "EXSOFT", "Q"))
                                             for k in BUDGETS)))


if __name__ == "__main__":
    main()
