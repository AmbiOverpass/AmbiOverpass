"""Evaluate planners on AmbiOverpass-M3 ("target near anchor"), offline from the Overpass cache.
Same protocol as main_table.py: gold truth, 20% "no preference", cost 0.05, K in {1, 2, 3}, 5 seeds, LLM prior
with the dev-split calibration (slots without a fitted alpha, e.g. anchor, use alpha = 0, i.e. uniform).
Metrics: EX (identical result set), EXSOFT (Jaccard), #Questions; plus exact − greedy gap (pilot go/no-go).

  python scripts/eval_m3.py [prior=llmcal|uniform]
  M_BENCH=m4 python scripts/eval_m3.py ...      # AmbiOverpass-M4 (two anchors); adds K = none (no budget, cost only)
  test2 (frozen configuration, run once): scripts/run_test2_eval.sh
"""
import hashlib
import json
import os
import random
import statistics
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ambitopo import AskAll, Belief, EIGReadings, EIGClasses, Greedy, Natural, NoAsk, Optimal, PlainMCTS, Setting, ShapleyGreedy, ShapleyRollout, SimulatedUser, TopoMCTS, run_episode
from ambitopo.belief import jaccard
from ambitopo.data import load_rows, to_sample
from ambitopo.near import build_near, build_near2
from ambitopo.overpass import Executor
from ambitopo.readings import lookup_areas
_argv, sys.argv = sys.argv, sys.argv[:1]          # main_table parses argv at import
from main_table import LLMPlanner, ReActPlanner
sys.argv = _argv

PRIOR_MODE = sys.argv[1] if len(sys.argv) > 1 else "llmcal"
BENCH = os.environ.get("M_BENCH", "m3")
POLS = [x for x in os.environ.get("POLS", "").split(";") if x]     # run only these policies
BUDGETS = (1, 2, 3, None)
LAM, NOISE, SEEDS = 0.05, 0.2, 5
# robustness checks : other cost, noise and user streams; defaults unchanged
LAM = float(os.environ.get("AMBI_LAM", LAM))
NOISE = float(os.environ.get("AMBI_NOISE", NOISE))
SEED0 = int(os.environ.get("SEED_OFFSET", "0"))
CALIB = json.load(open("out/prior_calib.json"))
WHAT = {"area": "which place is meant", "tag": "which kind of map object is meant",
        "anchor": "which kind of nearby landmark is meant", "anchor1": "which kind of the first nearby landmark is meant",
        "anchor2": "which kind of the second nearby landmark is meant", "dist1": "which radius around the first landmark is meant",
        "dist2": "which radius around the second landmark is meant", "dist": "which search radius in metres is meant",
        "type": "which geometry is meant (node = point, way = line or outline, relation = grouped object, nwr = any)"}


RAW = None
KIND = {"anchor1": "anchor", "anchor2": "anchor", "dist1": "dist", "dist2": "dist"}



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

def toplabel(p, a, platt):
    """p^a with its top reading's probability replaced by Platt scaling sigmoid(c0 + c1 * logit p^a(top)); the other
    readings keep their p^a proportions. platt None -> plain p^a."""
    import math
    q = [x ** a for x in p]
    q = [x / sum(q) for x in q]
    if platt is None:
        return q
    t = max(range(len(q)), key=lambda i: q[i])
    z = min(max(q[t], 1e-6), 1 - 1e-6)
    top = 1 / (1 + math.exp(-(platt[0] + platt[1] * math.log(z / (1 - z)))))
    rest = 1 - q[t]
    return [top if i == t else (x / rest * (1 - top) if rest > 0 else (1 - top) / (len(q) - 1)) for i, x in enumerate(q)]


def llm_prior(llm, request, topo, item_id=None):
    """Per-slot prior p^alpha. With CALIB_FILE set (landmark calibration), raw LLM probabilities come from
    out/llm_prior_raw_landmark.json and alpha is looked up by slot kind."""
    import re
    global RAW
    out = {}
    if os.environ.get("CALIB_FILE") and item_id is not None:
        if RAW is None:
            RAW = json.load(open(os.environ.get("RAW_FILE", "out/llm_prior_raw_landmark.json")))
        cal = json.load(open(os.environ["CALIB_FILE"]))
        for v in topo.order:
            n = len(topo.slots[v].readings)
            if n < 2:
                continue
            a = cal.get(KIND.get(v, v), 0.0)
            p = RAW.get(item_id, {}).get(v, [1.0 / n] * n)
            if isinstance(a, dict):                # top-label calibration
                out[v] = toplabel(p, a["alpha"], a["platt"])
                continue
            if isinstance(a, list):                # bootstrap predictive calibration: mean of p^alpha over DEV bootstrap alphas
                acc = [0.0] * n
                for b in a:
                    q = [x ** b for x in p]
                    acc = [t + x / sum(q) for t, x in zip(acc, q)]
                out[v] = [t / len(a) for t in acc]
                continue
            q = [x ** a for x in p]
            out[v] = [x / sum(q) for x in q]
        return out
    for v in topo.order:
        labels = [r.label for r in topo.slots[v].readings]
        if len(labels) < 2:
            continue
        a = CALIB.get(v, 0.0)
        if a == 0.0:
            out[v] = [1.0 / len(labels)] * len(labels)
            continue
        opts = "\n".join("%d. %s" % (i + 1, l) for i, l in enumerate(labels))
        txt = llm.chat([{"role": "user", "content": "A user typed this map search: \"%s\"\nFor the question of %s, how likely is "
                         "each option to be what the user means? Options:\n%s\nGive a probability for every option (they should sum "
                         "to 1). Output JSON only: a list of numbers in option order." % (request, WHAT.get(v, v), opts)}],
                       temperature=0.0, max_tokens=120)
        m = re.search(r"\[[^\]]*\]", txt)
        try:
            p = [max(0.0, float(x)) for x in json.loads(m.group(0))][:len(labels)]
            assert len(p) == len(labels) and sum(p) > 0
            p = [0.05 / len(p) + 0.95 * x / sum(p) for x in p]
        except Exception:
            p = [1.0 / len(labels)] * len(labels)
        q = [x ** a for x in p]
        out[v] = [x / sum(q) for x in q]
    return out


def one(job):
    it, row = job
    s = to_sample(row)
    ex = Executor("out/cache")
    areas = lookup_areas(s.spec.area_name, ex, offline=True)
    seed = int(hashlib.md5(it["sample_id"].encode()).hexdigest()[:8], 16)
    if "anchor_kinds" not in it:                 # single landmark (m3, m3x)
        built = build_near(s.spec, s.tag_candidates, areas, it["anchor_kind"], ex, offline=True, seed=seed, gold=it["gold"])
    else:
        seed = int(hashlib.md5(it["item_id"].encode()).hexdigest()[:8], 16)
        built = build_near2(s.spec, s.tag_candidates, areas, tuple(it["anchor_kinds"]), ex, offline=True, seed=seed, gold=it["gold"])
    if built is None:            # gold reading empty under Overpass's exact `around`, or the area request failed
        return None
    topo, b, gold = built
    if PRIOR_MODE == "llmcal":
        from ambitopo.llm import OpenAICompatible
        P = llm_prior(None if os.environ.get("CALIB_FILE") else OpenAICompatible(), it["x_vague"], topo, it["item_id"])   # cached priors need no LLM
        w = []
        for h in b.hyps:
            x = 1.0
            for v, r in zip(b.slots, h.assign):
                if v in P:
                    x *= P[v][r]
            w.append(x)
        b = Belief(topo, b.hyps, prior=w)
    g = next(i for i, h in enumerate(b.hyps) if list(h.assign) == list(gold))
    amb = [v for v in b.slots if b.splits(b.root, v)]
    rec = {"id": it["item_id"], "split": it.get("split"), "hyps": len(b.hyps), "classes": b.n_classes, "slots": len(amb), "res": {}}
    for K in BUDGETS:
        st = Setting(b, askable=amb, budget=K, lam=LAM, noise=NOISE, show_cost=None)
        for seed_i in range(SEED0, SEED0 + SEEDS):
            pols = [("Direct", lambda: NoAsk()), ("Ask-all", lambda: AskAll()), ("ClarifyGPT-style", lambda: Natural()), ("Info gain (Qiu)", lambda: EIGReadings()),
                    ("LLM planner", lambda: LLMPlanner(it["x_vague"], topo)), ("Generic MCTS", lambda: PlainMCTS(n_sim=30, seed=seed_i)),
                    ("Greedy", lambda: Greedy()), ("TopoMCTS", lambda: TopoMCTS(n_sim=100, seed=seed_i)),
                    ("TopoMCTS-LA", lambda: TopoMCTS(n_sim=300, seed=seed_i, prior="lookahead")),
                    ("ShapleyGreedy", lambda: ShapleyGreedy()),
                    ("ShapleyRollout", lambda: ShapleyRollout()),
                    ("TopoMCTS-SV", lambda: TopoMCTS(n_sim=300, seed=seed_i, prior="lookahead_shapley", rollout="shapley")),
                    ("Exact", lambda: Optimal(canonical=True)),
                    ("ReAct", lambda: ReActPlanner(it["x_vague"], topo)), ("Info gain (classes)", lambda: EIGClasses()),
                    ("TopoMCTS-SV-t0.5", lambda: TopoMCTS(n_sim=300, seed=seed_i, prior="lookahead_shapley", rollout="shapley", plan_temp=0.5)),
                    ("TopoMCTS-SV-t0.25", lambda: TopoMCTS(n_sim=300, seed=seed_i, prior="lookahead_shapley", rollout="shapley", plan_temp=0.25))]     # built lazily: the LLM baselines need an endpoint only when they run
            if POLS:
                pols = [x for x in pols if x[0] in POLS]
            pols = [(n, f()) for n, f in pols]
            for name, pol in pols:
                e = run_episode(st, pol, make_user(b, g, it["item_id"], noise=NOISE, seed=seed_i * 1000003 + seed))
                rec["res"].setdefault("%s|K=%s" % (name, K), []).append(
                    {"EX": float(b.hyps[e.submitted].result == b.hyps[g].result),
                     "EXSOFT": jaccard(b.hyps[e.submitted].result, b.hyps[g].result), "Q": e.asks})
    return rec


def main():
    rows = {r["sample_id"]: r for r in load_rows()}
    items = [json.loads(l) for l in open("out/ambioverpass_%s.jsonl" % BENCH)]
    if os.environ.get("SPLIT") in ("dev", "test"):
        items = [it for it in items if it.get("split") == os.environ["SPLIT"]]
    with Pool(16) as pool:
        R = [r for r in pool.imap_unordered(one, [(it, rows[it["sample_id"]]) for it in items]) if r]
    json.dump(R, open("out/eval_%s_%s%s%s.json" % (BENCH, PRIOR_MODE, ("_" + os.environ["SPLIT"]) if os.environ.get("SPLIT") else "", os.environ.get("OUT_TAG", "")), "w"))
    m = lambda k: statistics.mean(r[k] for r in R)
    print("M3 items %d | joint readings %.0f | result classes %.1f | slots that split %.2f | prior %s"
          % (len(R), m("hyps"), m("classes"), m("slots"), PRIOR_MODE))
    names = [k.split("|")[0] for k in R[0]["res"] if k.endswith("K=1")]
    full = R
    for part in ("dev", "test", "all"):
        R = [r for r in full if part == "all" or r["split"] == part]
        if not R:
            continue
        print("== split %s (%d items)" % (part, len(R)))
        report(R, names)


def report(R, names):
    mean = lambda K, n, f: statistics.mean(statistics.mean(x[f] for x in r["res"]["%s|K=%s" % (n, K)]) for r in R)
    for K in BUDGETS:
        print("K=%s  " % K + " | ".join("%s %.3f [%.2f]" % (n, mean(K, n, "EX"), mean(K, n, "Q")) for n in names))
    if "Exact" not in names or "Greedy" not in names:
        return
    for K in BUDGETS:
        d = [statistics.mean(x["EX"] for x in r["res"]["Exact|K=%s" % K]) - statistics.mean(x["EX"] for x in r["res"]["Greedy|K=%s" % K]) for r in R]
        print("exact − greedy EX at K=%s: %+.3f" % (K, statistics.mean(d)))


if __name__ == "__main__":
    main()
