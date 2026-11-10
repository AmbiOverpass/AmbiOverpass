"""Calibration uncertainty in the planning prior.

  CALIB_CV=dev  python scripts/calib_cv.py   # 4-fold source CV on dev, point alpha vs bootstrap predictive -> out/calib_cv_dev.jsonl
  CALIB_CV=fit  python scripts/calib_cv.py   # fit both on all dev items -> out/prior_calib_boot.json (alpha samples per kind)
"""
import hashlib
import json
import math
import os
import random
import statistics as S
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import eval_m3 as E
sys.argv = _argv
from landmark_prior import build
from ambitopo import Belief, EIGClasses, Greedy, Setting, ShapleyRollout, SimulatedUser, run_episode
from ambitopo.data import load_rows
from ambitopo.overpass import Executor

KIND = {"anchor1": "anchor", "anchor2": "anchor", "dist1": "dist", "dist2": "dist"}
KINDS = ("area", "tag", "anchor", "dist", "type")
GRID = [i / 20 for i in range(61)]
B, FOLDS, SEEDS = 50, 4, 5
RAW = json.load(open(os.environ.get("RAW_FILE", "out/llm_prior_raw_landmark.json")))
CAL_OUT = os.environ.get("CAL_OUT", "")          # with CALIB_CV=fit: write only this top-label calibration file (audit reruns)
ROWS = None


def src(item_id):
    return item_id.split("|")[0]


def fit_alpha(pairs):
    def ll(a):
        t = 0.0
        for p, gi in pairs:
            q = [x ** a for x in p]
            t += math.log(q[gi] / sum(q))
        return t
    return max(GRID, key=ll) if pairs else 0.0


def fit(gold, ids, boot):
    """gold: item -> {slot: gold index}. Returns {kind: [alpha]} (one alpha = point fit; B alphas = bootstrap)."""
    by_src = {}
    for i in ids:
        by_src.setdefault(src(i), []).append(i)
    srcs = sorted(by_src)
    rng = random.Random(0)
    draws = [srcs] if not boot else [[rng.choice(srcs) for _ in srcs] for _ in range(B)]
    out = {k: [] for k in KINDS}
    for d in draws:
        sel = [i for s in d for i in by_src[s]]
        for k in KINDS:
            out[k].append(fit_alpha([(RAW[i][v], g) for i in sel for v, g in gold[i].items()
                                     if KIND.get(v, v) == k and v in RAW.get(i, {})]))
    return out


def sig(x):
    return 1 / (1 + math.exp(-x))


def logit(x):
    x = min(max(x, 1e-6), 1 - 1e-6)
    return math.log(x / (1 - x))


def power(p, a):
    q = [x ** a for x in p]
    return [x / sum(q) for x in q]


def fit_top(pairs, a):
    """Platt scaling of the top label of p^a: (a0, b) maximising the likelihood of "top = gold"."""
    ev = []
    for p, gi in pairs:
        q = power(p, a)
        t = max(range(len(q)), key=lambda i: q[i])
        ev.append((logit(q[t]), t == gi))
    if not ev:
        return (0.0, 1.0)
    def ll(c):
        return sum(math.log(max(1e-12, sig(c[0] + c[1] * z) if y else 1 - sig(c[0] + c[1] * z))) for z, y in ev)
    return max(((i / 20, j / 20) for i in range(-60, 61) for j in range(0, 61)), key=ll)


def toplabel_prior(p, a, c):
    q = power(p, a)
    t = max(range(len(q)), key=lambda i: q[i])
    top = sig(c[0] + c[1] * logit(q[t]))
    rest = 1 - q[t]
    return [top if i == t else (x / rest * (1 - top) if rest > 0 else (1 - top) / (len(q) - 1)) for i, x in enumerate(q)]


def fit_toplabel(gold, ids):
    point = fit(gold, ids, False)
    out = {}
    for k in KINDS:
        a = point[k][0]
        pairs = [(RAW[i][v], g) for i in ids for v, g in gold[i].items() if KIND.get(v, v) == k and v in RAW.get(i, {})]
        out[k] = {"alpha": a, "platt": list(fit_top(pairs, a)) if k != "area" else None}
    return out


def slot_prior(p, alphas):
    if isinstance(alphas, dict):                 # top-label calibration
        return power(p, alphas["alpha"]) if alphas["platt"] is None else toplabel_prior(p, alphas["alpha"], alphas["platt"])
    acc = [0.0] * len(p)
    for a in alphas:
        q = [x ** a for x in p]
        z = sum(q)
        acc = [s + x / z for s, x in zip(acc, q)]
    return [x / len(alphas) for x in acc]


def built(it):
    ex = Executor("out/cache")
    r = build(it, ROWS, ex)
    if r is None:
        return None
    topo, b, g = r
    return it, topo, b, list(g)


def run(job):
    it, topo, b0, gold, cals = job
    seed = int(hashlib.md5((it["item_id"] if "anchor_kinds" in it else it["sample_id"]).encode()).hexdigest()[:8], 16)
    g = next(i for i, h in enumerate(b0.hyps) if list(h.assign) == gold)
    out = []
    for pname, cal in cals.items():
        P = {v: slot_prior(RAW[it["item_id"]][v], cal[KIND.get(v, v)]) for v in topo.order
             if len(topo.slots[v].readings) >= 2 and v in RAW.get(it["item_id"], {})}
        w = []
        for h in b0.hyps:
            x = 1.0
            for v, r in zip(b0.slots, h.assign):
                if v in P:
                    x *= P[v][r]
            w.append(x)
        b = Belief(topo, b0.hyps, prior=w)
        amb = [v for v in b.slots if b.splits(b.root, v)]
        for K in (1, 2, 3, None):
            st = Setting(b, askable=amb, budget=K, lam=E.LAM, noise=E.NOISE, show_cost=None)
            for m, mk in (("ShapleyRollout", ShapleyRollout), ("Info gain (classes)", EIGClasses), ("Greedy", Greedy)):
                for s in range(SEEDS):
                    e = run_episode(st, mk(), SimulatedUser(b, g, noise=E.NOISE, seed=s * 1000003 + seed))
                    ex_ = float(b.hyps[e.submitted].result == b.hyps[g].result)
                    out.append({"id": it["item_id"], "bench": "m4" if "anchor_kinds" in it else "m3", "prior": pname,
                                "method": m, "K": K, "seed": s, "EX": ex_, "Q": e.asks})
    return out


def main():
    global ROWS
    ROWS = {r["sample_id"]: r for r in load_rows()}
    items = [json.loads(l) for b in ("m3", "m4") for l in open("out/ambioverpass_%s.jsonl" % b)]
    items = [it for it in items if it["split"] == "dev"]
    with Pool(16) as pool:
        B_ = [x for x in pool.map(built, items) if x]
    gold = {it["item_id"]: dict(zip(topo.order, g)) for it, topo, b, g in B_}
    mode = os.environ.get("CALIB_CV", "dev")
    if mode == "fit":
        ids = sorted(gold)
        if CAL_OUT:              # CAL_KIND = toplabel | boot | point: one calibration file, {kind: dict | [alphas]}
            ck = os.environ.get("CAL_KIND", "toplabel")
            res = fit_toplabel(gold, ids) if ck == "toplabel" else fit(gold, ids, ck == "boot")
            json.dump(res, open(CAL_OUT, "w"))
            print(ck, {k: (v if isinstance(v, dict) else "%d alphas, mean %.2f" % (len(v), S.mean(v))) for k, v in res.items()})
            return
        res = {"point": fit(gold, ids, False), "boot": fit(gold, ids, True), "toplabel": fit_toplabel(gold, ids)}
        json.dump(res, open("out/prior_calib_boot.json", "w"))
        json.dump(res["toplabel"], open("out/prior_calib_toplabel.json", "w"))
        print("toplabel", res["toplabel"])
        for k in KINDS:
            a = res["boot"][k]
            print("%-6s point %.2f | bootstrap mean %.2f, 5-95%% [%.2f, %.2f]" % (k, res["point"][k][0], S.mean(a),
                  sorted(a)[int(0.05 * B)], sorted(a)[int(0.95 * B) - 1]))
        return
    srcs = sorted({src(i) for i in gold})
    random.Random(0).shuffle(srcs)
    fold = {s: j % FOLDS for j, s in enumerate(srcs)}
    jobs = []
    for f in range(FOLDS):
        train = [i for i in gold if fold[src(i)] != f]
        cals = {"point": fit(gold, train, False), "boot": fit(gold, train, True), "toplabel": fit_toplabel(gold, train)}
        print("fold %d: train sources %d | point %s | boot type mean %.2f" % (
            f, len({src(i) for i in train}), {k: v[0] for k, v in cals["point"].items()}, S.mean(cals["boot"]["type"])), flush=True)
        print("   toplabel platt", {k: v["platt"] for k, v in cals["toplabel"].items()}, flush=True)
        jobs += [(it, topo, b, g, cals) for it, topo, b, g in B_ if fold[src(it["item_id"])] == f]
    with Pool(16) as pool, open(os.environ.get("CV_OUT", "out/calib_cv_dev.jsonl"), "w") as fo:
        for rs in pool.imap_unordered(run, jobs):
            for r in rs:
                fo.write(json.dumps(r) + "\n")
    summarize()


def summarize():
    R = [json.loads(l) for l in open(os.environ.get("CV_OUT", "out/calib_cv_dev.jsonl"))]
    cell = {}
    for r in R:
        f = r["EX"] - E.LAM * r["Q"] if r["K"] is None else r["EX"]
        cell.setdefault((r["bench"], r["K"], r["method"], r["prior"]), {}).setdefault(r["id"], []).append(f)
    PR = ("point", "boot", "toplabel")
    tot = {p: [] for p in PR}
    for bench in ("m3", "m4"):
        for K in (1, 2, 3, None):
            line = []
            for m in ("ShapleyRollout", "Info gain (classes)", "Greedy"):
                v = {p: cell[(bench, K, m, p)] for p in PR}
                ids = sorted(v["point"])
                mp = {p: S.mean(S.mean(v[p][i]) for i in ids) for p in v}
                if m == "ShapleyRollout":
                    for p in mp:
                        tot[p].append(mp[p])
                line.append("%s %s" % (m, " / ".join("%.3f" % mp[p] for p in PR)))
            print("%s K=%s (n=%d) | %s" % (bench, K, len(ids), " | ".join(line)))
    print("priors: point / bootstrap predictive / top-label calibrated")
    for p in PR[1:]:
        print("final planner, mean over 8 cells: point %.4f | %s %.4f -> %s" % (
            S.mean(tot["point"]), p, S.mean(tot[p]), "ADOPT " + p if S.mean(tot[p]) >= S.mean(tot["point"]) else "keep point"))


if __name__ == "__main__":
    main()
