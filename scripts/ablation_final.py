"""Ablation of the final system on landmark TEST + test2.

  AMBI_EXTRA_ROWS=out/onl_rows.jsonl python scripts/ablation_final.py   -> out/ablation_final.jsonl, report on stdout
  python scripts/ablation_final.py report                               -> out/ablation_final.md
"""
import collections
import json
import os
import random
import statistics as S
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import three_baselines as T
sys.argv = _argv
from ablation_sr import keep_empty_belief
from ambitopo import Belief, Setting, ShapleyGreedy, ShapleyRollout, SimulatedUser

KIND = {"anchor1": "anchor", "anchor2": "anchor", "dist1": "dist", "dist2": "dist"}
# ABL_PRIOR=perm4: the 4-order averaged LLM prior with its DEV calibration (2026-10-22 audit); the power-only
# calibration (A3) is the point alpha of the same DEV fit
_V = os.environ.get("ABL_PRIOR", "")
if _V:
    RAW = {**json.load(open("out/llm_prior_raw_landmark_%s.json" % _V)), **json.load(open("out/llm_prior_raw_test2_%s.json" % _V))}
    # main calibration = the one selected by the DEV source CV (bootstrap predictive for perm4); A3 = point power
    CAL = {"toplabel": json.load(open("out/prior_calib_boot_%s.json" % _V)),
           "power": {k: v[0] for k, v in json.load(open("out/prior_calib_point_%s.json" % _V)).items()}}
else:
    RAW = {**json.load(open("out/llm_prior_raw_landmark.json")), **json.load(open("out/llm_prior_raw_test2.json"))}
    CAL = {"toplabel": json.load(open("out/prior_calib_toplabel.json")), "power": json.load(open("out/prior_calib_landmark.json"))}
VARIANTS = ("full", "A1 no pruning", "A2 no result classes", "A3 single power calibration", "A4 uniform prior",
            "A5 Shapley -> EC2 base", "A6 no look-ahead")
BUDGETS = (1, 2, 3, None)
OUT = "out/ablation_final%s.jsonl" % (("_" + _V) if _V else "")


def prior(topo, item_id, cal):
    P = {}
    for v in topo.order:
        n = len(topo.slots[v].readings)
        if n < 2:
            continue
        if cal is None:
            P[v] = [1.0 / n] * n
            continue
        a = cal.get(KIND.get(v, v), 0.0)
        p = RAW.get(item_id, {}).get(v, [1.0 / n] * n)
        if isinstance(a, list):          # bootstrap predictive calibration
            acc = [0.0] * n
            for b in a:
                acc = [t + x for t, x in zip(acc, T.E.toplabel(p, b, None))]
            P[v] = [t / len(a) for t in acc]
            continue
        P[v] = T.E.toplabel(p, a["alpha"], a["platt"]) if isinstance(a, dict) else T.E.toplabel(p, a, None)
    return P


def weights(bb, P):
    out = []
    for h in bb.hyps:
        x = 1.0
        for v, r in zip(bb.slots, h.assign):
            if v in P:
                x *= P[v][r]
        out.append(x)
    return out


def one(job):
    bench, split, it = job
    got = T.item_belief(bench, it)
    if got is None:
        return []
    topo, hyps, _, g, base = got
    P = {k: prior(topo, it["item_id"], CAL[k]) for k in CAL}
    P["uniform"] = prior(topo, it["item_id"], None)
    proto = Belief(topo, hyps)
    w = weights(proto, P["toplabel"])
    gold_assign = tuple(hyps[g].assign)
    variants = {"full": (Belief(topo, hyps, prior=w), ShapleyRollout),
                "A1 no pruning": (keep_empty_belief(bench, it, lambda bb: weights(bb, P["toplabel"])), ShapleyRollout),
                "A2 no result classes": (Belief(topo, hyps, prior=w, threshold=1.01), ShapleyRollout),
                "A3 single power calibration": (Belief(topo, hyps, prior=weights(proto, P["power"])), ShapleyRollout),
                "A4 uniform prior": (Belief(topo, hyps, prior=weights(proto, P["uniform"])), ShapleyRollout),
                "A5 Shapley -> EC2 base": (Belief(topo, hyps, prior=w), lambda: ShapleyRollout(base="greedy")),
                "A6 no look-ahead": (Belief(topo, hyps, prior=w), ShapleyGreedy)}
    src = it["sample_id"]
    out = []
    for name, (b, mk) in variants.items():
        if b is None:
            continue
        gi = next(i for i, h in enumerate(b.hyps) if tuple(h.assign) == gold_assign)
        amb = [v for v in b.slots if b.splits(b.root, v)]
        for K in BUDGETS:
            st = Setting(b, askable=amb, budget=K, lam=T.LAM, noise=T.NOISE, show_cost=None)
            for seed in range(T.SEEDS):
                user = SimulatedUser(b, gi, noise=T.NOISE, seed=seed * 1000003 + base)
                cls, q, t, stop, asked, trace = T.episode(st, mk(), user, K)
                sub = [i for i in b.root if b.cls[i] == cls]
                ex = float(b.hyps[sub[0]].result == b.hyps[gi].result) if sub else 0.0
                out.append({"id": it["item_id"], "src": src, "bench": bench, "split": split, "variant": name,
                            "K": "None" if K is None else K, "seed": seed, "EX": ex, "Q": q, "ms": t})
    return out


def cboot(diffs_by_src, n=4000):
    S_ = list(diffs_by_src.values())
    rng = random.Random(0)
    xs = []
    for _ in range(n):
        smp = [S_[rng.randrange(len(S_))] for _ in S_]
        xs.append(sum(sum(s) for s in smp) / sum(len(s) for s in smp))
    xs.sort()
    tot = sum(sum(s) for s in S_) / sum(len(s) for s in S_)
    return tot, xs[int(0.025 * n)], xs[int(0.975 * n)]


def report():
    per = collections.defaultdict(list)
    meta = {}
    for l in open(OUT):
        r = json.loads(l)
        f = r["EX"] - T.LAM * r["Q"] if r["K"] == "None" else r["EX"]
        per[(r["bench"], str(r["K"]), r["variant"], r["id"])].append((f, r["Q"]))
        meta[r["id"]] = (r["src"], r["split"])
    lines = ["Delta = variant - full (points), 95% CI clustered by source request; * = CI excludes 0.", ""]
    for bench, label in (("m3", "single landmark"), ("m4", "two landmarks")):
        for part in ("union", "test", "test2"):
            ids = sorted({i for (b, K, v, i) in per if b == bench and v == "full" and (part == "union" or meta[i][1] == part)})
            lines.append("### %s, %s (n = %d)" % (label, part, len(ids)))
            lines.append("| variant | K=1 | K=2 | K=3 | no budget |")
            lines.append("| --- | --- | --- | --- | --- |")
            for v in VARIANTS:
                row = []
                for K in ("1", "2", "3", "None"):
                    use = [i for i in ids if (bench, K, v, i) in per]
                    if not use:
                        row.append("-")
                        continue
                    m = S.mean(S.mean(x[0] for x in per[(bench, K, v, i)]) for i in use)
                    q = S.mean(S.mean(x[1] for x in per[(bench, K, v, i)]) for i in use)
                    if v == "full":
                        row.append("%.3f [%.2f q]" % (m, q))
                        continue
                    d = collections.defaultdict(list)
                    for i in use:
                        d[meta[i][0]].append(S.mean(x[0] for x in per[(bench, K, v, i)]) - S.mean(x[0] for x in per[(bench, K, "full", i)]))
                    t, lo, hi = cboot(d)
                    row.append("%.3f (%+.1f%s)" % (m, 100 * t, "*" if lo > 0 or hi < 0 else ""))
                lines.append("| %s | %s |" % (v, " | ".join(row)))
            lines.append("")
    open(OUT.replace(".jsonl", ".md"), "w").write("\n".join(lines))
    print("\n".join(lines))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
        return
    jobs = []
    for bench, files in (("m3", (("test", "out/ambioverpass_m3.jsonl"), ("test2", "out/ambioverpass_m3x.jsonl"))),
                         ("m4", (("test", "out/ambioverpass_m4.jsonl"), ("test2", "out/ambioverpass_m4x.jsonl")))):
        for split, f in files:
            for l in open(f):
                it = json.loads(l)
                if it.get("split") == split:
                    jobs.append((bench, split, it))
    random.Random(0).shuffle(jobs)          # mix slow two-landmark items across workers
    with Pool(int(os.environ.get("NPROC", 32))) as pool, open(OUT, "w") as f:
        for recs in pool.imap_unordered(one, jobs):
            for r in recs:
                f.write(json.dumps(r) + "\n")
    report()


if __name__ == "__main__":
    main()
