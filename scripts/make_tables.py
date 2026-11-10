"""Rebuild the numbers of Tables III-V and the analysis of the paper from the per-request result files in out/.
Run from the repository root:

  python3 scripts/make_tables.py        -> prints Tables III and IV, writes out/paper_numbers.json

Landmark subsets: union of the original TEST split and the fresh test2 split, top-label calibrated prior for every
method (out/eval_{m3,m4}_llmcal_test_toplabel.json + out/eval_{m3,m4}x_llmcal_test2.json). Single-object subset: the
paper's TEST runs (docs_src/exp_final.json, unchanged). Ablation: out/ablation_final.jsonl. All CIs are 95% bootstrap
CIs clustered by source request (items generated from one OverpassNL query are resampled together).
"""
import collections
import json
import random
import statistics as S

import os

LAM, TOP = 0.05, "ShapleyRollout"
# PRIOR=perm4: the 4-order averaged LLM prior, calibration refitted on DEV (official setting from 2026-10-22, see
# notes/audit_20261022.md) -> docs_src/exp_paper_perm4.json; unset: the earlier prior -> docs_src/exp_paper.json
PR = os.environ.get("PRIOR", "iterC1")     # official setting of the paper
LM_TEST = "out/eval_%%s_llmcal_test_audit_%s.json" % PR if PR else "out/eval_%s_llmcal_test_toplabel.json"
LM_T2 = "out/eval_%%sx_llmcal_test2_audit_%s.json" % PR if PR else "out/eval_%sx_llmcal_test2.json"
SPR, APR, CPR = (os.environ.get(k, d) for k, d in (("SINGLE", "iter4"), ("ABL", "iterC1"), ("CASE", "iterC1b")))     # tags of the single-object runs, ablation, case
KS = ("1", "2", "3", "None")


LONG = {"Ask every ambiguity (Sphinteract/AmbiSQL)": "Ask-all", "Ask until results agree (ClarifyGPT)": "ClarifyGPT-style",
        "Info gain over readings (Qiu et al.)": "Info gain (Qiu)", "LLM planner (Qwen3-14B)": "LLM planner",
        "Generic MCTS (VisInteract)": "Generic MCTS", "Greedy (topology importance)": "Greedy", "Exact planner": "Exact",
        "ReAct (Qwen3-14B)": "ReAct", "Info gain over result classes": "Info gain (classes)", "Direct": "Direct",
        "ShapleyGreedy": "ShapleyGreedy", "ShapleyRollout": "ShapleyRollout", "TopoMCTS-SV": "TopoMCTS-SV"}


def per_item(files, rename=None):
    per = {}
    for f in files:
        for r in json.load(open(f)):
            for key, eps in r["res"].items():
                m, k = key.split("|")
                if rename is not None:
                    if m not in rename:
                        continue
                    m = rename[m]
                K = k[2:]
                g = (lambda x: x["EX"] - LAM * x["Q"]) if K == "None" else (lambda x: x["EX"])
                per.setdefault(m, {})[(r["id"], K)] = (S.mean(g(x) for x in eps), S.mean(x["Q"] for x in eps),
                                                       S.mean(x["EX"] for x in eps))
    return per


def cboot(groups, n=4000):
    G = list(groups.values())
    rng = random.Random(0)
    xs = []
    for _ in range(n):
        smp = [G[rng.randrange(len(G))] for _ in G]
        xs.append(sum(sum(s) for s in smp) / sum(len(s) for s in smp))
    xs.sort()
    return [sum(sum(s) for s in G) / sum(len(s) for s in G), xs[int(0.025 * n)], xs[int(0.975 * n)]]


def cell(per, K):
    ids = sorted(set.intersection(*[{i for i, k in d if k == K} for d in per.values()]))
    out = {"n": len(ids), "src": len({i.split("|")[0] for i in ids}), "mean": {}, "Q": {}, "EX": {}, "sig": {}}
    for m, d in per.items():
        out["mean"][m] = S.mean(d[(i, K)][0] for i in ids)
        out["Q"][m] = S.mean(d[(i, K)][1] for i in ids)
        out["EX"][m] = S.mean(d[(i, K)][2] for i in ids)
        if m != TOP:
            g = collections.defaultdict(list)
            for i in ids:
                g[i.split("|")[0]].append(per[TOP][(i, K)][0] - d[(i, K)][0])
            out["sig"][m] = cboot(g)
    return out


def main():
    P = {"single": {}}
    for K in ("1", "2"):
        if PR:
            per = per_item(["out/main_table_m2b_test_llmcal_gold_K%s_audit_%s.json" % (K, SPR)], LONG)
        else:
            per = per_item(["out/main_table_m2b_test_llmcal_gold_K%s.json" % K, "out/main_table_m2b_test_llmcal_gold_K%s_newbase.json" % K], LONG)
            per["Ask-all"] = per_item(["out/main_table_m2b_test_llmcal_gold_K%s_askretry.json" % K], LONG)["Ask-all"]   # Ask-all re-asks after "no preference" 
        P["single"][K] = cell(per, K)
    for b in ("m3", "m4"):
        test = per_item([LM_TEST % b])
        t2 = per_item([LM_T2 % b])
        if not PR:      # Ask-all re-asks a slot after "no preference" like every other policy 
            test["Ask-all"] = per_item(["out/eval_%s_llmcal_test_toplabel_askretry.json" % b])["Ask-all"]
            t2["Ask-all"] = per_item(["out/eval_%sx_llmcal_test2_askretry.json" % b])["Ask-all"]
        union = {m: {**test[m], **t2[m]} for m in test if m in t2}
        P[b] = {part: {K: cell(per, K) for K in KS} for part, per in (("union", union), ("test", test), ("test2", t2))}
    # ablation (union), delta vs full, clustered by source
    rows = collections.defaultdict(list)
    src = {}
    for l in open("out/ablation_final_%s.jsonl" % APR if PR else "out/ablation_final.jsonl"):
        r = json.loads(l)
        f = r["EX"] - LAM * r["Q"] if r["K"] == "None" else r["EX"]
        rows[(r["bench"], str(r["K"]), r["variant"], r["id"])].append((f, r["Q"]))
        src[r["id"]] = r["src"]
    A = {}
    for b in ("m3", "m4"):
        for K in KS:
            full = {i: S.mean(x[0] for x in v) for (bb, k, var, i), v in rows.items() if bb == b and k == K and var == "full"}
            A["%s|%s" % (b, K)] = {"n": len(full)}
            for var in sorted({k[2] for k in rows}):
                vals = {i: S.mean(x[0] for x in v) for (bb, k, vv, i), v in rows.items() if bb == b and k == K and vv == var}
                q = S.mean(S.mean(x[1] for x in v) for (bb, k, vv, i), v in rows.items() if bb == b and k == K and vv == var)
                g = collections.defaultdict(list)
                for i in vals:
                    if i in full:
                        g[src[i]].append(vals[i] - full[i])
                A["%s|%s" % (b, K)][var] = {"mean": S.mean(vals.values()), "Q": q, "d": cboot(g) if var != "full" else None}
    P["abl"] = A
    # where the advantage over topology greedy comes from (number of ambiguous slots), questions without a budget
    slots = {}
    for b in ("m3", "m4"):
        for f in (LM_TEST % b, LM_T2 % b):
            for r in json.load(open(f)):
                slots[r["id"]] = r["slots"]
    W = {}
    for b in ("m3", "m4"):
        per = {m: {**per_item([LM_TEST % b])[m], **per_item([LM_T2 % b])[m]}
               for m in (TOP, "Greedy")}
        for grp, cond in (("le5", lambda x: x <= 5), ("ge6", lambda x: x >= 6)):
            for K in ("1", "3", "None"):
                ids = [i for (i, k) in per[TOP] if k == K and cond(slots[i])]
                if ids:
                    g = collections.defaultdict(list)
                    for i in ids:
                        g[i.split("|")[0]].append(per[TOP][(i, K)][0] - per["Greedy"][(i, K)][0])
                    W["%s|%s|%s" % (b, grp, K)] = {"n": len(ids), "d": cboot(g)}
    P["where"] = W
    # benchmark structure (difficulty grades, state counts) and the case study
    st = {}
    for b in ("m3", "m4"):
        R = json.load(open("out/structure_%s.json" % b)) + json.load(open("out/structure_%sx.json" % b))
        for name, RR in (("all", R), ("eval", [r for r in R if r["split"] in ("test", "test2")])):
            st["%s|%s" % (b, name)] = {"n": len(RR), "slots": S.mean(r["slots"] for r in RR), "hyps": S.mean(r["hyps"] for r in RR),
                                       "classes": S.mean(r["classes"] for r in RR),
                                       "reg": dict(collections.Counter(r["regime"] for r in RR)),
                                       "hist": S.mean(r["hist"] for r in RR), "dag": S.mean(r["dag"] for r in RR),
                                       "canon": S.mean(r["canon"] for r in RR)}
    P["struct"] = st
    P["case"] = json.load(open("out/case_%s.json" % CPR if PR else "out/case_toplabel.json"))
    json.dump(P, open("out/paper_numbers.json", "w"), indent=1)
    NAME = [("Direct", "No clarification"), ("Ask-all", "Sphinteract"), ("ClarifyGPT-style", "ClarifyGPT"), ("Info gain (Qiu)", "EIG"),
            ("LLM planner", "LLM planner"), ("ReAct", "ReAct agent"), ("Generic MCTS", "Generic MCTS"), ("Greedy", "EC2-Greedy"),
            (TOP, "AmbiOverpass")]
    print("Table III  single object (test, n=%d): EX K=1, EX K=2, #Q K=2" % P["single"]["1"]["n"])
    for m, name in NAME:
        print("  %-18s %.3f  %.3f  %.2f" % (name, P["single"]["1"]["EX"][m], P["single"]["2"]["EX"][m], P["single"]["2"]["Q"][m]))
    for b, title, ks in (("m3", "single landmark", ("2", "3")), ("m4", "two landmarks", ("1", "2", "3"))):
        print("Table IV  %s (test + held-out, n=%d): EX at K=%s, utility and #Q without a budget" % (title, P[b]["union"]["1"]["n"], ",".join(ks)))
        for m, name in NAME:
            u = P[b]["union"]
            print("  %-18s %s  %.3f  %.1f" % (name, "  ".join("%.3f" % u[k]["EX"][m] for k in ks), u["None"]["mean"][m], u["None"]["Q"][m]))
    print("Paired differences of AmbiOverpass to each method (points, * = 95% CI excludes 0):")
    for b in ("m3", "m4"):
        for K in KS:
            c = P[b]["union"][K]
            print(b, K, c["n"], c["src"], "ours %.3f" % c["mean"][TOP],
                  {m[:12]: "%+.1f%s" % (100 * v[0], "*" if v[1] > 0 or v[2] < 0 else "") for m, v in c["sig"].items()})


if __name__ == "__main__":
    main()
