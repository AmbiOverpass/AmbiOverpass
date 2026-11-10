"""Ablation of the final planner (one-step Shapley look-ahead = ShapleyRollout) on the TEST splits, with the SAME items,
priors and user random streams as the paper's main runs (so the full row equals the main table).
Variants: full; no validity pruning (empty readings kept); no result classes (plan over raw readings);
Shapley -> one-step importance (look-ahead whose base policy is the EC2 topology greedy).
Also: the case study (Lons-le-Saunier hydrants, truthful user, no budget) for topology greedy vs the final planner.

  CALIB_FILE=out/prior_calib_landmark.json python scripts/ablation_sr.py   -> out/ablation_sr.jsonl, out/case_sr.json
"""
import hashlib
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import three_baselines as T
from ambitopo import Belief, Greedy, Setting, ShapleyRollout, SimulatedUser
from ambitopo.data import to_sample
from ambitopo.near import build_near, build_near2
from ambitopo.overpass import Executor
from ambitopo.planners import shapley_values, coalition_gain, one_step_ask
from ambitopo.readings import lookup_areas

CELLS = {"m3": (3, None), "m4": (3, None)}


def keep_empty_belief(b_name, it, w_of):
    s = to_sample(T.ROWS[it["sample_id"]])
    ex = Executor("out/cache")
    areas = lookup_areas(s.spec.area_name, ex, offline=True)
    if b_name == "m3":
        kb = build_near(s.spec, s.tag_candidates, areas, it["anchor_kind"], ex, offline=True, seed=0, gold=it["gold"], keep_empty=True)
    else:
        kb = build_near2(s.spec, s.tag_candidates, areas, tuple(it["anchor_kinds"]), ex, offline=True, seed=0, gold=it["gold"], keep_empty=True)
    if kb is None:
        return None
    topo, b, _ = kb
    return Belief(topo, b.hyps, prior=w_of(b))


def one(job):
    b_name, it = job
    got = T.item_belief(b_name, it)
    if got is None:
        return []
    topo, hyps, w, g, base = got
    P = T.E.llm_prior(None, it["x_vague"], topo, it["item_id"])

    def w_of(bb):
        out = []
        for h in bb.hyps:
            x = 1.0
            for v, r in zip(bb.slots, h.assign):
                if v in P:
                    x *= P[v][r]
            out.append(x)
        return out
    gold_assign = tuple(hyps[g].assign)
    variants = {"full": Belief(topo, hyps, prior=w), "no_classes": Belief(topo, hyps, prior=w, threshold=1.01),
                "shapley_to_ec2": Belief(topo, hyps, prior=w), "no_pruning": keep_empty_belief(b_name, it, w_of)}
    out = []
    for name, b in variants.items():
        if b is None:
            continue
        gi = next(i for i, h in enumerate(b.hyps) if tuple(h.assign) == gold_assign)
        amb = [v for v in b.slots if b.splits(b.root, v)]
        for K in CELLS[b_name]:
            st = Setting(b, askable=amb, budget=K, lam=T.LAM, noise=T.NOISE, show_cost=None)
            for seed in range(T.SEEDS):
                pol = ShapleyRollout(base="greedy" if name == "shapley_to_ec2" else "shapley")
                user = SimulatedUser(b, gi, noise=T.NOISE, seed=seed * 1000003 + base)
                cls, q, t, stop, asked, trace = T.episode(st, pol, user, K)
                # success = the submitted class returns exactly the gold objects (raw readings: compare results)
                sub = [i for i in b.root if b.cls[i] == cls]
                ex = float(b.hyps[sub[0]].result == b.hyps[gi].result) if sub else 0.0
                out.append({"id": it["item_id"], "bench": b_name, "variant": name, "K": "None" if K is None else K, "seed": seed,
                            "EX": ex, "Q": q})
    return out


def case():
    it = next(json.loads(l) for l in open("out/ambioverpass_m4.jsonl") if json.loads(l)["item_id"] == "train_04628|near:park+supermarket")
    topo, hyps, w, g, base = T.item_belief("m4", it)
    b = Belief(topo, hyps, prior=w)
    amb = [v for v in b.slots if b.splits(b.root, v)]
    st = Setting(b, askable=amb, budget=None, lam=T.LAM, noise=T.NOISE, show_cost=None)
    phi = shapley_values(b, b.root, amb)
    rec = {"acc_root": b.acc_stop(b.root), "phi": phi, "delta1": {v: coalition_gain(b, b.root, (v,)) for v in amb},
           "net1": {v: one_step_ask(st, b.root, v) for v in amb}, "delta_all": coalition_gain(b, b.root, tuple(amb))}
    for name, pol in (("greedy", Greedy()), ("planner", ShapleyRollout())):
        alive, used, log = b.root, 0, []
        while True:
            a = pol.act(st, alive, used)
            if a.kind != "ask":
                break
            alive = b.observe(alive, a.slot, b.value(g, a.slot)) or alive
            used += 1
            log.append([a.slot, round(b.acc_stop(alive), 3)])
        cls = b.map_class(alive)
        rec[name] = {"asked": log, "correct": cls == b.cls[g], "final_acc": b.acc_stop(alive)}
    json.dump(rec, open("out/case_sr.json", "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    case()
    jobs = [(b, json.loads(l)) for b in CELLS for l in open("out/ambioverpass_%s.jsonl" % b)]
    jobs = [(b, it) for b, it in jobs if it.get("split") == "test"]
    with Pool(48) as pool, open("out/ablation_sr.jsonl", "w") as f:
        for recs in pool.imap_unordered(one, jobs):
            for r in recs:
                f.write(json.dumps(r) + "\n")
    print("done")
