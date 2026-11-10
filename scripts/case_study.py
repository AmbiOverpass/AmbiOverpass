"""Case study of one two-landmark request under the paper's final configuration (top-label calibrated prior):
the dialogue of AmbiOverpass (ShapleyRollout) and of the baselines with a noise-free user, step by step, and the
coordinates needed to draw how the candidate objects shrink to the intended ones.

  CALIB_FILE=out/prior_calib_toplabel.json python scripts/case_study.py [ITEM_ID] [OUT_JSON]
"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import eval_m3 as E
sys.argv = _argv
from ambitopo import AskAll, Belief, EIGReadings, Greedy, Natural, Setting, ShapleyRollout
from ambitopo.planners import coalition_gain, shapley_values
from ambitopo.data import load_rows, to_sample
from ambitopo.near import _fetch, build_near2
from ambitopo.overpass import Executor
from ambitopo.readings import lookup_areas

IID = sys.argv[1] if len(sys.argv) > 1 else "train_04628|near:park+supermarket"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out/case_study.json"
it = next(json.loads(l) for f in ("out/ambioverpass_m4.jsonl", "out/ambioverpass_m4x.jsonl") if os.path.exists(f)
          for l in open(f) if json.loads(l)["item_id"] == IID)
rows = {r["sample_id"]: r for r in load_rows()}
s = to_sample(rows[it["sample_id"]])
ex = Executor("out/cache")
areas = lookup_areas(s.spec.area_name, ex, offline=True)
topo, b, gold = build_near2(s.spec, s.tag_candidates, areas, tuple(it["anchor_kinds"]), ex, offline=True,
                            seed=int(hashlib.md5(IID.encode()).hexdigest()[:8], 16), gold=it["gold"])
P = E.llm_prior(None, it["x_vague"], topo, IID)
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
label = lambda v, r: topo.slots[v].readings[r].label


def state(alive):
    cm = b.class_mass(alive)
    tot = sum(cm.values())
    obj = {}
    for i in alive:
        for o in b.hyps[i].result:
            obj[o] = obj.get(o, 0.0) + b.w[i] / tot
    return {"readings": len(alive), "classes": len(cm), "acc": b.acc_stop(alive),
            "p_gold_class": cm.get(b.cls[g], 0.0) / tot, "p_obj": obj}


def dialogue(pol, budget):
    st = Setting(b, askable=amb, budget=budget, lam=E.LAM, noise=E.NOISE, show_cost=None)
    alive, used, steps = b.root, 0, [dict(state(b.root), ask=None)]
    while True:
        cands = st.candidates(alive)
        phi = shapley_values(b, alive, cands) if cands else {}
        a = pol.act(st, alive, used)
        steps[-1]["phi"] = phi
        steps[-1]["delta1"] = {v: coalition_gain(b, alive, (v,)) for v in cands}
        if a.kind != "ask":
            break
        r = b.value(g, a.slot)
        alive, used = (b.observe(alive, a.slot, r) or alive), used + 1
        steps.append(dict(state(alive), ask=a.slot, answer=label(a.slot, r)))
    sub = b.representative(alive, b.map_class(alive))
    return {"steps": steps, "submitted": {v: label(v, x) for v, x in zip(b.slots, b.hyps[sub].assign)},
            "correct": b.hyps[sub].result == b.hyps[g].result, "n_submitted": len(b.hyps[sub].result)}


out = {"item": IID, "x_clear": it["x_clear"], "x_vague": it["x_vague"], "readings": len(b.hyps), "classes": b.n_classes,
       "slots": {v: [label(v, r) for r in range(len(topo.slots[v].readings))] for v in b.slots}, "ambiguous": amb,
       "gold": {v: label(v, x) for v, x in zip(b.slots, b.hyps[g].assign)}, "gold_objects": sorted(b.hyps[g].result),
       "prior": {v: P[v] for v in P}, "lam": E.LAM, "noise": E.NOISE, "dialogues": {}}
for name, pol in (("AmbiOverpass", ShapleyRollout()), ("Topology greedy", Greedy()), ("Ask all", AskAll()),
                  ("Ask until agree", Natural()), ("Info gain, readings", EIGReadings())):
    for K in (None, 3):
        out["dialogues"]["%s|K=%s" % (name, K)] = dialogue(pol, K)

# coordinates: targets, landmarks of the gold kinds, per area (single-anchor superset requests in the cache)
geo = {}
for kind, slot in zip(it["anchor_kinds"], ("anchor1", "anchor2")):
    tags, main, anchors, data = _fetch(s.spec, s.tag_candidates, areas, kind, ex, True)
    for ai, rws in data.items():
        for typ, eid, lat, lon, kv in rws:
            geo.setdefault(eid, {"lat": lat, "lon": lon, "area": [], "anchor": []})
            if ai not in geo[eid]["area"]:
                geo[eid]["area"].append(ai)
            for k, f in enumerate(anchors):
                if f.match(kv):
                    geo[eid]["anchor"].append([slot, f.raw])
out["geo"] = geo
out["areas"] = [{"id": a["id"], "name": a["name"], "level": a.get("admin_level") or a.get("place")} for a in areas]
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
for k, d in out["dialogues"].items():
    print("%-28s correct=%s  asked=%s" % (k, d["correct"], [(x["ask"], x["answer"], round(x["acc"], 3), x["readings"], x["classes"])
                                                          for x in d["steps"][1:]]))
print("root: readings %d classes %d acc %.3f" % (len(b.hyps), b.n_classes, b.acc_stop(b.root)))
print("geo objects", len(geo), "gold objects", len(out["gold_objects"]), "missing coords",
      sum(o not in geo for o in out["gold_objects"]))
