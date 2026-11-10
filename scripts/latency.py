"""Planning latency per decision on the benchmark items (realistic setting), and the exact
planner's cold-start cost. Execution latency is reported separately from the collection logs."""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo import Greedy, Optimal, PlainMCTS, Setting, TopoMCTS
from ambitopo.data import build, load_rows, to_sample
from ambitopo.overpass import Executor

bench = {}
for l in open("out/ambioverpass_m.jsonl"):
    it = json.loads(l)
    bench[it["sample_id"]] = tuple(v if v != "dist" else "area" for v in it["weakened"])
ex = Executor("out/cache")
T = {"greedy": [], "topo_mcts@30": [], "plain_mcts@30": [], "topo_mcts@100": [], "optimal_cold": []}
for row in load_rows():
    s = to_sample(row)
    if s is None or s.sample_id not in bench:
        continue
    build(s, ex, vary=bench[s.sample_id], offline=True)
    if s.belief is None:
        continue
    b = s.belief
    amb = [v for v in b.slots if b.splits(b.root, v)]
    st = Setting(b, askable=[v for v in amb if v in ("area", "tag", "dist")], lam=0.05, noise=0.2, show_cost=0.05)
    for name, pol in (("greedy", Greedy()), ("topo_mcts@30", TopoMCTS(n_sim=30)),
                      ("plain_mcts@30", PlainMCTS(n_sim=30)), ("topo_mcts@100", TopoMCTS(n_sim=100))):
        t0 = time.perf_counter()
        for _ in range(5):
            pol.act(st, b.root, 0)
        T[name].append((time.perf_counter() - t0) / 5 * 1000)
    t0 = time.perf_counter()
    Optimal().act(st, b.root, 0)
    T["optimal_cold"].append((time.perf_counter() - t0) * 1000)
for k, v in T.items():
    print("%-15s ms per decision: mean %.2f  median %.2f  max %.1f  (n=%d)"
          % (k, statistics.mean(v), statistics.median(v), max(v), len(v)))
