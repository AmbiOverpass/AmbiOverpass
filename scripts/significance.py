"""Paired bootstrap (over benchmark items) for the key comparisons in the main table.
Each item's numbers are averages over the same simulated truths for every method, so differences
are paired per item."""
import json
import random
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "out/structural_bench_uniform.jsonl"
R = [json.loads(l) for l in open(path)]
R = [r for r in R if "skip" not in r and r["ambiguous_slots"]]
COLS = [("1 question", "ask=all|B=1|noise=0.0|lam=0.001|show=None", "success", 1),
        ("2 questions", "ask=all|B=2|noise=0.0|lam=0.001|show=None", "success", 1),
        ("2 questions, noisy", "ask=all|B=2|noise=0.2|lam=0.001|show=None", "success", 1),
        ("asks to certainty", "ask=all|B=None|noise=0.0|lam=0.001|show=None", "asks", -1),
        ("realistic utility", "ask=taxonomy|B=None|noise=0.2|lam=0.050|show=0.05", "utility", 1)]
PAIRS = [("greedy_ec2", "ask_all"), ("greedy_ec2", "eig_readings"), ("topo_mcts", "plain_mcts"),
         ("topo_mcts", "greedy_ec2"), ("optimal", "topo_mcts")]
rng = random.Random(0)
print("# paired bootstrap, %s, n=%d items, 2000 resamples (difference A - B; for asks, lower is better)" % (path, len(R)))
print("| comparison | " + " | ".join(c[0] for c in COLS) + " |")
print("| --- |" + " --- |" * len(COLS))
for a, b in PAIRS:
    cells = []
    for _, key, metric, sign in COLS:
        d = [r["runs"][key][a][metric] - r["runs"][key][b][metric] for r in R if key in r["runs"]]
        boots = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(2000))
        lo, hi = boots[50], boots[1949]
        sig = "*" if (lo > 0 or hi < 0) else ""
        cells.append("%+.3f [%+.3f, %+.3f]%s" % (statistics.mean(d), lo, hi, sig))
    print("| %s − %s | %s |" % (a, b, " | ".join(cells)))
print("\n* = 95% interval excludes zero")
