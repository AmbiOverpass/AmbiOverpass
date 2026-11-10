"""Phase 1: execute every sample's readings against Overpass (single process, rate-limited).
Fills the shared cache; evaluation then runs offline and in parallel. Safe to stop and restart."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo.data import build, load_rows, to_sample
from ambitopo.overpass import Executor

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="out/cache")
ap.add_argument("--log", default="out/collect.jsonl")
ap.add_argument("--limit", type=int, default=10 ** 9)
ap.add_argument("--sleep", type=float, default=6.0)
ap.add_argument("--retry-failed", action="store_true")
ap.add_argument("--redo-failed", action="store_true",
                help="re-process samples whose earlier record had a failed request (implies --retry-failed)")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
args = ap.parse_args()


import glob
done = set()
for lp in glob.glob(os.path.join(os.path.dirname(args.log) or ".", "collect*.jsonl")):
    for l in open(lp):
        r = json.loads(l)
        failed = r["stats"].get("error") or r["stats"].get("areas_failed", 0) > 0
        if not (args.redo_failed and failed):
            done.add(r["id"])
if args.redo_failed:
    args.retry_failed = True
ex = Executor(args.cache, sleep=args.sleep, retry_failed=args.retry_failed)
samples = [s for s in (to_sample(r) for r in load_rows()) if s]
print("samples parsable: %d, already collected: %d" % (len(samples), len(done)), flush=True)
with open(args.log, "a") as log:
    for k, s in enumerate(samples[:args.limit]):
        if s.sample_id in done or k % args.nshards != args.shard:
            continue
        t0, r0 = time.time(), ex.n_requests
        build(s, ex)
        rec = {"id": s.sample_id, "stats": s.exec_stats, "requests": ex.n_requests - r0,
               "secs": round(time.time() - t0, 1),
               "belief": s.belief.summary() if s.belief else None}
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print(k, json.dumps(rec, ensure_ascii=False), flush=True)
print("done", flush=True)
