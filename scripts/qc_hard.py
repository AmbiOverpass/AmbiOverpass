"""Quality control for AmbiOverpass-H / H2 items (adds to `flags`, writes <file>_qc.jsonl):
  place dropped     none of the place words of the clear request (capitalised or non-Latin, not the first
                    word) survives in the hard request
  not recoverable   an LLM judge (shown the gold kind) says a careful reader could not work it out
The judge sees the answer, so it errs towards "recoverable"; it only removes clearly broken items.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ambitopo.llm import OpenAICompatible

JUDGE = ("Request: \"%s\"\nCould a careful reader work out that the user mainly wants %s (it may be one of a few "
         "plausible readings)? Answer only yes or no.")


STOP = {"What", "Which", "Where", "Show", "List", "Find", "How", "Are", "Is", "The", "All", "OpenStreetMap", "OSM"}


def place_words(clear):
    toks = re.findall(r"[^\s,.?!()'\"]+", clear)[1:]
    return [t for t in toks if len(t) >= 3 and t not in STOP and (t[0].isupper() or any(ord(c) > 127 for c in t))]


def place_kept(it):
    words = place_words(it["x_clear"]) or [it["area_name"] or ""]
    return any(w.lower() in it["x_hard"].lower() for w in words)


def main():
    src = sys.argv[1]
    llm = OpenAICompatible()
    items = [json.loads(l) for l in open(src)]

    def qc(it):
        it = dict(it, flags=list(it["flags"]))
        if it["area_name"] is not None and not place_kept(it):
            it["flags"].append("place dropped")
        kind = "%s (OpenStreetMap %s)" % (it["value"].replace("_", " "), it["gold_tag"])
        ans = llm.chat([{"role": "user", "content": JUDGE % (it["x_hard"], kind)}], temperature=0.0, max_tokens=5)
        it["judge"] = ans.strip().lower()
        if not it["judge"].startswith("yes"):
            it["flags"].append("not recoverable")
        return it

    with ThreadPoolExecutor(16) as pool:
        out = list(pool.map(qc, items))
    dst = src.replace(".jsonl", "_qc.jsonl")
    with open(dst, "w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    c = lambda k: sum(k in it["flags"] for it in out)
    print("%s: %d items | place dropped %d | not recoverable %d | pass all %d"
          % (dst, len(out), c("place dropped"), c("not recoverable"), sum(not it["flags"] for it in out)))


if __name__ == "__main__":
    main()
