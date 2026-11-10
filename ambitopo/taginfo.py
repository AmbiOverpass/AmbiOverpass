"""Global OSM tag statistics from taginfo (taginfo.openstreetmap.org), used to ground LLM-proposed tags:
a tag no OSM object carries is a hallucination and cannot be a reading.

One request per key (`key/values`, top 999 values by count) covers most lookups; values outside the
top 999 fall back to one `tag/stats` request. Everything is cached on disk; requests are throttled.
Runs where taginfo is reachable (needs internet access).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request

API = "https://taginfo.openstreetmap.org/api/4/"
UA = "ambitopo/0.1 (research)"
KEY_RE = re.compile(r"^[a-z0-9_:.\-]+$")


class TagInfo:
    def __init__(self, cache_dir: str = "out/taginfo", min_interval: float = 0.25):
        self.dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.min_interval = min_interval
        self._last = 0.0
        self._net = threading.Lock()                       # one request at a time, throttled
        self._key_locks: dict[str, threading.Lock] = {}
        self._mem: dict[str, dict] = {}
        self._guard = threading.Lock()

    def _get(self, path: str, params: dict) -> dict:
        url = API + path + "?" + urllib.parse.urlencode(params)
        with self._net:
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": UA})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        out = json.loads(r.read())
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 + 3 * attempt)
            self._last = time.time()
        return out

    def _cached(self, name: str, fetch) -> dict:
        with self._guard:
            if name in self._mem:
                return self._mem[name]
            lock = self._key_locks.setdefault(name, threading.Lock())
        with lock:
            if name in self._mem:
                return self._mem[name]
            fn = os.path.join(self.dir, urllib.parse.quote(name, safe="") + ".json")
            if os.path.exists(fn):
                d = json.load(open(fn))
            else:
                d = fetch()
                json.dump(d, open(fn, "w"))
            with self._guard:
                self._mem[name] = d
            return d

    def _values(self, key: str) -> dict:
        def fetch():
            d = self._get("key/values", {"key": key, "page": 1, "rp": 999, "sortname": "count_all",
                                         "sortorder": "desc"})
            return {"total": d.get("total", 0), "values": {x["value"]: x["count"] for x in d.get("data", [])}}
        return self._cached("key=" + key, fetch)

    def count(self, tag: str) -> int:
        """Number of OSM objects carrying key=value (0 = no such tag in OSM)."""
        if "=" not in tag:
            return 0
        key, val = tag.split("=", 1)
        if not KEY_RE.match(key) or not val:
            return 0
        kv = self._values(key)
        if val in kv["values"]:
            return kv["values"][val]
        if kv["total"] <= len(kv["values"]):
            return 0
        d = self._cached("tag=" + tag, lambda: self._get("tag/stats", {"key": key, "value": val}))
        return next((x["count"] for x in d.get("data", []) if x.get("type") == "all"), 0)
