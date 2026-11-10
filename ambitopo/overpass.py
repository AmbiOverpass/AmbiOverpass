"""The only module that talks to Overpass.

Lessons carried over from the old AmbiOverpass executor:
  - public overpass-api.de throttles by server CPU and blocked us three times; maps.mail.ru is a
    global-data mirror reachable from our servers. Never add a regional mirror (it returns valid empty
    results outside its region, indistinguishable from a wrong query).
  - HTTP 200 can carry an error page; detect it, never record it as an empty result.
  - cap the response size; a country-wide query can return hundreds of MB.
  - the mirror's gateway cuts requests at ~45 s and fails intermittently; retry with backoff.
Successful responses (and "too big") are cached on disk by query hash, so every experiment that
reruns reads the cache instead of the network.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

ENDPOINTS = [u.strip() for u in os.environ.get(
    "AMBITOPO_ENDPOINTS", "https://maps.mail.ru/osm/tools/overpass/api/interpreter").split(",") if u.strip()]
USER_AGENT = "ambitopo/0.1 (research)"
MAX_BYTES = 12 * 1024 * 1024


@dataclass
class Response:
    ok: bool
    text: str = ""
    error: str = ""
    cached: bool = False


class Executor:
    def __init__(self, cache_dir: str, sleep: float = 6.0, retries: int = 2, http_timeout: int = 180,
                 backoff: float = 10.0, retry_failed: bool = False):
        self.cache_dir = cache_dir
        self.sleep = sleep
        self.retries = retries
        self.http_timeout = http_timeout
        self.backoff = backoff
        self.retry_failed = retry_failed     # False: a query that failed before is not re-sent
        self.n_requests = 0          # network requests actually sent (cache hits excluded)
        self._last = 0.0
        self._lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, query: str) -> str:
        return os.path.join(self.cache_dir, hashlib.sha256(query.encode()).hexdigest() + ".json")

    def cached(self, query: str) -> Response | None:
        p = self._path(query)
        if os.path.exists(p):
            d = json.load(open(p))
            return Response(d["ok"], d.get("text", ""), d.get("error", ""), cached=True)
        return None

    def run(self, query: str, offline: bool = False) -> Response:
        hit = self.cached(query)
        if hit is not None:
            return hit
        fail = self._path(query)[:-5] + ".fail.json"
        if os.path.exists(fail) and not self.retry_failed:
            return Response(False, error="failed_before: " + json.load(open(fail)).get("error", ""), cached=True)
        if offline:
            return Response(False, error="not_cached")
        res = Response(False, error="no_attempt")
        for attempt in range(self.retries):
            with self._lock:
                gap = self.sleep - (time.time() - self._last)
                if gap > 0:
                    time.sleep(gap)
                self._last = time.time()
            endpoint = ENDPOINTS[attempt % len(ENDPOINTS)]
            try:
                self.n_requests += 1
                req = urllib.request.Request(endpoint, data=urllib.parse.urlencode({"data": query}).encode(),
                                             headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.http_timeout) as r:
                    body = r.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    res = Response(False, error="too_big")
                    break
                text = body.decode("utf-8", "replace")
                if "runtime error" in text or "<html" in text[:300].lower():
                    res = Response(False, error="server: " + re.sub(r"\s+", " ", text)[:160])
                else:
                    res = Response(True, text=text)
                    break
            except Exception as e:              # HTTP 504/429, socket timeout, reset
                res = Response(False, error=repr(e)[:160])
            if attempt + 1 < self.retries:
                time.sleep(self.backoff * (attempt + 1))
        if res.ok or res.error == "too_big":
            json.dump({"ok": res.ok, "text": res.text, "error": res.error}, open(self._path(query), "w"))
        else:   # remember the failure (gateway timeouts on large areas tend to repeat)
            json.dump({"error": res.error, "at": time.time()}, open(fail, "w"))
        return res


def csv_rows(text: str, ncols: int) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == ncols:
            rows.append(parts)
    return rows


def area_lookup_query(name: str) -> str:
    return ('[out:csv(::id,"name","admin_level","boundary","place";false;"\t")][timeout:60];'
            'area["name"="%s"];out;' % name.replace('"', '\\"'))


def superset_query(area_id: str, tag_raws: list[str], keys: list[str]) -> str:
    """Union of all tag readings in one area, CSV with only the keys the filters need.
    Element type and dropped conditions are applied locally on the returned rows."""
    cols = ",".join(["::type", "::id"] + ['"%s"' % k.replace('"', '\\"') for k in keys])
    union = "".join("nwr%s(area.searchArea);" % t for t in tag_raws)
    return ('[out:csv(%s;false;"\t")][timeout:90];area(id:%s)->.searchArea;(%s);out;'
            % (cols, area_id, union))


def around_query(radius: str, lat: str, lon: str, main_raw: str, keys: list[str]) -> str:
    """All objects matching the main condition within `radius`; type and dropped conditions are
    applied locally. One request per candidate radius (exact around semantics, no geometry needed)."""
    cols = ",".join(["::type", "::id"] + ['"%s"' % k.replace('"', '\\"') for k in keys])
    return ('[out:csv(%s;false;"\t")][timeout:90];(nwr%s(around:%s,%s,%s););out;'
            % (cols, main_raw, radius, lat, lon))
