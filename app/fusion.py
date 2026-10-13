"""One store, several cameras: the people two cameras both see are one person.

A camera pair is described by *shared regions*: the same piece of floor, drawn once in each camera's view
(data/fusion.json, see docs/fusion.example.json). One camera is the store's overview (``primary``): its track
ids and its counts are the store's. A person standing in a shared region on a second camera is linked to the
person standing in that region on the overview camera, and from then on

* carries the same id on both cameras (label, events, Congruence Eye ``visitorId``),
* gets one verdict: the clothing votes of both views are pooled, or, where a region names the camera with the
  better view (``"trust"``: a close-up of the billing desk against an overview from above), that camera's
  verdict on the clothing is the store's; a staff zone or a manual tag on either camera counts for both,
* is counted once in the store totals; the second view adds its zones (a queue at the billing desk) to the
  person's journey instead of creating a visit of its own.

With several people in one region the pairs are chosen by clothing colour; a link is kept while both tracks
live and drops a couple of seconds after either side has left the region.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("fusion")

LINK_GRACE = 2.0          # seconds a link survives after one side has left the shared region
LINK_CONFIRM = 1.0        # a pairing must be the answer this long before it becomes a link (a foot point that strays
                          # into a region for a moment must not marry two different people)
VETO_COLOUR_COST = 0.6    # one person here on each camera, but their clothing is this different: not the same person
MAX_COLOUR_COST = 0.45    # several candidates: clothing colours further apart than this are not the same person
EDGE = 0.002              # foot points are pulled this far inside the frame before a region test


def _inside(x: float, y: float, pts: list) -> bool:
    x, y = min(max(x, EDGE), 1.0 - EDGE), min(max(y, EDGE), 1.0 - EDGE)
    c, n = False, len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def colour_cost(a, b) -> float | None:
    """0 = same clothing colour, 1 = opposite; None when one of the two has not been sampled yet.
    a, b: OpenCV HSV medians of the chest (h 0-179, s and v 0-255)."""
    if not a or not b:
        return None
    (h1, s1, v1), (h2, s2, v2) = a, b
    dv, ds = abs(v1 - v2) / 255.0, abs(s1 - s2) / 255.0
    grey1, grey2 = s1 < 50 or v1 < 60, s2 < 50 or v2 < 60       # black, grey or white: the hue means nothing
    if grey1 and grey2:
        return 0.8 * dv + 0.2 * ds
    if grey1 != grey2:                                           # a colour against black / grey / white
        return min(1.0, 0.5 + 0.5 * ds)
    dh = min(abs(h1 - h2), 180 - abs(h1 - h2)) / 90.0
    return 0.5 * dh + 0.25 * ds + 0.25 * dv


class StoreFusion:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._mtime = None
        self.primary = ""
        self.regions: list[dict] = []       # {"name": str, "views": {camera id: polygon}}
        self.links: dict[tuple, dict] = {}  # (camera id, track id) -> {"pid", "region", "since", "last"}
        self._pending: dict[tuple, tuple] = {}   # (camera id, track id) -> (overview track id, proposed since)
        self._lock = threading.Lock()
        self.error = ""
        self.reload()

    # ---- configuration ----------------------------------------------------
    def reload(self) -> None:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            mtime = None
        if mtime == self._mtime:
            return
        self._mtime = mtime
        primary, regions, error = "", [], ""
        if mtime is not None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                primary = str(raw.get("primary") or "")
                for r in raw.get("regions") or []:
                    views = {str(c): [[float(p[0]), float(p[1])] for p in pts] for c, pts in (r.get("views") or {}).items() if len(pts) >= 3}
                    if primary in views and len(views) >= 2:
                        trust = str(r.get("trust") or "")
                        regions.append({"name": str(r.get("name") or f"region {len(regions) + 1}"), "views": views,
                                        "trust": trust if trust in views else ""})
            except Exception as e:
                primary, regions, error = "", [], f"{self.path.name}: {e}"
                log.warning("store fusion configuration not usable: %s", error)
        with self._lock:
            self.primary, self.regions, self.error = primary, regions, error
            self.links.clear()
            self._pending.clear()
        if regions:
            log.info("store fusion: overview camera %s, shared regions: %s", primary, ", ".join(r["name"] for r in regions))

    @property
    def active(self) -> bool:
        return bool(self.primary and self.regions)

    def status(self) -> dict:
        with self._lock:
            return {"active": self.active, "primary": self.primary, "regions": [{"name": r["name"], "cameras": sorted(r["views"])} for r in self.regions],
                    "links": [{"camera": k[0], "track": k[1], "store_id": l["pid"], "region": l["region"]} for k, l in self.links.items()],
                    "error": self.error}

    # ---- matching ---------------------------------------------------------
    def _region_of(self, cam: str, foot) -> str | None:
        for r in self.regions:
            pts = r["views"].get(cam)
            if pts and _inside(foot[0], foot[1], pts):
                return r["name"]
        return None

    @staticmethod
    def _assign(D: list[dict], P: list[dict]) -> list[tuple[dict, dict]]:
        """Pair the people of the second camera (D) with those of the overview camera (P) in one region."""
        if not D or not P:
            return []
        if len(D) == 1 and len(P) == 1:               # one person here on each camera: the same person,
            c = colour_cost(D[0].get("torso"), P[0].get("torso"))
            return [] if c is not None and c > VETO_COLOUR_COST else [(D[0], P[0])]      # unless dressed quite differently
        cand = []
        for d in D:
            for p in P:
                c = colour_cost(d.get("torso"), p.get("torso"))
                if c is not None and c <= MAX_COLOUR_COST:      # several people: wait until colours are known
                    cand.append((c, d["id"], p["id"], d, p))
        out, used_d, used_p = [], set(), set()
        for c, did, pid, d, p in sorted(cand, key=lambda x: x[:3]):
            if did in used_d or pid in used_p:
                continue
            used_d.add(did)
            used_p.add(pid)
            out.append((d, p))
        return out

    def tick(self, analytics: dict, now: float) -> None:
        """Re-evaluate the links and hand every camera what the other view knows. `analytics`: camera id -> Analytics."""
        self.reload()
        if not self.active or self.primary not in analytics:
            for an in analytics.values():
                an.apply_fusion({}, now, "")
            return
        views = {cam: an.fusion_view(now) for cam, an in analytics.items()}      # each under its own lock, one at a time
        prim = {v["id"]: v for v in views[self.primary]}
        per_cam: dict[str, dict] = {cam: {} for cam in analytics}
        with self._lock:
            proposed: dict[tuple, int] = {}
            for cam, vs in views.items():
                if cam == self.primary:
                    continue
                det = {v["id"]: v for v in vs}
                for key in [k for k in self.links if k[0] == cam]:           # links that no longer hold
                    l = self.links[key]
                    d, p = det.get(key[1]), prim.get(l["pid"])
                    if d is None or p is None:
                        del self.links[key]
                        continue
                    if self._region_of(cam, d["foot"]) == l["region"] and self._region_of(self.primary, p["foot"]) == l["region"]:
                        l["last"] = now
                    elif now - l["last"] > LINK_GRACE:
                        del self.links[key]
                taken = {l["pid"] for k, l in self.links.items() if k[0] == cam}
                for r in self.regions:                                        # new links, region by region
                    if cam not in r["views"]:
                        continue
                    D = [d for d in vs if (cam, d["id"]) not in self.links and self._region_of(cam, d["foot"]) == r["name"]]
                    P = [p for p in views[self.primary] if p["id"] not in taken and self._region_of(self.primary, p["foot"]) == r["name"]]
                    for d, p in self._assign(D, P):
                        key = (cam, d["id"])
                        proposed[key] = p["id"]
                        pid, since = self._pending.get(key, (None, now))
                        if pid != p["id"]:
                            self._pending[key] = (p["id"], now)          # a new answer: the clock starts again
                        elif now - since >= LINK_CONFIRM:
                            self.links[key] = {"pid": p["id"], "region": r["name"], "since": now, "last": now}
                            taken.add(p["id"])
            self._pending = {k: v for k, v in self._pending.items() if proposed.get(k) == v[0] and k not in self.links}
            for (cam, tid), l in self.links.items():
                d = next((v for v in views.get(cam, []) if v["id"] == tid), None)
                p = prim.get(l["pid"])
                if d is None or p is None:
                    continue
                # which of the two cameras has the better view of clothing in this region ("trust" in fusion.json)?
                trust = next((r["trust"] for r in self.regions if r["name"] == l["region"]), "")
                per_cam[cam][tid] = {"store_id": p["id"], "store_cam": self.primary, "duplicate": True, "peers": [p],
                                     "trust": "own" if trust == cam else "peer" if trust == self.primary else ""}
                e = per_cam[self.primary].setdefault(p["id"], {"store_id": None, "store_cam": "", "duplicate": False, "peers": [], "trust": ""})
                e["peers"].append(d)
                e["trust"] = "own" if trust == self.primary else "peer" if trust == cam else e["trust"]
        others = sorted(c for c in analytics if c != self.primary)
        for cam, an in analytics.items():                                     # outside this object's lock
            # people only a second camera sees keep that camera's own number, marked with a letter ("B-7"): the bare
            # numbers belong to the overview camera, so one number never means two different people
            an.apply_fusion(per_cam.get(cam, {}), now, "" if cam == self.primary else f"{chr(66 + others.index(cam) % 25)}-")
