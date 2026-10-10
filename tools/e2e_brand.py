"""End-to-end check of brand attribution and phone demo detection on a recorded shop video.

Drives the first (or --camera) camera with datasets/e2e_shop_<day>.mp4, temporarily lowers serve_seconds so the
compressed recording can attribute customers, then restores the camera URL and the threshold.
  .venv\\Scripts\\python tools\\e2e_brand.py [--video ...] [--camera ce31] [--seconds 150]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "http://localhost:8000"
results: list[dict] = []


def check(name, ok, expected=None, actual=None, note=""):
    status = "PASS" if ok else ("CHECK" if ok is None else "FAIL")
    results.append({"check": name, "status": status, "expected": expected, "actual": actual, "note": note})
    print(f"  [{status:5}] {name}: expected={expected} actual={actual} {note}")


def api(method, path, body=None):
    r = requests.request(method, ENGINE + path, json=body, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code} {r.text[:200]}")
    return r.json() if r.text else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "datasets" / "e2e_shop_2026-10-09.mp4"))
    ap.add_argument("--camera", default=None)
    ap.add_argument("--seconds", type=int, default=150)
    a = ap.parse_args()
    cfg = api("GET", "/api/config")
    cam = next(c for c in cfg["cameras"] if c["id"] == (a.camera or cfg["cameras"][0]["id"]))
    cam_id = cam["id"]
    brands = [b["name"] for b in cfg["staff"].get("brands", [])]
    check("brands configured", bool(brands), "GoFix, GoGizmo", brands)
    thr = api("GET", "/api/config")["thresholds"]
    old_serve = thr["serve_seconds"]
    api("PUT", "/api/config/thresholds", {**thr, "serve_seconds": 3})
    raw = json.loads((ROOT / "data" / "config.json").read_text(encoding="utf-8"))
    rid = cam.get("remote_id")
    orig = (raw.get("congruence", {}).get("url_overrides") or {}).get(str(rid)) if rid else cam["url"]
    api("PUT", f"/api/cameras/{cam_id}", {"name": cam["name"], "url": str(Path(a.video).resolve())})
    t0 = time.time()
    seen_brand: dict[str, int] = {}
    served_max = 0
    served_today: dict = {}
    phone_demo = phone_personal = 0
    staff_brand_tracks = 0
    try:
        while time.time() - t0 < a.seconds:
            time.sleep(4)
            st = api("GET", "/api/stats")["cameras"].get(cam_id, {})
            for t in st.get("tracks", []):
                if t.get("staff") and t.get("brand"):
                    seen_brand[t["brand"]] = seen_brand.get(t["brand"], 0) + 1
                if not t.get("staff") and t.get("served"):
                    served_max = max(served_max, max(t["served"].values()))
            staff_brand_tracks = max(staff_brand_tracks, sum(1 for t in st.get("tracks", []) if t.get("staff") and t.get("brand")))
            served_today = st.get("served_today") or served_today
            evs = st.get("events", [])
            phone_demo = max(phone_demo, sum(1 for e in evs if e.get("type") == "phone_demo"))
            phone_personal = max(phone_personal, sum(1 for e in evs if e.get("type") == "phone"))
    finally:
        api("PUT", f"/api/cameras/{cam_id}", {"name": cam["name"], "url": orig or ""})
        api("PUT", "/api/config/thresholds", {**thr, "serve_seconds": old_serve})
        print(f"  camera {cam_id} restored, serve_seconds back to {old_serve}")
    check("staff tracks get a brand from the logo colour", bool(seen_brand), "GoFix and/or GoGizmo", seen_brand)
    check("customers accumulate attended seconds per brand", served_max > 0, "> 0 s", round(served_max, 1))
    check("walk-ins attributed to a brand (served_today)", bool(served_today), "{brand: n}", served_today)
    check("phone episodes classified", None, "demo vs personal events", {"phone_demo": phone_demo, "phone_personal": phone_personal},
          "depends on phones visible in the recording")
    ev = api("GET", "/api/events?hours=1")
    ev = ev if isinstance(ev, list) else ev.get("events", ev)
    served_ev = [e for e in ev if e.get("type") == "served"]
    check("'served' events recorded", len(served_ev) >= 1, ">= 1", len(served_ev), (served_ev[0]["message"] if served_ev else ""))
    fails = [r for r in results if r["status"] == "FAIL"]
    (ROOT / "logs" / "e2e_brand_report.json").write_text(json.dumps({"results": results}, indent=1, default=str), encoding="utf-8")
    print(f"== done: pass {sum(r['status'] == 'PASS' for r in results)} fail {len(fails)} check {sum(r['status'] == 'CHECK' for r in results)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
