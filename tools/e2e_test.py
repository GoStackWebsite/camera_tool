"""End-to-end test of the Congruence Eye pipeline with real data from this engine's history.

What it does (idempotent, re-runnable):
  1. logs in to Congruence Eye as an admin, creates/reuses an isolated test store "E2E-TEST" with one ENTRANCE
     camera, an ENTRY_LINE / SECTION / STAFF_ONLY / QUEUE zone and an edge device (API key cached in
     data/e2e_state.json);
  2. checks the edge contract (GET /api/ingest/config with the key);
  3. replays a day of REAL visits recorded by this engine (data/retail.db, --source-day, default yesterday)
     as ENTRY / EXIT / ZONE events with the recorded gender, age band and staff flag, shifted into the last
     two hours so the server accepts them; plus queue snapshots, a heatmap, heartbeats, store-ops events,
     two of the day's real captured snapshots (PERIODIC + AI_MANAGER) and two AI detections with evidence;
  4. validates what the server stored through its analytics API (traffic, demographics, zones, queue,
     AI manager, live, store-ops, snapshot) against what was sent;
  5. writes logs/e2e_report.json (with the URLs and expected numbers for the UI check) and prints a table.

  .venv\\Scripts\\python tools\\e2e_test.py [--source-day 2026-10-09] [--visits 60] [--deactivate]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.congruence import ce_age_band, iso  # noqa: E402  (the engine's own mapping of ages / timestamps)

STATE = ROOT / "data" / "e2e_state.json"
REPORT = ROOT / "logs" / "e2e_report.json"
SNAP_DIR = Path(r"D:/New folder/congruence-eye-smarterasp/App_Data/uploads/snapshots")
STORE_CODE = "E2E-TEST"


class CE:
    def __init__(self, base: str, email: str, password: str):
        self.base = base.rstrip("/")
        self.s = requests.Session()
        r = self.s.post(f"{self.base}/api/auth/login", json={"email": email, "password": password}, timeout=20)
        r.raise_for_status()
        j = r.json()
        self.token = j.get("accessToken") or j.get("token") or j.get("access_token")
        self.user = j.get("user") or {}
        if not self.token:
            raise SystemExit("login gave no token")

    def call(self, method: str, path: str, body=None, **kw):
        r = self.s.request(method, f"{self.base}{path}", json=body,
                           headers={"Authorization": f"Bearer {self.token}"}, timeout=30, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code} {r.text[:300]}")
        return r.json() if r.text else None

    def edge(self, method: str, path: str, key: str, body=None, batch: str | None = None, **kw):
        headers = {"X-Api-Key": key}
        if batch:
            headers["X-Batch-Id"] = batch
        return self.s.request(method, f"{self.base}{path}", json=body, headers=headers, timeout=60, **kw)


def ensure(items, pred, create):
    for it in items:
        if pred(it):
            return it, False
    return create(), True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", default="http://localhost:3000")
    ap.add_argument("--email", default="admin@congruence.in")
    ap.add_argument("--password", default=os.environ.get("CE_PASSWORD", "Password@123"))
    ap.add_argument("--source-day", default=(dt.date.today() - dt.timedelta(days=1)).isoformat())
    ap.add_argument("--source-camera", default="ce31")
    ap.add_argument("--visits", type=int, default=60)
    ap.add_argument("--deactivate", action="store_true", help="deactivate the test store at the end")
    a = ap.parse_args()

    results: list[dict] = []

    def check(name: str, ok: bool | None, expected=None, actual=None, note: str = ""):
        status = "PASS" if ok else ("CHECK" if ok is None else "FAIL")
        results.append({"check": name, "status": status, "expected": expected, "actual": actual, "note": note})
        print(f"  [{status:5}] {name}: expected={expected} actual={actual} {note}")

    print("== 1. servers")
    ce = CE(a.ce, a.email, a.password)
    health = requests.get(f"{a.ce}/api/health", timeout=10).json()
    check("CE health", health.get("status") == "ok", "ok", health.get("status"))
    check("CE login role", bool(ce.user), "a user", ce.user.get("role"))
    try:
        eng = requests.get("http://localhost:8000/api/health", timeout=10).json()
        check("engine reachable", True, "cameras list", f"{len(eng.get('cameras', []))} cameras, {sum(1 for c in eng.get('cameras', []) if c.get('connected'))} connected")
    except Exception as e:  # engine is not needed for the server test
        check("engine reachable", None, "reachable", str(e)[:80])

    print("== 2. test store, camera, zones, edge device")
    regions = ce.call("GET", "/api/regions")
    regions = regions if isinstance(regions, list) else regions.get("items", [])
    region_id = regions[0]["id"]
    stores = ce.call("GET", "/api/stores")
    store, created = ensure(stores, lambda s: s.get("code") == STORE_CODE, lambda: ce.call("POST", "/api/stores", {
        "code": STORE_CODE, "name": "E2E Test Store (auto)", "regionId": region_id, "city": "Test", "timezone": "Asia/Kolkata",
        "openTime": "09:00", "closeTime": "21:00", "format": "HIGH_STREET", "isActive": True}))
    sid = store["id"]
    if not store.get("isActive", True):
        ce.call("PATCH", f"/api/stores/{sid}", {"isActive": True})
    print(f"  store #{sid} {'created' if created else 'reused'}")
    cams = ce.call("GET", f"/api/cameras?storeId={sid}")
    cam, _ = ensure(cams, lambda c: c["name"] == "E2E Entrance", lambda: ce.call("POST", "/api/cameras", {"storeId": sid, "name": "E2E Entrance", "type": "ENTRANCE"}))
    cid = cam["id"]
    zones = ce.call("GET", f"/api/zones?storeId={sid}")
    want = {
        "E2E door": {"type": "ENTRY_LINE", "polygon": [[0.1, 0.9], [0.9, 0.9]], "inDirection": "UP"},
        "E2E section": {"type": "SECTION", "polygon": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6], [0.1, 0.6]], "engagedDwellSec": 30},
        "E2E staff": {"type": "STAFF_ONLY", "polygon": [[0.6, 0.2], [0.9, 0.2], [0.9, 0.5], [0.6, 0.5]]},
        "E2E queue": {"type": "QUEUE", "polygon": [[0.6, 0.6], [0.9, 0.6], [0.9, 0.85], [0.6, 0.85]], "queueAlertLength": 3, "queueAlertWaitSec": 60},
    }
    zid: dict[str, int] = {}
    for name, spec in want.items():
        z, _ = ensure(zones, lambda z: z["cameraId"] == cid and z["name"] == name and z.get("isActive", True),
                      lambda: ce.call("POST", "/api/zones", {"storeId": sid, "cameraId": cid, "name": name, **spec}))
        zid[name] = z["id"]
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    devices = ce.call("GET", f"/api/edge-devices?storeId={sid}")
    key = state.get("api_key") if state.get("store_id") == sid else None
    if key and ce.edge("GET", "/api/ingest/config", key).status_code != 200:
        key = None
    if not key:
        dev, created = ensure(devices, lambda d: d["name"] == "E2E edge", lambda: ce.call("POST", "/api/edge-devices", {"storeId": sid, "name": "E2E edge"}))
        key = dev.get("apiKey")
        if not key:
            key = ce.call("POST", f"/api/edge-devices/{dev['id']}/rotate-key").get("apiKey")
        if not key:
            raise SystemExit("could not obtain an edge API key")
        STATE.write_text(json.dumps({"store_id": sid, "device_id": dev["id"], "api_key": key}))
    cfg = ce.edge("GET", "/api/ingest/config", key)
    check("edge contract GET /api/ingest/config", cfg.status_code == 200, 200, cfg.status_code)
    cfgj = cfg.json() if cfg.status_code == 200 else {}
    check("edge config lists the camera + 4 zones", {c["id"] for c in cfgj.get("cameras", [])} == {cid} and len([z for z in cfgj.get("zones", []) if z["cameraId"] == cid]) >= 4,
          f"camera {cid}, 4 zones", f"cameras {[c['id'] for c in cfgj.get('cameras', [])]}, zones {len(cfgj.get('zones', []))}")

    print(f"== 3. replay real visits of {a.source_day} ({a.source_camera}) into store #{sid}")
    db = sqlite3.connect(f"file:{ROOT / 'data' / 'retail.db'}?mode=ro", uri=True)
    d0 = dt.datetime.fromisoformat(a.source_day).timestamp()
    rows = db.execute(
        "SELECT track_id, start_ts, end_ts, dwell, engaged, is_staff, zones, gender, age FROM visits "
        "WHERE camera_id=? AND start_ts>=? AND start_ts<? AND dwell>=5 ORDER BY start_ts LIMIT ?",
        (a.source_camera, d0, d0 + 86400, a.visits)).fetchall()
    if not rows:
        raise SystemExit(f"no visits for {a.source_day} on {a.source_camera} in data/retail.db")
    now = time.time()
    shift = (now - 240) - max(r[2] or r[1] for r in rows)       # last replayed event ends 4 minutes ago
    run = uuid.uuid4().hex[:8]
    events: list[dict] = []
    exp = {"entry": 0, "exit": 0, "staff_entry": 0, "male": 0, "female": 0, "unknown": 0, "section_visits": 0, "staff_zone": 0, "engaged": 0}
    for i, (tid, st, en, dwell, engaged, is_staff, zones_s, gender, age) in enumerate(rows):
        st += shift
        en = (en or st - shift + dwell) + shift
        vid = f"e2e-{run}-{i}"
        g = gender if gender in ("M", "F") else "U"
        base = {"cameraId": cid, "trackId": vid, "visitorId": vid, "isStaff": bool(is_staff), "gender": g, "ageBand": ce_age_band(age)}
        events.append({**base, "type": "ENTRY", "ts": iso(st)})
        events.append({**base, "type": "EXIT", "ts": iso(en), "dwellSec": round(en - st, 1)})
        if is_staff:
            exp["staff_entry"] += 1
            events.append({**base, "type": "ZONE_ENTER", "ts": iso(st + 1), "zoneId": zid["E2E staff"]})
            events.append({**base, "type": "ZONE_EXIT", "ts": iso(en - 1), "zoneId": zid["E2E staff"], "dwellSec": round(en - st - 2, 1)})
            exp["staff_zone"] += 1
        else:
            exp["entry"] += 1; exp["exit"] += 1
            exp[{"M": "male", "F": "female"}.get(g, "unknown")] += 1
            if dwell >= 20:                                 # the longer visits browse the section
                events.append({**base, "type": "ZONE_ENTER", "ts": iso(st + 2), "zoneId": zid["E2E section"]})
                events.append({**base, "type": "ZONE_EXIT", "ts": iso(en - 1), "zoneId": zid["E2E section"], "dwellSec": round(en - st - 3, 1)})
                exp["section_visits"] += 1
                if en - st - 3 >= 30:
                    exp["engaged"] += 1
    first_ts = min(float(dt.datetime.fromisoformat(e["ts"]).timestamp()) for e in events)
    sent = {}
    for k in range(0, len(events), 500):
        batch = events[k:k + 500]
        r = ce.edge("POST", "/api/ingest/events", key, {"events": batch}, batch=hashlib.sha1(f"{run}-events-{k}".encode()).hexdigest())
        sent.setdefault("events", []).append(r.status_code)
        if r.status_code != 200:
            print("   events batch failed:", r.status_code, r.text[:300])
    check("POST /api/ingest/events", all(s == 200 for s in sent["events"]), f"{len(events)} events accepted", sent["events"])
    q = [{"zoneId": zid["E2E queue"], "ts": iso(now - 200 + 30 * j), "queueLength": [2, 3, 4, 5, 3, 1][j], "avgWaitSec": [20, 35, 50, 70, 40, 10][j], "counterStaffed": j != 3} for j in range(6)]
    r = ce.edge("POST", "/api/ingest/queue", key, {"snapshots": q}, batch=f"{run}-queue")
    check("POST /api/ingest/queue", r.status_code == 200, 200, r.status_code, r.text[:120] if r.status_code != 200 else "")
    grid = [round(((x * 7 + y * 3) % 11) / 10, 2) for y in range(18) for x in range(32)]
    r = ce.edge("POST", "/api/ingest/heatmap", key, {"cameraId": cid, "ts": iso(now - 100), "width": 32, "height": 18, "grid": grid}, batch=f"{run}-heat")
    check("POST /api/ingest/heatmap", r.status_code == 200, 200, r.status_code, r.text[:120] if r.status_code != 200 else "")
    r = ce.edge("POST", "/api/ingest/heartbeat", key, {"version": "e2e-test/1.0", "cameras": [{"cameraId": cid, "status": "ONLINE", "fps": 15.0}]})
    check("POST /api/ingest/heartbeat", r.status_code == 200, 200, r.status_code, r.text[:120] if r.status_code != 200 else "")
    ops = [{"type": "STORE_OPENED", "ts": iso(first_ts - 60), "cameraId": cid, "details": "e2e: first person seen"},
           {"type": "SOP_BREACH", "ts": iso(now - 150), "cameraId": cid, "details": "e2e: visitor lingering for 12 min"}]
    r = ce.edge("POST", "/api/ingest/storeops", key, {"events": ops}, batch=f"{run}-ops")
    check("POST /api/ingest/storeops", r.status_code == 200, 200, r.status_code, r.text[:120] if r.status_code != 200 else "")
    # real captures of the source day: one periodic snapshot and one as AI-manager evidence
    captures = sorted(glob.glob(str(SNAP_DIR / "*" / a.source_day / "cam*.jpg")))
    captures = [c for c in captures if os.path.getsize(c) > 60_000] or captures
    snap_ids = {}
    for kind, path in (("PERIODIC", captures[-1] if captures else None), ("AI_MANAGER", captures[len(captures) // 2] if captures else None)):
        if not path:
            check(f"snapshot upload {kind}", None, "a capture file", "none found on disk"); continue
        with open(path, "rb") as f:
            r = ce.edge("POST", "/api/ingest/snapshot", key, data={"cameraId": str(cid), "kind": kind, "ts": iso(now - 90)},
                        files={"file": ("snapshot.jpg", f.read(), "image/jpeg")})
        ok = r.status_code == 200 and r.json().get("snapshotId")
        if ok:
            snap_ids[kind] = r.json()["snapshotId"]
        check(f"POST /api/ingest/snapshot ({kind}, {Path(path).name})", bool(ok), "snapshotId", r.json() if r.status_code == 200 else r.text[:120])
    dets = [{"type": "MOBILE_USAGE", "cameraId": cid, "ts": iso(now - 120), "status": "BREACHED", "details": "e2e: staff on phone 12s", "confidence": 0.9, **({"snapshotId": snap_ids["AI_MANAGER"]} if "AI_MANAGER" in snap_ids else {})},
            {"type": "UNATTENDED_CUSTOMER", "cameraId": cid, "zoneId": zid["E2E section"], "ts": iso(now - 110), "status": "BREACHED", "details": "e2e: customer waiting 45s", **({"snapshotId": snap_ids["AI_MANAGER"]} if "AI_MANAGER" in snap_ids else {})}]
    r = ce.edge("POST", "/api/ingest/detections", key, {"detections": dets}, batch=f"{run}-det")
    check("POST /api/ingest/detections", r.status_code == 200, 200, r.status_code, r.text[:120] if r.status_code != 200 else "")

    print("== 4. validate through the analytics API (what the server stored)")
    today = dt.date.today().isoformat()
    qs = f"storeId={sid}&from={today}&to={today}"
    traffic = ce.call("GET", f"/api/analytics/traffic?{qs}")
    t = traffic["totals"]
    check("traffic.footfallIn == non-staff ENTRY sent", t["footfallIn"] == exp["entry"], exp["entry"], t["footfallIn"])
    check("traffic.footfallOut == non-staff EXIT sent", t["footfallOut"] == exp["exit"], exp["exit"], t["footfallOut"])
    check("traffic.staffExcluded == staff ENTRY sent", t["staffExcluded"] == exp["staff_entry"], exp["staff_entry"], t["staffExcluded"])
    check("traffic.uniqueVisitors == distinct non-staff visitors", t["uniqueVisitors"] == exp["entry"], exp["entry"], t["uniqueVisitors"])
    check("traffic.engaged (section dwell >= 30 s)", t["engaged"] == exp["engaged"], exp["engaged"], t["engaged"], "server rule may differ")
    ov = ce.call("GET", f"/api/analytics/overview?{qs}")
    check("overview.kpis.footfall == traffic.footfallIn", ov["kpis"]["footfall"]["value"] == t["footfallIn"], t["footfallIn"], ov["kpis"]["footfall"]["value"])
    demo = ce.call("GET", f"/api/analytics/demographics?{qs}")
    g = demo["gender"]
    check("demographics male/female == sent (non-staff)", g["male"] == exp["male"] and g["female"] == exp["female"], f"M {exp['male']} F {exp['female']} U {exp['unknown']}", f"M {g['male']} F {g['female']} U {g['unknown']}")
    zs = ce.call("GET", f"/api/analytics/zones?{qs}")
    zs = zs if isinstance(zs, list) else zs.get("items", [])
    sec = next((z for z in zs if z["id"] == zid["E2E section"]), None)
    check("zones: section visits == ZONE_ENTER sent", bool(sec) and sec["visits"] == exp["section_visits"], exp["section_visits"], sec and sec["visits"])
    check("zones: section gender split present", bool(sec) and (sec.get("male", 0) + sec.get("female", 0)) > 0, "> 0", sec and (sec.get("male"), sec.get("female")))
    qa = ce.call("GET", f"/api/analytics/queue?{qs}")
    qtxt = json.dumps(qa)[:160]
    check("queue analytics has the queue zone", str(zid["E2E queue"]) in json.dumps(qa) or "E2E queue" in json.dumps(qa), "zone present", qtxt)
    ai = ce.call("GET", f"/api/ai-manager/summary?{qs}")
    check("ai-manager.summary.totals.detections == 2", ai["totals"]["detections"] == 2, 2, ai["totals"]["detections"])
    dl = ce.call("GET", f"/api/ai-manager/detections?{qs}")
    items = dl.get("items", dl) if isinstance(dl, dict) else dl
    with_snap = [d for d in items if d.get("snapshotUrl")]
    check("ai-manager detections carry evidence snapshot", len(with_snap) == 2, 2, len(with_snap))
    live = ce.call("GET", f"/api/analytics/live?storeId={sid}")
    lc = next((c for c in live.get("cameras", []) if c["id"] == cid), {})
    check("live: camera ONLINE after heartbeat", lc.get("status") == "ONLINE", "ONLINE", lc.get("status"))
    check("live: todayFootfall == footfallIn", live.get("todayFootfall") == t["footfallIn"], t["footfallIn"], live.get("todayFootfall"))
    ops_a = ce.call("GET", f"/api/analytics/storeops?{qs}")
    oc = next((o for o in ops_a.get("openClose", []) if o["storeId"] == sid), {})
    check("storeops: actualOpen recorded", bool(oc.get("actualOpen")), "HH:MM", oc.get("actualOpen"))
    check("storeops: SOP breach counted", ops_a.get("totals", {}).get("sopBreaches", 0) >= 1, ">= 1", ops_a.get("totals", {}).get("sopBreaches"))
    hm = ce.call("GET", f"/api/analytics/heatmap?{qs}&cameraId={cid}")
    check("heatmap endpoint returns data for the camera", bool(hm) and ("grid" in json.dumps(hm) or "cells" in json.dumps(hm) or len(json.dumps(hm)) > 50), "grid", json.dumps(hm)[:100])
    snap = ce.s.get(f"{a.ce}/api/cameras/{cid}/snapshot", headers={"Authorization": f"Bearer {ce.token}"}, timeout=30)
    check("GET /api/cameras/:id/snapshot serves the uploaded capture", snap.status_code == 200 and (snap.headers.get("content-type", "").startswith("image") or "url" in snap.text), "image", f"{snap.status_code} {snap.headers.get('content-type', '')[:30]}")
    infra = ce.call("GET", f"/api/infra/health?storeIds={sid}")
    st0 = (infra.get("stores") or [{}])[0]
    check("infra health: edge ONLINE, 1/1 camera online", st0.get("edgeStatus") == "ONLINE" and st0.get("camerasOnline") == 1, "ONLINE 1/1", f"{st0.get('edgeStatus')} {st0.get('camerasOnline')}/{st0.get('camerasTotal')}")

    if a.deactivate:
        ce.call("PATCH", f"/api/stores/{sid}", {"isActive": False})
        print(f"  store #{sid} deactivated")

    fails = [r for r in results if r["status"] == "FAIL"]
    report = {"ran_at": dt.datetime.now().isoformat(timespec="seconds"), "store_id": sid, "camera_id": cid, "zones": zid, "run": run,
              "source": {"day": a.source_day, "camera": a.source_camera, "visits": len(rows), "events": len(events)}, "expected": exp,
              "ui_check": {"store": f"E2E Test Store (auto) (#{sid})", "date": today, "pages": ["/dashboard", "/analytics/traffic", "/analytics/demographics", "/analytics/zones", "/analytics/queue", "/ai-manager", "/storeops", "/live"],
                           "expect": {"footfall": t["footfallIn"], "male": g["male"], "female": g["female"], "staffExcluded": t["staffExcluded"], "sectionVisits": sec and sec["visits"], "detections": ai["totals"]["detections"]}},
              "results": results, "summary": {"pass": sum(r["status"] == "PASS" for r in results), "fail": len(fails), "check": sum(r["status"] == "CHECK" for r in results)}}
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"== done: {report['summary']}  report: {REPORT}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
