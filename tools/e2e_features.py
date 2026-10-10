"""End-to-end test of the market-parity features on the running engine (http://localhost:8000):

  1. alerts: phone push (ntfy), WhatsApp Cloud API and webhook, against a local mock server;
  2. live POS: webhook bills in several shapes, CSV drop folder, conversion tiles, server contract;
  3. counting accuracy audit against a recorded window;
  4. pass-by line / capture rate: a "Pass-by" ENTRY_LINE zone in Congruence Eye reaches the engine;
  5. evidence clips + learned staff model: the first camera is pointed at a recorded video of the shop
     (datasets/e2e_shop_<day>.mp4) until a crowd alert produces an MP4 clip; the camera's own RTSP URL is restored.

  .venv\\Scripts\\python tools\\e2e_features.py [--video datasets/e2e_shop_2026-10-09.mp4] [--skip-video]
Writes logs/e2e_features_report.json."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "http://localhost:8000"
CE = "http://localhost:3000"
REPORT = ROOT / "logs" / "e2e_features_report.json"
results: list[dict] = []


def check(name, ok, expected=None, actual=None, note=""):
    status = "PASS" if ok else ("CHECK" if ok is None else "FAIL")
    results.append({"check": name, "status": status, "expected": expected, "actual": actual, "note": note})
    print(f"  [{status:5}] {name}: expected={expected} actual={actual} {note}")


def api(method, path, body=None, **kw):
    r = requests.request(method, ENGINE + path, json=body, timeout=60, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code} {r.text[:200]}")
    return r.json() if r.text else None


class Mock(BaseHTTPRequestHandler):
    hits: list[dict] = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        Mock.hits.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body.decode("utf-8", "replace")})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"messages":[{"id":"wamid.test"}]}')

    def log_message(self, *a):
        pass


def ce_login():
    l = requests.post(f"{CE}/api/auth/login", json={"email": "admin@congruence.in", "password": "Password@123"}, timeout=20).json()
    return l.get("accessToken") or l.get("token")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "datasets" / "e2e_shop_2026-10-09.mp4"))
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--camera", default=None, help="engine camera id to drive with the video (default: first camera)")
    a = ap.parse_args()
    cfg = api("GET", "/api/config")
    cams = cfg["cameras"]
    cam_id = a.camera or cams[0]["id"]

    print("== 1. alerts: push (ntfy), WhatsApp, webhook")
    srv = HTTPServer(("127.0.0.1", 8765), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old = dict(cfg["notifications"])
    test_cfg = {"ntfy_url": "http://127.0.0.1:8765", "ntfy_topic": "e2e-topic", "ntfy_token": "",
                "whatsapp_api_base": "http://127.0.0.1:8765", "whatsapp_token": "e2e-token", "whatsapp_phone_id": "1234567890", "whatsapp_to": "919876543210, 919876543211",
                "webhook_url": "http://127.0.0.1:8765/webhook", "types": old.get("types")}
    api("PUT", "/api/config/notifications", test_cfg)
    Mock.hits.clear()
    res = api("POST", "/api/notifications/test")
    check("notification test: push ok", str(res.get("push", "")).startswith("ok"), "ok (200)", res.get("push"))
    check("notification test: whatsapp ok for both recipients", "error" not in str(res.get("whatsapp", "error")) and res.get("whatsapp", "").count("ok") == 2, "2 x ok", res.get("whatsapp"))
    check("notification test: webhook ok", str(res.get("webhook", "")).startswith("ok"), "ok (200)", res.get("webhook"))
    paths = [h["path"] for h in Mock.hits]
    check("mock received ntfy topic POST", "/e2e-topic" in paths, "/e2e-topic", paths)
    wa = [h for h in Mock.hits if h["path"] == "/1234567890/messages"]
    body_ok = all(json.loads(h["body"]).get("messaging_product") == "whatsapp" and h["headers"].get("authorization") == "Bearer e2e-token" for h in wa)
    check("mock received WhatsApp Cloud API messages (bearer + payload)", len(wa) == 2 and body_ok, "2 messages", f"{len(wa)} messages, payload ok={body_ok}")
    ntfy = next((h for h in Mock.hits if h["path"] == "/e2e-topic"), {})
    check("ntfy carries Title header", bool(ntfy.get("headers", {}).get("title")), "Title", ntfy.get("headers", {}).get("title"))
    # restore (masked secrets are returned as *** by GET: only touch the fields the test changed)
    restore = {k: ("" if k in ("ntfy_topic", "whatsapp_phone_id", "whatsapp_to", "webhook_url", "whatsapp_token", "ntfy_token") else old.get(k, ""))
               for k in test_cfg if k != "types"}
    restore["ntfy_url"] = old.get("ntfy_url") or "https://ntfy.sh"
    restore["whatsapp_api_base"] = old.get("whatsapp_api_base") or "https://graph.facebook.com/v21.0"
    restore["webhook_url"] = old.get("webhook_url", "")
    api("PUT", "/api/config/notifications", restore)
    srv.shutdown()

    print("== 2. live POS")
    run = int(time.time())
    bills = [{"billNo": f"E2E-{run}-1", "ts": dt.datetime.now().isoformat(timespec="seconds"), "amount": 1499.0, "itemsCount": 2, "staffCode": "S01"},
             {"invoice_no": f"E2E-{run}-2", "time": time.time() - 300, "total": "2,250.50", "qty": 3, "cashier": "S02"},
             {"bill": f"E2E-{run}-3", "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "net_amount": 499, "items": 1}]
    before = api("GET", "/api/pos/status")["today"]["bills"]
    r = api("POST", "/api/pos/transactions?forward=false", {"transactions": bills[:1]})
    r2 = api("POST", "/api/pos/transactions?forward=false", bills[1:])          # plain list with aliases
    check("POS webhook accepts standard + aliased bills", r["accepted"] == 1 and r2["accepted"] == 2 and r2["rejected"] == 0, "1 + 2 accepted", f"{r['accepted']} + {r2['accepted']} (rejected {r2['rejected']})")
    bad = api("POST", "/api/pos/transactions?forward=false", [{"amount": 10}])
    check("POS webhook rejects a bill without a number", bad["rejected"] == 1 and bad["accepted"] == 0, "rejected 1", bad)
    st = api("GET", "/api/pos/status")
    check("POS status counts today's bills", st["today"]["bills"] >= before + 3, f">= {before + 3}", st["today"]["bills"])
    inbox = Path(st["inbox"])
    csv_path = inbox / f"e2e_{run}.csv"
    csv_path.write_text(f"Invoice,Date,Grand Total,Items,Salesman\nE2E-{run}-4,{dt.datetime.now().strftime('%d/%m/%Y %H:%M')},899,1,S03\nE2E-{run}-5,{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},\"1,200\",2,S01\n", encoding="utf-8")
    time.sleep(6)                                                            # the watcher ignores files younger than 5 s
    sc = api("POST", "/api/pos/scan")
    check("CSV drop folder import", sc["bills"] == 2 and not csv_path.exists(), "2 bills, file moved to done/", f"{sc['bills']} bills, moved={not csv_path.exists()}")
    tr = api("GET", "/api/traffic?hours=24")
    check("engine conversion uses live bills", tr["sales"]["bills"] >= 5 and tr.get("conversion") is not None, ">= 5 bills, conversion set", f"{tr['sales']['bills']} bills, conversion {tr.get('conversion')}")
    listed = [t for t in api("GET", "/api/pos/transactions?hours=1") if t["bill_no"].startswith(f"E2E-{run}")]
    check("bills listed with normalised fields", len(listed) == 5 and all(t["amount"] > 0 for t in listed), "5 bills", f"{len(listed)} bills, amounts {[t['amount'] for t in listed]}")
    # server contract the connector uses (POST /api/pos/transactions with the edge key), on the isolated test store
    state_f = ROOT / "data" / "e2e_state.json"
    if state_f.exists():
        stt = json.loads(state_f.read_text())
        sys.path.insert(0, str(ROOT))
        from app.pos import normalise, to_ce
        payload = {"transactions": [to_ce(normalise(b), stt["store_id"]) for b in bills]}
        tok = ce_login()
        H0 = {"Authorization": f"Bearer {tok}"}
        requests.patch(f"{CE}/api/stores/{stt['store_id']}", json={"isActive": True}, headers=H0, timeout=30)   # analytics scope = active stores
        rr = requests.post(f"{CE}/api/pos/transactions", json=payload, headers={"X-Api-Key": stt["api_key"]}, timeout=30)
        check("Congruence Eye accepts the forwarded bill format (edge key)", rr.status_code == 200, 200, f"{rr.status_code} {rr.text[:100]}")
        day = dt.date.today().isoformat()
        conv = requests.get(f"{CE}/api/analytics/conversion?storeId={stt['store_id']}&from={day}&to={day}", headers=H0, timeout=30).json()
        check("Congruence Eye conversion shows the bills", conv["totals"]["transactions"] >= 3 and conv["totals"]["revenue"] > 0, ">= 3 transactions", f"{conv['totals']['transactions']} transactions, revenue {conv['totals']['revenue']}")
        requests.patch(f"{CE}/api/stores/{stt['store_id']}", json={"isActive": False}, headers=H0, timeout=30)
    else:
        check("Congruence Eye POS contract", None, "run tools/e2e_test.py first", "no data/e2e_state.json")

    print("== 3. counting accuracy audit")
    day0 = dt.datetime(2026, 10, 9, 18, 0).timestamp()
    live = api("GET", f"/api/audits/live?camera_id={cam_id}&since={day0}")
    check("audit live counter answers", "in" in live and "out" in live, "in/out", {k: live[k] for k in ("in", "out", "pass")})
    eng = requests.get(f"{ENGINE}/api/audits/live", params={"camera_id": cam_id, "since": day0}, timeout=30).json()
    counts = api("GET", f"/api/audits/live?camera_id={cam_id}&since={day0}")
    window_in = counts["in"]
    r = requests.post(f"{ENGINE}/api/audits", json={"camera_id": cam_id, "start": day0, "end": time.time(), "manual_in": window_in, "manual_out": max(0, counts["out"] - 5), "note": "e2e: manual = engine (in), engine+5 (out)"}, timeout=30)
    check("audit stored", r.status_code == 201, 201, r.status_code)
    au = r.json() if r.status_code == 201 else {}
    check("accuracy IN 100% when manual equals engine", au.get("accuracy_in") == 100.0, 100.0, au.get("accuracy_in"))
    exp_out = round(max(0.0, 100 - 5 / max(counts["out"] - 5, 1) * 100), 1) if counts["out"] > 5 else None
    check("accuracy OUT computed from the delta", exp_out is None or au.get("accuracy_out") == exp_out, exp_out, au.get("accuracy_out"))
    check("audit listed", any(x["id"] == au.get("id") for x in api("GET", "/api/audits")), "present", "ok")
    r = requests.post(f"{ENGINE}/api/audits", json={"camera_id": cam_id, "start": day0, "end": day0 - 10, "manual_in": 1}, timeout=30)
    check("audit rejects end before start", r.status_code == 422, 422, r.status_code)

    print("== 4. pass-by line from Congruence Eye -> engine")
    tok = ce_login()
    H = {"Authorization": f"Bearer {tok}"}
    cam_cfg = next(c for c in cams if c["id"] == cam_id)
    rid, sid = cam_cfg.get("remote_id"), None
    zid = None
    if rid:
        stores = requests.get(f"{CE}/api/stores", headers=H, timeout=30).json()
        store_name = cfg["store"]["name"]
        sid = next((s["id"] for s in stores if s["name"] == store_name), None)
        if sid:
            for z in requests.get(f"{CE}/api/zones?storeId={sid}", headers=H, timeout=30).json():   # leftovers of an aborted run
                if z.get("name") == "Pass-by (e2e)" and z.get("isActive", True):
                    requests.delete(f"{CE}/api/zones/{z['id']}", headers=H, timeout=30)
            z = requests.post(f"{CE}/api/zones", json={"storeId": sid, "cameraId": rid, "name": "Pass-by (e2e)", "type": "ENTRY_LINE", "polygon": [[0.78, 0.72], [0.98, 0.64]], "inDirection": "LEFT"}, headers=H, timeout=30)
            zid = z.json().get("id") if z.status_code < 300 else None
            check("Pass-by ENTRY_LINE zone created on the server", bool(zid), "zone id", f"{z.status_code} {z.text[:80]}" if not zid else zid)
            api("POST", "/api/congruence/sync")
            c2 = next(c for c in api("GET", "/api/config")["cameras"] if c["id"] == cam_id)
            check("engine mapped it to passby_line (count line untouched)", bool(c2.get("passby_line")) and c2.get("line") == cam_cfg.get("line"), "passby_line set", f"passby_line={c2.get('passby_line')} line same={c2.get('line') == cam_cfg.get('line')}")
            st = api("GET", "/api/stats")["cameras"][cam_id]
            live_cam = next((c for c in api("GET", "/api/health")["cameras"] if c["camera_id"] == cam_id), {}).get("connected")
            check("stats expose has_passby / capture_rate_pct", (st.get("has_passby") is True and "capture_rate_pct" in st) if live_cam else None,
                  "has_passby true", st.get("has_passby"), "" if live_cam else "camera offline: live stats not refreshed")
    else:
        check("pass-by mapping", None, "managed camera", "camera is not managed by Congruence Eye")

    print("== 5. evidence clips + learned staff model on recorded video")
    if a.skip_video or not Path(a.video).exists():
        check("video replay", None, "datasets/e2e_shop_*.mp4", "skipped")
    else:
        raw = json.loads((ROOT / "data" / "config.json").read_text(encoding="utf-8"))
        orig_override = (raw.get("congruence", {}).get("url_overrides") or {}).get(str(rid)) if rid else next(c["url"] for c in raw["cameras"] if c["id"] == cam_id)
        api("PUT", f"/api/cameras/{cam_id}", {"name": cam_cfg["name"], "url": str(Path(a.video).resolve())})
        t0 = time.time()
        connected = tracks = 0
        model_hits = 0
        alert = None
        while time.time() - t0 < 150:
            time.sleep(5)
            st = api("GET", "/api/stats")["cameras"].get(cam_id, {})
            h = next((c for c in api("GET", "/api/health")["cameras"] if c["camera_id"] == cam_id), {})
            connected = connected or h.get("connected")
            tracks = max(tracks, len(st.get("tracks", [])))
            model_hits = max(model_hits, sum(1 for t in st.get("tracks", []) if (t.get("model_pct") or 0) >= 70))
            opens = api("GET", "/api/alerts?hours=1")
            alert = next((x for x in opens if x.get("camera_id") == cam_id and x.get("clip")), None)
            if alert:
                break
        check("video source connected and people tracked", bool(connected) and tracks >= 3, ">= 3 tracks", f"connected={bool(connected)} max tracks={tracks}")
        check("learned model marks staff on the recording (model_pct >= 70)", model_hits >= 1, ">= 1 track", model_hits)
        check("an alert with an evidence clip was raised", bool(alert), "alert with clip", alert and {"type": alert["type"], "clip": alert["clip"]})
        if alert:
            time.sleep(12)
            clips = api("GET", "/api/clips")
            mine = next((c for c in clips if c["file"] == alert["clip"]), None)
            check("clip file written (> 50 KB)", bool(mine) and mine["size"] > 50_000, "> 50 KB", mine and mine["size"])
            rr = requests.get(f"{ENGINE}/clips/{alert['clip']}", timeout=30)
            check("clip served as video/mp4", rr.status_code == 200 and "mp4" in rr.headers.get("content-type", ""), "200 video/mp4", f"{rr.status_code} {rr.headers.get('content-type')}")
        # restore the real camera URL
        api("PUT", f"/api/cameras/{cam_id}", {"name": cam_cfg["name"], "url": orig_override or ""})
        print(f"  camera {cam_id} restored to its own stream")
    if zid:
        requests.delete(f"{CE}/api/zones/{zid}", headers=H, timeout=30)
        api("POST", "/api/congruence/sync")
        c3 = next(c for c in api("GET", "/api/config")["cameras"] if c["id"] == cam_id)
        check("pass-by zone removed again, engine follows", not c3.get("passby_line"), "no passby_line", c3.get("passby_line"))

    fails = [r for r in results if r["status"] == "FAIL"]
    rep = {"ran_at": dt.datetime.now().isoformat(timespec="seconds"), "camera": cam_id, "results": results,
           "summary": {"pass": sum(r["status"] == "PASS" for r in results), "fail": len(fails), "check": sum(r["status"] == "CHECK" for r in results)}}
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(f"== done: {rep['summary']}  report: {REPORT}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
