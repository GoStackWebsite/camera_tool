"""Offline test of zone editing (camera view): no cameras, no real Congruence Eye, no real credentials.

    .venv\\Scripts\\python.exe tools\\test_zone_edit.py

Part 1 edits a hand-made camera in a temporary config. Part 2 runs a throwaway Congruence Eye look-alike on a
random local port and edits a managed camera through it: the change must reach the server in editor (16:9
stage) coordinates and come back through the normal sync.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import zone_edit                                        # noqa: E402
from app.analytics import Analytics                              # noqa: E402
from app.config import CameraConfig, ConfigStore, Thresholds, ZoneConfig   # noqa: E402
from app.congruence import fetch_config                          # noqa: E402
from app.manager import Manager                                  # noqa: E402
from app.store import Store                                      # noqa: E402
from app.sync import Reconciler, frame_to_stage, orient_line, stage_to_frame  # noqa: E402

RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, what: str) -> None:
    RESULTS.append((bool(ok), what))
    print(("  ok   " if ok else "  FAIL ") + what)


def raises(status: int, fn, *a) -> bool:
    try:
        fn(*a)
    except zone_edit.ZoneEditError as e:
        return e.status == status
    return False


SQUARE = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]


def build(tmp: Path):
    cs = ConfigStore(tmp / "config.json")
    store = Store(tmp / "retail.db")
    mgr = Manager(SimpleNamespace(), cs, store, None, None)
    return cs, store, mgr


# ---------------------------------------------------------------- geometry and validation
print("geometry / validation")
for fw, fh in ((2560, 1440), (1280, 960), (2560, 1080), (0, 0)):
    pts = [[0.2, 0.3], [0.85, 0.6], [0.5, 0.95]]
    back = [stage_to_frame(frame_to_stage(p, fw, fh), fw, fh) for p in pts]
    check(all(abs(a[0] - b[0]) < 1e-3 and abs(a[1] - b[1]) < 1e-3 for a, b in zip(pts, back)), f"frame -> stage -> frame round trip at {fw}x{fh}")
check(frame_to_stage([0.0, 0.0], 1280, 960) == [0.125, 0.0], "a 4:3 frame's left edge sits at x = 0.125 on the 16:9 stage")
check(raises(422, zone_edit.clean_points, [[0.1, 0.1], [0.2, 0.2]]), "fewer than 3 corners is refused")
check(raises(422, zone_edit.clean_points, [[0.1, 0.1], [0.1001, 0.1], [0.1, 0.1001]]), "a zone with no area is refused")
check(zone_edit.clean_points([[-1, 0.2], [0.5, 2], [0.9, 0.1], [-1, 0.2]]) == [[0.0, 0.2], [0.5, 1.0], [0.9, 0.1]], "corners are clamped to the frame and the closing duplicate dropped")
check(raises(422, zone_edit.clean_name, "   "), "an empty name is refused")

# ---------------------------------------------------------------- hand-made camera
print("hand-made camera")
with tempfile.TemporaryDirectory() as d:
    cs, store, mgr = build(Path(d))
    cam = CameraConfig(id="cam1", name="Cam", url="rtsp://x", enabled=True,
                       zones=[ZoneConfig("Desk", SQUARE, "display"), ZoneConfig("Till", [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9]], "queue")])
    cs.config.cameras.append(cam)
    cs.config.congruence.zones = {"cam1/Desk": 22}
    an = Analytics(cam, Thresholds(), store, None)
    mgr.runtimes["cam1"] = SimpleNamespace(cfg=cam, analytics=an)
    person = SimpleNamespace(zones={"Desk": 100.0}, zone_seq=["Desk"], pending_side=0, pending_n=0, history=[], side=0, side_pt=None)
    an.tracks[7] = person
    an.zone_stats["Desk"] = {"visits": 4, "dwell_total": 40.0}
    store.add_zone_visit({"camera_id": "cam1", "zone": "Desk", "track_id": 1, "enter_ts": 1.0, "exit_ts": 9.0, "dwell": 8.0, "is_staff": False})
    moved = [[0.15, 0.1], [0.45, 0.1], [0.45, 0.5], [0.3, 0.6], [0.15, 0.4]]
    r = zone_edit.apply_local(cs, mgr, "cam1", [
        {"op": "update", "zone": "Desk", "name": "Front desk", "role": "checkout", "points": moved},
        {"op": "create", "name": "Shelf", "role": "display", "points": [[0.5, 0.1], [0.7, 0.1], [0.7, 0.3]]},
        {"op": "update", "zone": "Till", "name": "Shelf"},
        {"op": "delete", "zone": "Nope"},
    ])
    z = {z.name: z for z in cs.config.camera("cam1").zones}
    check(r["applied"] == 2 and len(r["errors"]) == 2, "two changes applied, the duplicate name and the unknown zone reported")
    check(set(z) == {"Front desk", "Till", "Shelf"}, "zone renamed, zone added, the others untouched")
    check(z["Front desk"].role == "checkout" and z["Front desk"].points == moved, "new role and the five-corner shape are saved")
    check(json.loads((Path(d) / "config.json").read_text())["cameras"][0]["zones"][0]["name"] == "Front desk", "config.json on disk has the change")
    check(person.zones == {"Front desk": 100.0} and person.zone_seq == ["Front desk"], "a person standing in the renamed zone keeps the visit")
    check(an.zone_stats.get("Front desk", {}).get("visits") == 4, "live counters follow the new name")
    rows = store.conn.execute("SELECT zone FROM zone_visits").fetchall()
    check([r[0] for r in rows] == ["Front desk"], "stored zone history follows the new name")
    check(cs.config.congruence.zones == {"cam1/Front desk": 22}, "a hand-made Congruence Eye mapping follows the new name")
    r = zone_edit.apply_local(cs, mgr, "cam1", [{"op": "delete", "zone": "Shelf"}])
    check(r["applied"] == 1 and "Shelf" not in {z.name for z in cs.config.camera("cam1").zones}, "zone removed")
    # count line: add, move, unchanged, refused, removed
    walker = SimpleNamespace(zones={}, zone_seq=[], pending_side=1, pending_n=1, history=[(1.0, 0.5, 0.3)], side=1, side_pt=(0.5, 0.9))
    an.tracks[8] = walker
    door = [[0.3, 0.5], [0.7, 0.52]]
    r = zone_edit.apply_local(cs, mgr, "cam1", [{"op": "line", "points": door}])
    check(r["applied"] == 1 and cs.config.camera("cam1").line == door and an.cam.line == door, "count line added and live in the analytics")
    check(walker.side == -1 and walker.pending_side == 0, "a person above the new line is on its OUT side straight away (no phantom crossing)")
    r = zone_edit.apply_local(cs, mgr, "cam1", [{"op": "line", "points": [[0.3, 0.5], [0.7, 0.52]]}])
    check(r["applied"] == 0 and not r["errors"], "saving the same line again changes nothing")
    r = zone_edit.apply_local(cs, mgr, "cam1", [{"op": "line", "points": [[0.3, 0.5], [0.305, 0.5]]}, {"op": "line"}])
    check(r["applied"] == 0 and len(r["errors"]) == 2 and r["errors"][0]["zone"] == "count line", "a too-short line and a line without ends are refused")
    r = zone_edit.apply_local(cs, mgr, "cam1", [{"op": "line", "points": [door[1], door[0]]}])
    check(r["applied"] == 1 and cs.config.camera("cam1").line == [door[1], door[0]], "Swap IN/OUT reverses the line")
    r = zone_edit.apply_local(cs, mgr, "cam1", [{"op": "line", "remove": True}])
    check(r["applied"] == 1 and cs.config.camera("cam1").line is None, "count line removed")
    store.close()

# ---------------------------------------------------------------- IN side -> Congruence Eye inDirection
print("count line direction")
for line, want in (([[0.1, 0.5], [0.9, 0.5]], "DOWN"), ([[0.9, 0.5], [0.1, 0.5]], "UP"), ([[0.5, 0.1], [0.5, 0.9]], "LEFT"),
                   ([[0.5, 0.9], [0.5, 0.1]], "RIGHT"), ([[0.665, 0.405], [0.785, 0.425]], "DOWN"), ([[0.2, 0.2], [0.6, 0.9]], "LEFT")):
    got = zone_edit.in_direction(line)
    back = orient_line(line, got)
    check(got == want and back == line, f"{line} -> IN {want}, and the server's rule gives the same line back")

# ---------------------------------------------------------------- managed camera against a look-alike server
print("managed camera (look-alike Congruence Eye)")
ZONES = {
    10: {"id": 10, "storeId": 5, "cameraId": 3, "name": "Counter", "type": "SECTION", "polygon": [[0.2, 0.2], [0.5, 0.2], [0.5, 0.6], [0.2, 0.6]], "isActive": True},
    11: {"id": 11, "storeId": 5, "cameraId": 3, "name": "Queue", "type": "QUEUE", "polygon": [[0.6, 0.5], [0.9, 0.5], [0.9, 0.9], [0.6, 0.9]], "isActive": True},
    12: {"id": 12, "storeId": 5, "cameraId": 3, "name": "Door", "type": "ENTRY_LINE", "polygon": [[0.1, 0.9], [0.9, 0.9]], "inDirection": "UP", "isActive": True},
}
CALLS: list[tuple[str, str, dict]] = []
TOKENS = {"tok-admin": {"name": "Asha", "email": "asha@example.test", "role": "ADMIN"},
          "tok-mgr": {"name": "Ravi", "email": "ravi@example.test", "role": "STORE_MANAGER"}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _user(self):
        return TOKENS.get((self.headers.get("Authorization") or "").replace("Bearer ", ""))

    def do_GET(self):
        if self.path == "/api/ingest/config":
            if self.headers.get("X-Api-Key") != "edge-key":
                return self._send(401, {"message": "bad key"})
            return self._send(200, {"store": {"id": 5, "name": "Test shop"}, "cameras": [{"id": 3, "name": "Floor", "isActive": True}],
                                    "zones": [z for z in ZONES.values() if z["isActive"]], "aiManagerRules": []})
        if self.path == "/api/auth/me":
            u = self._user()
            return self._send(200, u) if u else self._send(401, {"message": "Unauthorized"})
        self._send(404, {"message": "not found"})

    def _write(self, method: str):
        body = self._body()
        if self.path == "/api/auth/login":
            tok = {"right": "tok-admin", "manager": "tok-mgr"}.get(body.get("password"))
            return self._send(200, {"accessToken": tok, "user": TOKENS[tok]}) if tok else self._send(401, {"message": "Invalid email or password"})
        u = self._user()
        if u is None:
            return self._send(401, {"message": "Unauthorized"})
        if u["role"] != "ADMIN":
            return self._send(403, {"message": "Forbidden resource"})
        CALLS.append((method, self.path, body))
        if method == "POST" and self.path == "/api/zones":
            if any(p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1 for p in body["polygon"]):
                return self._send(400, {"message": "polygon points must be normalised"})
            zid = max(ZONES) + 1
            ZONES[zid] = {"id": zid, "isActive": True, **body}
            return self._send(201, ZONES[zid])
        zid = int(self.path.rsplit("/", 1)[1])
        if zid not in ZONES:
            return self._send(404, {"message": "Zone not found"})
        if method == "DELETE":
            ZONES[zid]["isActive"] = False
            return self._send(200, {"ok": True})
        ZONES[zid].update(body)
        self._send(200, ZONES[zid])

    def do_POST(self):
        self._write("POST")

    def do_PATCH(self):
        self._write("PATCH")

    def do_DELETE(self):
        self._write("DELETE")


srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}"

with tempfile.TemporaryDirectory() as d:
    cs, store, mgr = build(Path(d))
    cc = cs.config.congruence
    cc.enabled, cc.managed, cc.base_url, cc.api_key = True, True, BASE, "edge-key"
    connector = SimpleNamespace(remote=None, remote_cached=None)
    connector.accept_config = lambda data, dispatch=True: setattr(connector, "remote", data)
    rec = Reconciler(cs, mgr, store, connector)
    data = fetch_config(BASE, "edge-key")
    connector.accept_config(data)
    rec.run(data)
    cam = cs.config.camera("ce3")
    check(cam is not None and [z.name for z in cam.zones] == ["Counter", "Queue"] and cam.line is not None, "first sync mirrors two zones and the count line")
    # a 4:3 camera: the server's 16:9 editor coordinates and the frame coordinates differ
    mgr.update_camera(__import__("dataclasses").replace(cam, frame_w=1280, frame_h=960))
    rec.rederive("ce3")
    cam = cs.config.camera("ce3")
    store.add_zone_visit({"camera_id": "ce3", "zone": "Counter", "track_id": 1, "enter_ts": 1.0, "exit_ts": 9.0, "dwell": 8.0, "is_staff": False})

    check(zone_edit.ce_login(BASE, "a@example.test", "right")["user"]["can_edit"] is True, "an admin sign-in may edit")
    check(zone_edit.ce_login(BASE, "m@example.test", "manager")["user"]["can_edit"] is False, "a store-manager sign-in is told it cannot edit")
    check(raises(401, zone_edit.ce_login, BASE, "a@example.test", "wrong"), "a wrong password is a 401")
    check(zone_edit.ce_me(BASE, "tok-admin")["user"]["name"] == "Asha" and raises(401, zone_edit.ce_me, BASE, "stale"), "session check: valid token named, stale token 401")

    upd = [{"op": "update", "zone": "Counter", "name": "Display counter", "role": "display", "points": [[0.0, 0.25], [1.0, 0.25], [1.0, 0.75], [0.0, 0.75]]}]
    check(raises(401, zone_edit.apply_managed, cs, connector, rec, "ce3", upd, ""), "no sign-in: refused with 401, nothing written")
    check(raises(403, zone_edit.apply_managed, cs, connector, rec, "ce3", upd, "tok-mgr") and not CALLS, "non-admin: refused with 403, nothing written")

    r = zone_edit.apply_managed(cs, connector, rec, "ce3", upd + [
        {"op": "create", "name": "Trial room", "role": "staff", "points": [[0.5, 0.05], [0.9, 0.05], [0.9, 0.2]]},
        {"op": "update", "zone": "Queue", "name": "Queue", "role": "queue", "points": [list(p) for p in cam.zones[1].points]},
        {"op": "create", "name": "Aisle", "role": "checkout", "points": SQUARE},
        {"op": "delete", "zone": "Queue"},
    ], "tok-admin")
    check(r["applied"] == 3 and r["synced"] and len(r["errors"]) == 1 and "checkout" in r["errors"][0]["error"], "3 written, unchanged zone skipped, unsupported role reported")
    patch = next(c for c in CALLS if c[0] == "PATCH")
    check(patch[1] == "/api/zones/10" and set(patch[2]) == {"name", "polygon"}, "PATCH carries only what changed (name and shape, not the role)")
    check(patch[2]["polygon"] == [[0.125, 0.25], [0.875, 0.25], [0.875, 0.75], [0.125, 0.75]], "the shape reaches the server in 16:9 editor coordinates")
    post = next(c for c in CALLS if c[0] == "POST")
    check(post[2]["storeId"] == 5 and post[2]["cameraId"] == 3 and post[2]["type"] == "STAFF_ONLY", "new zone created on the right store / camera as STAFF_ONLY")
    check(ZONES[11]["isActive"] is False, "removing a zone marks it inactive on the server (it can be restored there)")
    cam = cs.config.camera("ce3")
    z = {z.name: z for z in cam.zones}
    check(set(z) == {"Display counter", "Trial room"}, "after the sync back the camera shows the server's zones")
    check(zone_edit.same_points(z["Display counter"].points, [[0.0, 0.25], [1.0, 0.25], [1.0, 0.75], [0.0, 0.75]]), "the edited shape comes back exactly where it was dragged")
    check(z["Trial room"].role == "staff" and z["Trial room"].remote_id == 13, "the new zone is linked to its server id")
    check(cs.config.congruence.zones == {"ce3/Display counter": 10, "ce3/Trial room": 13}, "the event mapping was regenerated")
    rows = store.conn.execute("SELECT zone FROM zone_visits").fetchall()
    check([r[0] for r in rows] == ["Display counter"], "history follows a rename made through the server")
    n = len(CALLS)
    r = zone_edit.apply_managed(cs, connector, rec, "ce3", [{"op": "update", "zone": "Trial room", "name": "Trial room"}], "tok-admin")
    check(r["applied"] == 0 and len(CALLS) == n, "saving without changes writes nothing")

    # count line on the server (camera is 4:3, so editor and frame coordinates differ)
    cam = cs.config.camera("ce3")
    r = zone_edit.apply_managed(cs, connector, rec, "ce3", [{"op": "line", "points": [list(p) for p in cam.line]}], "tok-admin")
    check(r["applied"] == 0 and len(CALLS) == n, "an untouched count line is not re-sent")
    door = [[0.2, 0.4], [0.8, 0.45]]                        # IN = down, the opposite of the server's UP line
    check(raises(401, zone_edit.apply_managed, cs, connector, rec, "ce3", [{"op": "line", "points": door}], ""), "moving the line needs a sign-in")
    r = zone_edit.apply_managed(cs, connector, rec, "ce3", [{"op": "line", "points": door}], "tok-admin")
    patch = CALLS[-1]
    check(r["applied"] == 1 and patch[0] == "PATCH" and patch[1] == "/api/zones/12" and set(patch[2]) == {"polygon", "inDirection"},
          "the line is PATCHed on its ENTRY_LINE zone (shape and IN side only)")
    check(patch[2]["inDirection"] == "DOWN" and patch[2]["polygon"] == [frame_to_stage(p, 1280, 960) for p in door],
          "it reaches the server in 16:9 editor coordinates with IN = DOWN")
    cam = cs.config.camera("ce3")
    check(zone_edit.same_points(cam.line, door) and cam.remote_line["in_direction"] == "DOWN", "after the sync back the camera counts on the line as dragged")
    r = zone_edit.apply_managed(cs, connector, rec, "ce3", [{"op": "line", "points": [door[1], door[0]]}], "tok-admin")
    cam = cs.config.camera("ce3")
    check(r["applied"] == 1 and CALLS[-1][2]["inDirection"] == "UP" and zone_edit.same_points(cam.line, [door[1], door[0]]), "Swap IN/OUT turns the server's inDirection round")
    r = zone_edit.apply_managed(cs, connector, rec, "ce3", [{"op": "line", "remove": True}], "tok-admin")
    cam = cs.config.camera("ce3")
    check(r["applied"] == 1 and ZONES[12]["isActive"] is False and cam.line is None, "removing the line marks its zone inactive, the camera has no line")
    r = zone_edit.apply_managed(cs, connector, rec, "ce3", [{"op": "line", "points": door}], "tok-admin")
    post = CALLS[-1]
    cam = cs.config.camera("ce3")
    check(r["applied"] == 1 and post[0] == "POST" and post[2]["type"] == "ENTRY_LINE" and post[2]["cameraId"] == 3 and post[2]["inDirection"] == "DOWN"
          and not post[2]["name"].lower().startswith("pass"), "a camera without a line gets a new ENTRY_LINE (not a pass-by line)")
    check(zone_edit.same_points(cam.line, door) and cam.remote_line["zone_id"] == max(ZONES), "the new line is live and linked to its server zone")
    store.close()

srv.shutdown()
failed = [w for ok, w in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
sys.exit(1 if failed else 0)
