"""Congruence Eye edge connector.

Pushes this app's analytics to a Congruence Eye server as the store's edge device (``X-Api-Key``),
the same contract its own ai-service uses (docs/API_CONTRACT*.md in the Congruence Eye repo):

  GET  /api/ingest/config      store, cameras, zones, AI Manager rules (drives the mapping UI)
  POST /api/ingest/events      ENTRY / EXIT / ZONE_ENTER / ZONE_EXIT per visitor
  POST /api/ingest/queue       queue length and wait per QUEUE zone
  POST /api/ingest/heatmap     32 x 18 foot-point grid per camera and window
  POST /api/ingest/heartbeat   camera ONLINE / OFFLINE + fps
  POST /api/ingest/storeops    STORE_OPENED / STORE_CLOSED / COUNTER_UNATTENDED / SOP_BREACH
  POST /api/ingest/detections  AI Manager breaches (MOBILE_USAGE, UNATTENDED_CUSTOMER, QUEUE_BREACH, ...)
  POST /api/ingest/snapshot    periodic JPEG (faces already blurred by the pipeline)

Everything except heartbeats and snapshots goes through a SQLite write-ahead spool and is deleted only
after the server accepted it, so an outage of the server or the network loses nothing. Batches carry an
``X-Batch-Id`` derived from the spooled row ids, so a retry after a lost response is a no-op server side.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

log = logging.getLogger("congruence")

VERSION = "cam-analytics/1.1"
HEAT_W, HEAT_H = 32, 18            # grid Congruence Eye expects
MAX_BATCH = 2000
DEAD_STATUS = {400, 413, 422}      # the server will never accept these: park them, never retry
PERMANENT_403 = "another store"    # wrong mapping / key: same treatment
TIMEOUT = 15

# kind -> (path, wrapper key or None for one-row payloads)
KINDS = {
    "events": ("/api/ingest/events", "events"),
    "queue": ("/api/ingest/queue", "snapshots"),
    "storeops": ("/api/ingest/storeops", "events"),
    "detections": ("/api/ingest/detections", "detections"),
    "heatmap": ("/api/ingest/heatmap", None),
}


def iso(t: float) -> str:
    """Epoch seconds -> ISO-8601 UTC with milliseconds and a Z suffix (the server insists on an offset)."""
    return datetime.fromtimestamp(round(t, 3), tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def ce_age_band(age) -> str:
    if age is None:
        return "U"
    a = int(age)
    if a < 18:
        return "0-17"
    if a < 25:
        return "18-24"
    if a < 35:
        return "25-34"
    if a < 45:
        return "35-44"
    if a < 55:
        return "45-54"
    return "55+"


def fetch_config(base_url: str, api_key: str) -> dict:
    """GET /api/ingest/config; raises on any failure with a readable message."""
    r = requests.get(base_url.rstrip("/") + "/api/ingest/config",
                     headers={"X-Api-Key": api_key, "User-Agent": VERSION}, timeout=TIMEOUT)
    if r.status_code == 401:
        raise RuntimeError("the server rejected the API key (401)")
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def summarise_config(data: dict) -> dict:
    st = data.get("store") or {}
    return {
        "store": {"id": st.get("id"), "name": st.get("name"), "code": st.get("code"), "timezone": st.get("timezone")},
        "cameras": [{"id": c["id"], "name": c.get("name"), "type": c.get("type"), "status": c.get("status")}
                    for c in data.get("cameras", [])],
        "zones": [{"id": z["id"], "camera_id": z.get("cameraId"), "name": z.get("name"), "type": z.get("type")}
                  for z in data.get("zones", []) if z.get("isActive", True)],
        "rules": [r.get("type") for r in data.get("aiManagerRules", [])],
    }


class Spool:
    """Write-ahead queue in SQLite: rows are removed only after the server accepted them."""

    def __init__(self, path: Path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS spool (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS spool_kind ON spool(kind, id)")
        self.db.execute("CREATE TABLE IF NOT EXISTS dead (id INTEGER PRIMARY KEY, kind TEXT, payload TEXT, created REAL, status INTEGER, error TEXT)")
        self.db.commit()

    def put(self, kind: str, payload: dict) -> None:
        with self.lock:
            self.db.execute("INSERT INTO spool(kind, payload, created) VALUES (?, ?, ?)", (kind, json.dumps(payload), time.time()))
            self.db.commit()

    def take(self, kind: str, limit: int) -> list[tuple[int, dict]]:
        with self.lock:
            rows = self.db.execute("SELECT id, payload FROM spool WHERE kind = ? ORDER BY id LIMIT ?", (kind, limit)).fetchall()
        return [(r[0], json.loads(r[1])) for r in rows]

    def take_ids(self, ids: list[int]) -> list[tuple[int, dict]]:
        with self.lock:
            rows = self.db.execute(f"SELECT id, payload FROM spool WHERE id IN ({','.join('?' * len(ids))}) ORDER BY id", ids).fetchall()
        return [(r[0], json.loads(r[1])) for r in rows]

    def delete(self, ids: list[int]) -> None:
        with self.lock:
            self.db.executemany("DELETE FROM spool WHERE id = ?", [(i,) for i in ids])
            self.db.commit()

    def dead_letter(self, ids: list[int], status: int, error: str) -> None:
        with self.lock:
            q = ",".join("?" * len(ids))
            rows = self.db.execute(f"SELECT kind, payload, created FROM spool WHERE id IN ({q})", ids).fetchall()
            self.db.executemany("INSERT INTO dead(kind, payload, created, status, error) VALUES (?, ?, ?, ?, ?)",
                                [(k, p, c, status, error[:500]) for k, p, c in rows])
            self.db.executemany("DELETE FROM spool WHERE id = ?", [(i,) for i in ids])
            self.db.commit()

    def counts(self) -> tuple[int, int]:
        with self.lock:
            backlog = self.db.execute("SELECT COUNT(*) FROM spool").fetchone()[0]
            dead = self.db.execute("SELECT COUNT(*) FROM dead").fetchone()[0]
        return backlog, dead

    def trim(self, max_rows: int) -> None:
        with self.lock:
            n = self.db.execute("SELECT COUNT(*) FROM spool").fetchone()[0]
            if n > max_rows:
                self.db.execute("DELETE FROM spool WHERE id IN (SELECT id FROM spool ORDER BY id LIMIT ?)", (n - max_rows,))
                self.db.commit()


class CongruenceConnector:
    """Background sender. Analytics objects call ``visitor_event``; the alert manager calls ``alert``;
    the manager calls ``store_event``. Everything else is collected on timers from the running cameras."""

    def __init__(self, config_store, manager, data_dir: Path, spool_max_rows: int = 200_000):
        self.config_store = config_store
        self.manager = manager
        self.spool = Spool(Path(data_dir) / "congruence.db")
        self.spool_max_rows = spool_max_rows
        self.session_id = int(time.time())          # makes track ids unique across restarts
        self._http = requests.Session()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="congruence", daemon=True)
        self._next: dict[str, float] = {}
        self._pending: dict[str, list[int]] = {}    # kind -> ids of the batch whose response was lost
        self._backoff = 0.0
        self._counts_cache: tuple[float, int, int] = (0.0, 0, 0)
        self.remote: dict | None = None             # last /api/ingest/config
        self.remote_fetched_at = 0.0
        self._rules: dict[str, int] = {}            # AI Manager rule type -> rule id
        self.reconciler = None                      # managed mode: applies the server's cameras / zones (set by server)
        self._cache_path = Path(data_dir) / "congruence_remote.json"
        self.remote_cached: dict | None = None      # last config fetched by a previous run (status / offline info only)
        self.remote_cached_at = 0.0
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            self.remote_cached = raw.get("data") if isinstance(raw.get("data"), dict) else None
            self.remote_cached_at = float(raw.get("fetched_at") or 0.0)
        except Exception:
            pass
        self.online: bool | None = None
        self.last_error = ""
        self.last_ok = 0.0
        self.sent = {k: 0 for k in KINDS} | {"snapshot": 0, "heartbeat": 0}

    # ---- lifecycle --------------------------------------------------------
    @property
    def cfg(self):
        return self.config_store.config.congruence

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def apply(self) -> None:
        """Settings changed: forget the remote config and run the timers again right away."""
        self.remote = None
        self._next.clear()
        self._backoff = 0.0
        self.online = None
        self.last_error = ""
        self._wake.set()

    def configured(self) -> bool:
        c = self.cfg
        return bool(c.enabled and c.base_url and c.api_key)

    def request_refresh(self) -> None:
        """Fetch /api/ingest/config on the next tick (Sync now, settings saved)."""
        self._next["config"] = 0.0
        self._wake.set()

    def accept_config(self, data: dict, dispatch: bool = True) -> None:
        """Store a freshly fetched server config, cache it on disk and (managed mode) hand it to the reconciler."""
        self.remote = data
        self.remote_fetched_at = time.time()
        self._rules = {r["type"]: r["id"] for r in data.get("aiManagerRules", []) if r.get("id") and r.get("type")}
        try:
            self._cache_path.write_text(json.dumps({"fetched_at": self.remote_fetched_at, "data": data}), encoding="utf-8")
            self.remote_cached, self.remote_cached_at = data, self.remote_fetched_at
        except OSError:
            log.warning("could not cache the Congruence Eye config at %s", self._cache_path)
        if dispatch and self.cfg.managed and self.reconciler is not None:
            self.reconciler.request(data)

    # ---- mapping ----------------------------------------------------------
    def cam_id(self, local_cam: str) -> int | None:
        v = self.cfg.cameras.get(local_cam)
        return int(v) if v else None

    def zone_id(self, local_cam: str, zone: str | None) -> int | None:
        if not zone:
            return None
        v = self.cfg.zones.get(f"{local_cam}/{zone}")
        return int(v) if v else None

    def _first_cam(self) -> int | None:
        for v in self.cfg.cameras.values():
            if v:
                return int(v)
        return None

    # ---- hooks called by analytics / alerts / manager --------------------------
    def visitor_event(self, local_cam: str, etype: str, t, now: float, zone: str | None = None,
                      dwell: float | None = None) -> None:
        if not self.configured():
            return
        cid = self.cam_id(local_cam)
        if cid is None:
            return
        zid = None
        if zone is not None:
            zid = self.zone_id(local_cam, zone)
            if zid is None:             # unmapped zone: nothing to send
                return
        tid = f"{local_cam}-{self.session_id}-{t.id}"
        ev = {"cameraId": cid, "type": etype, "ts": iso(now), "trackId": tid, "visitorId": tid,
              "isStaff": bool(t.is_staff), "gender": t.gender if t.gender in ("M", "F") else "U",
              "ageBand": ce_age_band(t.age)}
        if zid is not None:
            ev["zoneId"] = zid
        if t.group_id:
            ev["groupId"] = str(t.group_id)[:64]
        if dwell is not None:
            ev["dwellSec"] = round(max(0.0, min(float(dwell), 7 * 86400)), 1)
        self.spool.put("events", ev)
        self._wake_soon()

    def alert(self, atype: str, message: str, camera_id: str | None, zone: str | None, started: float,
              track_id: int | None = None) -> None:
        """Map this app's SOP alerts to Congruence Eye store-ops events and AI Manager detections."""
        if not self.configured():
            return
        cid = self.cam_id(camera_id) if camera_id else self._first_cam()
        zid = self.zone_id(camera_id, zone) if camera_id else None
        ts = iso(started)
        details = message[:1000]

        def detection(rule: str) -> None:
            if cid is None:
                return
            d = {"type": rule, "cameraId": cid, "ts": ts, "status": "BREACHED", "details": details}
            if zid is not None:
                d["zoneId"] = zid
            if rule in self._rules:
                d["ruleId"] = self._rules[rule]
            self.spool.put("detections", d)

        def storeops(kind: str) -> None:
            ev = {"type": kind, "ts": ts, "details": details}
            if cid is not None:
                ev["cameraId"] = cid
            self.spool.put("storeops", ev)

        if atype == "unattended":
            storeops("COUNTER_UNATTENDED")
            detection("UNATTENDED_CUSTOMER")
        elif atype == "phone_usage":
            detection("MOBILE_USAGE")
        elif atype == "queue_breach":
            detection("QUEUE_BREACH")
        elif atype in ("crowd", "loiter"):
            storeops("SOP_BREACH")
        elif atype in ("store_open_late", "store_not_closed"):
            detection("STORE_OPEN_CLOSE")
        else:                           # camera_offline is covered by the heartbeat
            return
        self._wake_soon()

    def store_event(self, etype: str, now: float, details: str | None = None) -> None:
        """STORE_OPENED / STORE_CLOSED from the manager's store-day logic."""
        if not self.configured():
            return
        ev = {"type": etype, "ts": iso(now)}
        cid = self._first_cam()
        if cid is not None:
            ev["cameraId"] = cid
        if details:
            ev["details"] = details[:1000]
        self.spool.put("storeops", ev)
        self._wake_soon()

    def _wake_soon(self) -> None:
        pass    # the 1 s loop picks spooled rows up on the next flush; nothing to do

    # ---- status -----------------------------------------------------------
    def status(self) -> dict:
        now = time.time()
        ts, backlog, dead = self._counts_cache
        if now - ts > 2.0:
            backlog, dead = self.spool.counts()
            self._counts_cache = (now, backlog, dead)
        c = self.cfg
        st = (self.remote or {}).get("store") or {}
        return {
            "enabled": c.enabled, "configured": self.configured(), "base_url": c.base_url,
            "online": self.online, "last_error": self.last_error, "last_ok": self.last_ok,
            "backlog": backlog, "dead": dead, "sent": dict(self.sent),
            "store": st.get("name"), "mapped_cameras": sum(1 for v in c.cameras.values() if v),
            "mapped_zones": sum(1 for v in c.zones.values() if v),
            "managed": bool(c.managed), "remote_fetched_at": self.remote_fetched_at, "remote_cached_at": self.remote_cached_at,
            "sync": self.reconciler.status() if self.reconciler is not None else None,
        }

    # ---- background loop --------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("Congruence Eye tick failed")
            self._wake.wait(1.0)
            self._wake.clear()

    def _due(self, name: str, now: float, every: float) -> bool:
        if now >= self._next.get(name, 0.0):
            self._next[name] = now + max(1.0, float(every))
            return True
        return False

    def _runtimes(self) -> list:
        with self.manager.lock:
            return list(self.manager.runtimes.values())

    def _tick(self) -> None:
        if not self.configured():
            self.online = None
            return
        now = time.time()
        runtimes = self._runtimes()
        for r in runtimes:                      # every analytics instance reports its events to us
            if getattr(r.analytics, "emitter", None) is not self:
                r.analytics.emitter = self
        c = self.cfg
        if self.remote is None or self._due("config", now, 300):
            self._refresh_config()
        if self._due("queue", now, c.queue_seconds):
            self._collect_queues(runtimes, now)
        if self._due("heatmap", now, c.heatmap_seconds):
            self._collect_heatmaps(runtimes, now)
        if self._due("heartbeat", now, c.heartbeat_seconds):
            self._send_heartbeat(runtimes)
        if self._due("snapshot", now, c.snapshot_seconds):
            self._send_snapshots(runtimes, now)
        if self._due("flush", now, c.flush_seconds):
            self._flush()
        if self._due("trim", now, 600):
            self.spool.trim(self.spool_max_rows)

    # ---- collectors -------------------------------------------------------
    def _collect_queues(self, runtimes: list, now: float) -> None:
        for r in runtimes:
            cam = r.analytics.cam
            stats = r.pipeline.last_stats or {}
            zstats = {z.get("name"): z for z in stats.get("zones", [])}
            cam_staff = int(stats.get("staff", 0) or 0)
            for z in cam.zones:
                if z.role != "queue":
                    continue
                zid = self.zone_id(cam.id, z.name)
                if zid is None:
                    continue
                zs = zstats.get(z.name, {})
                self.spool.put("queue", {
                    "zoneId": zid, "ts": iso(now), "queueLength": int(zs.get("count", 0) or 0),
                    "avgWaitSec": round(float(zs.get("avg_wait") or 0.0), 1),
                    "counterStaffed": bool(zs.get("staff", 0) or cam_staff),
                })

    def _collect_heatmaps(self, runtimes: list, now: float) -> None:
        for r in runtimes:
            cid = self.cam_id(r.analytics.camera_id)
            grid = getattr(r.analytics, "ce_heat", None)
            if cid is None or grid is None:
                continue
            with r.analytics._lock:
                snap = grid.copy()
                grid[:] = 0
            if snap.any():
                self.spool.put("heatmap", {"cameraId": cid, "ts": iso(now), "width": HEAT_W, "height": HEAT_H,
                                           "grid": [int(v) for v in snap.ravel()]})

    def _send_heartbeat(self, runtimes: list) -> None:
        cams = []
        for r in runtimes:
            cid = self.cam_id(r.analytics.camera_id)
            if cid is None:
                continue
            c = r.camera
            row = {"cameraId": cid, "status": "ONLINE" if c.connected else "OFFLINE", "fps": round(max(0.0, float(c.fps or 0.0)), 1)}
            if not c.connected:
                row["reason"] = (c.last_error or "no frames")[:300]
            cams.append(row)
        if not cams:
            return
        r = self._post("/api/ingest/heartbeat", {"version": VERSION, "cameras": cams}, batch_id=None)
        if r is not None and r.ok:
            self.sent["heartbeat"] += 1

    def _send_snapshots(self, runtimes: list, now: float) -> None:
        if self.online is False:
            return
        for r in runtimes:
            cid = self.cam_id(r.analytics.camera_id)
            if cid is None or not r.camera.connected:
                continue
            # faces are already blurred when the camera says so; 16:9 letterbox so the server's zone editor
            # (a 16:9 stage) shows the whole frame and its coordinates map back onto the frame exactly
            jpeg = r.pipeline.jpeg("raw", False, 1280, aspect=16 / 9)
            if not jpeg:
                continue
            try:
                resp = self._http.post(self.cfg.base_url.rstrip("/") + "/api/ingest/snapshot",
                                       headers=self._headers(uuid.uuid4().hex),
                                       data={"cameraId": str(cid), "kind": "PERIODIC", "ts": iso(now)},
                                       files={"file": ("snapshot.jpg", jpeg, "image/jpeg")}, timeout=30)
            except requests.RequestException as e:
                self._fail(f"snapshot: {e}")
                return
            self._note(resp, "/api/ingest/snapshot")
            if resp.ok:
                self.sent["snapshot"] += 1

    # ---- sending ----------------------------------------------------------
    def _refresh_config(self) -> None:
        try:
            resp = self._http.get(self.cfg.base_url.rstrip("/") + "/api/ingest/config", headers=self._headers(None), timeout=TIMEOUT)
        except requests.RequestException as e:
            self._fail(f"config: {e}")
            return
        self._note(resp, "/api/ingest/config")
        if resp.ok:
            try:
                data = resp.json()
            except ValueError:
                self.last_error = "config: server did not return JSON"
                return
            if not isinstance(data, dict):
                self.last_error = "config: unexpected response"
                return
            self.accept_config(data)
            log.info("Congruence Eye: connected to store '%s'", (data.get("store") or {}).get("name"))

    def _flush(self) -> None:
        for kind, (path, key) in KINDS.items():
            while not self._stop.is_set():
                pending = self._pending.pop(kind, None)
                rows = self.spool.take_ids(pending) if pending else self.spool.take(kind, 1 if key is None else MAX_BATCH)
                if not rows:
                    break
                ids = [i for i, _ in rows]
                if kind == "detections":            # rule ids become known once the config was fetched
                    for _, p in rows:
                        if "ruleId" not in p and p.get("type") in self._rules:
                            p["ruleId"] = self._rules[p["type"]]
                body = rows[0][1] if key is None else {key: [p for _, p in rows]}
                batch_id = hashlib.sha1(f"{kind}:{self.session_id}:{','.join(map(str, ids))}".encode()).hexdigest()
                resp = self._post(path, body, batch_id)
                if resp is None:                     # network: resend exactly this batch later
                    self._pending[kind] = ids
                    return
                if resp.ok:
                    self.spool.delete(ids)
                    self.sent[kind] += len(ids)
                    continue
                if resp.status_code in DEAD_STATUS or (resp.status_code == 403 and PERMANENT_403 in resp.text):
                    log.warning("Congruence Eye rejected %d %s row(s): HTTP %s %s", len(ids), kind, resp.status_code, resp.text[:200])
                    self.spool.dead_letter(ids, resp.status_code, resp.text)
                    continue
                self._pending[kind] = ids            # 401 / 409 / 5xx: retry the same batch later
                return

    def _headers(self, batch_id: str | None) -> dict:
        h = {"X-Api-Key": self.cfg.api_key, "User-Agent": VERSION}
        if batch_id:
            h["X-Batch-Id"] = batch_id
        return h

    def _post(self, path: str, body: dict, batch_id: str | None):
        try:
            resp = self._http.post(self.cfg.base_url.rstrip("/") + path, json=body, headers=self._headers(batch_id), timeout=TIMEOUT)
        except requests.RequestException as e:
            self._fail(f"{path}: {e}")
            return None
        self._note(resp, path)
        return resp

    def _note(self, resp, path: str) -> None:
        if resp.ok:
            self.online = True
            self.last_ok = time.time()
            self.last_error = ""
            self._backoff = 0.0
            return
        self.online = True                           # the server answered, it just did not like the request
        hint = " (check the API key)" if resp.status_code == 401 else ""
        self.last_error = f"{path}: HTTP {resp.status_code}{hint} {resp.text[:160]}".strip()
        if resp.status_code in (401, 403, 429) or resp.status_code >= 500:
            self._delay()

    def _fail(self, msg: str) -> None:
        self.online = False
        self.last_error = msg
        self._delay()

    def _delay(self) -> None:
        self._backoff = min(60.0, max(2.0, self._backoff * 2))
        later = time.time() + self._backoff
        for name in ("flush", "heartbeat", "snapshot", "config"):
            self._next[name] = max(self._next.get(name, 0.0), later)
