"""FastAPI server: live MJPEG per camera, stats WebSocket, analytics / alerts / config API, dashboard."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import date, datetime

from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import socket
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import zone_edit
from .alerts import AlertManager
from .floor import FloorError
from .config import (STATIC_DIR, ZONE_ROLES, CameraConfig, ConfigStore, DemographicsConfig, NotifyConfig, ReportConfig,
                     StaffConfig, StoreConfig, Thresholds, ZoneConfig, _dc, build_rtsp_url, settings, uniform_kind,
                     unmask_url)
from .manager import Manager
from .notify import Notifier
from .reports import build_daily_report
from .store import Store, day_bounds

log = logging.getLogger("server")


from .config import CongruenceConfig, DATA_DIR, PosConfig
from .clips import ClipRecorder
from .congruence import CongruenceConnector, fetch_config, summarise_config
from .pos import PosService
from .sync import Reconciler


class Runtime:
    config_store: ConfigStore
    store: Store
    notifier: Notifier
    alerts: AlertManager
    manager: Manager
    connector: CongruenceConnector
    reconciler: Reconciler


rt = Runtime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    rt.config_store = ConfigStore(settings.config_path, settings.default_rtsp_url)
    rt.store = Store(settings.db_path)
    rt.notifier = Notifier(rt.config_store)
    rt.alerts = AlertManager(rt.store, rt.notifier)
    rt.manager = Manager(settings, rt.config_store, rt.store, rt.alerts, rt.notifier)
    rt.manager.report_builder = lambda d: build_daily_report(rt.store, rt.config_store.config, d)
    await run_in_threadpool(rt.manager.start)
    rt.connector = CongruenceConnector(rt.config_store, rt.manager, DATA_DIR)
    rt.manager.connector = rt.connector
    rt.alerts.listener = rt.connector.alert
    rt.reconciler = Reconciler(rt.config_store, rt.manager, rt.store, rt.connector)
    rt.connector.reconciler = rt.reconciler
    rt.manager.reconciler = rt.reconciler
    rt.clips = ClipRecorder(DATA_DIR / "clips", rt.manager, rt.store)      # evidence clips around alerts
    rt.alerts.clipper = rt.clips
    rt.alerts.base_url = f"http://{socket.gethostname()}:{settings.port}"
    rt.pos = PosService(rt.store, rt.connector, DATA_DIR / "pos_inbox")   # live POS bills -> conversion + server
    if rt.config_store.config.pos.inbox_enabled:
        rt.pos.start()
    rt.reconciler.start()
    rt.connector.start()
    log.info("Dashboard ready at http://localhost:%d (%d camera(s))", settings.port, len(rt.manager.runtimes))
    try:
        yield
    finally:
        rt.reconciler.stop()
        rt.pos.stop()
        rt.connector.stop()
        rt.manager.stop()
        rt.store.close()


app = FastAPI(title="Retail Camera Analytics", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
(DATA_DIR / "clips").mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=str(DATA_DIR / "clips")), name="clips")


# ---------------------------------------------------------------- helpers

def _runtime(cam_id: str):
    r = rt.manager.runtimes.get(cam_id)
    if r is None:
        raise HTTPException(404, f"camera '{cam_id}' is not running")
    return r


def _range(hours: float | None, day: str | None) -> tuple[float, float]:
    if day:
        try:
            d = datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(422, "day must be YYYY-MM-DD")
        return day_bounds(d.timestamp())
    now = time.time()
    if hours is None or hours <= 0:
        return day_bounds(now)[0], now          # today so far
    return now - hours * 3600, now


def _zone_roles() -> dict:
    return {(c.id, z.name): z.role for c in rt.config_store.config.cameras for z in c.zones}


def _managed() -> bool:
    return bool(rt.config_store.config.congruence.managed)


def _require_unmanaged() -> None:
    if _managed():
        raise HTTPException(409, "cameras, the count line and zones are managed by Congruence Eye: "
                                 "edit them there (Admin) and press Sync now")


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "cam"
    base, i = s, 2
    while rt.config_store.config.camera(s):
        s = f"{base}-{i}"
        i += 1
    return s


# ---------------------------------------------------------------- pages / video

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


def _part(data: bytes) -> bytes:
    return (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")


async def mjpeg(request: Request, cam_id: str, mode: str, heat: bool, width: int | None):
    last = -1
    while not await request.is_disconnected():
        r = rt.manager.runtimes.get(cam_id)
        if r is None:
            break
        seq = await run_in_threadpool(r.pipeline.wait_for_frame, last, 1.0)
        if seq == last:
            if last == -1 or not r.camera.connected:
                yield _part(r.pipeline.placeholder_jpeg())
            continue
        last = seq
        data = await run_in_threadpool(r.pipeline.jpeg, mode, heat, width)
        if data:
            yield _part(data)


@app.get("/video/{cam_id}")
async def video(request: Request, cam_id: str, mode: str = Query("annotated", pattern="^(annotated|raw)$"),
                heat: bool = False, width: int | None = Query(None, ge=160, le=3840)):
    _runtime(cam_id)
    return StreamingResponse(mjpeg(request, cam_id, mode, heat, width),
                             media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"})


@app.get("/snapshot/{cam_id}.jpg")
async def snapshot(cam_id: str, mode: str = Query("annotated", pattern="^(annotated|raw)$"), heat: bool = False,
                   width: int | None = Query(None, ge=160, le=3840)):
    r = _runtime(cam_id)
    data = (await run_in_threadpool(r.pipeline.jpeg, mode, heat, width)) if r.camera.connected else None
    data = data or r.pipeline.placeholder_jpeg()   # offline: status card instead of a stale frame
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- live

@app.get("/api/stats")
async def stats():
    return rt.manager.stats()


@app.websocket("/ws")
async def ws_stats(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(rt.manager.stats())
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.get("/api/health")
async def health(hours: float = Query(24, gt=0)):
    now = time.time()
    cams = rt.config_store.config.cameras
    rows = await run_in_threadpool(rt.store.health, now - hours * 3600, now, [c.id for c in cams])
    for row in rows:
        r = rt.manager.runtimes.get(row["camera_id"])
        cam = rt.config_store.config.camera(row["camera_id"])
        row["name"] = cam.name if cam else row["camera_id"]
        row["enabled"] = bool(cam and cam.enabled)
        row["connected"] = bool(r and r.camera.connected)
        row["error"] = r.camera.last_error if r else ""
        row["fps"] = round(r.camera.fps, 1) if r else 0
        row["inference_fps"] = round(r.pipeline.infer_fps, 1) if r else 0
    return {"cameras": rows, "model": settings.model, "notifier": {"sent": rt.notifier.sent, "last_error": rt.notifier.last_error}}


# ---------------------------------------------------------------- cameras

class CameraIn(BaseModel):
    id: str | None = Field(None, max_length=40, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1, max_length=60)
    url: str = Field("", max_length=500)
    user: str = Field("", max_length=100)
    password: str = Field("", max_length=200)
    enabled: bool = True
    target_fps: float = Field(0, ge=0, le=60)
    blur_faces: bool = True


class ZoneIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    role: str = Field("display", pattern="^(" + "|".join(ZONE_ROLES) + ")$")
    points: list[tuple[float, float]] = Field(min_length=3, max_length=64)


class LayoutIn(BaseModel):
    line: list[tuple[float, float]] | None = None
    zones: list[ZoneIn] = []


class TestCameraIn(BaseModel):
    url: str = Field(min_length=1, max_length=500)
    user: str = ""
    password: str = ""


def _camera_out(c: CameraConfig) -> dict:
    d = c.to_dict(mask=True)
    r = rt.manager.runtimes.get(c.id)
    d["running"] = r is not None
    d["connected"] = bool(r and r.camera.connected)
    d["resolution"] = f"{r.camera.width}x{r.camera.height}" if r and r.camera.width else ""
    return d


@app.get("/api/cameras")
async def list_cameras():
    return [_camera_out(c) for c in rt.config_store.config.cameras]


@app.post("/api/cameras", status_code=201)
async def add_camera(body: CameraIn):
    _require_unmanaged()
    if not body.url:
        raise HTTPException(422, "url is required")
    cam = CameraConfig(id=body.id or _slug(body.name), name=body.name,
                       url=build_rtsp_url(body.url, body.user, body.password), enabled=body.enabled,
                       target_fps=body.target_fps, blur_faces=body.blur_faces)
    try:
        await run_in_threadpool(rt.manager.add_camera, cam)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return _camera_out(cam)


@app.put("/api/cameras/{cam_id}")
async def update_camera(cam_id: str, body: CameraIn):
    old = rt.config_store.config.camera(cam_id)
    if old is None:
        raise HTTPException(404, "no such camera")
    if old.managed:
        # a camera from Congruence Eye: only its local RTSP URL override, fps and face blur can change here
        url_override = None
        new_url = unmask_url(body.url, old.url_override or old.url)   # masked URL with only the path changed: keep the password
        if new_url and "***" not in new_url:
            url_override = build_rtsp_url(new_url, body.user, body.password)
        try:
            cam = await run_in_threadpool(rt.reconciler.set_local, cam_id, url_override, body.target_fps, body.blur_faces)
        except KeyError:
            raise HTTPException(404, "no such camera")
        except ValueError as e:
            raise HTTPException(409, str(e))
        return _camera_out(cam)
    new_url = unmask_url(body.url, old.url)
    url = old.url if (not new_url or "***" in new_url) else build_rtsp_url(new_url, body.user, body.password)
    cam = replace(old, name=body.name, url=url, enabled=body.enabled, target_fps=body.target_fps, blur_faces=body.blur_faces)
    await run_in_threadpool(rt.manager.update_camera, cam)
    return _camera_out(cam)


@app.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: str):
    cam = rt.config_store.config.camera(cam_id)
    if cam is not None and cam.managed:
        raise HTTPException(409, "this camera comes from Congruence Eye: deactivate it there and press Sync now")
    try:
        await run_in_threadpool(rt.manager.remove_camera, cam_id)
    except KeyError:
        raise HTTPException(404, "no such camera")
    return {"ok": True}


@app.put("/api/cameras/{cam_id}/layout")
async def update_layout(cam_id: str, body: LayoutIn):
    old = rt.config_store.config.camera(cam_id)
    if old is None:
        raise HTTPException(404, "no such camera")
    if old.managed or _managed():
        raise HTTPException(409, "the count line and zones come from Congruence Eye: edit them there (Admin) and press Sync now")
    if body.line is not None and len(body.line) != 2:
        raise HTTPException(422, "line must have exactly 2 points")
    names = [z.name for z in body.zones]
    if len(names) != len(set(names)):
        raise HTTPException(422, "zone names must be unique")
    cam = replace(old, line=[[float(p[0]), float(p[1])] for p in body.line] if body.line else None,
                  zones=[ZoneConfig(z.name, [[float(p[0]), float(p[1])] for p in z.points], z.role) for z in body.zones])
    await run_in_threadpool(rt.manager.update_camera, cam)
    return _camera_out(cam)


class FloorMarkIn(BaseModel):
    img: tuple[float, float]           # picture point, fractions of the frame
    floor: tuple[float, float]         # where it is on the floor, in tiles (unit_m metres each)


class FloorIn(BaseModel):
    points: list[FloorMarkIn] = Field(max_length=40)
    unit_m: float | None = Field(None, gt=0, le=10)


@app.get("/api/floor")
async def floor_status():
    """Floor plan calibration of every camera (app/floor.py): marks, accuracy, and the floor -> picture mapping the
    page uses to draw the tile grid over the camera view."""
    return rt.manager.fusion.floor.status()


@app.put("/api/floor/{cam_id}")
async def floor_save(cam_id: str, body: FloorIn):
    if rt.config_store.config.camera(cam_id) is None:
        raise HTTPException(404, "no such camera")
    try:
        return await run_in_threadpool(rt.manager.fusion.floor.save_camera, cam_id,
                                       [{"img": list(p.img), "floor": list(p.floor)} for p in body.points], body.unit_m)
    except FloorError as e:
        raise HTTPException(422, str(e)) from None


@app.get("/api/floor/live")
async def floor_live():
    """Where everybody in view stands on the floor plan, per camera (metres), and the store id they are linked to."""
    f = rt.manager.fusion
    return {"positions": f.positions, "primary": f.primary, "active": f.active}


class ZoneChangeIn(BaseModel):
    op: str = Field(pattern="^(create|update|delete|line)$")   # line: the count line (2 points, + side of A->B = IN)
    zone: str | None = Field(None, max_length=80)        # current name (update / delete)
    name: str | None = Field(None, max_length=80)
    role: str | None = Field(None, pattern="^(" + "|".join(ZONE_ROLES) + ")$")
    points: list[tuple[float, float]] | None = Field(None, max_length=64)
    remove: bool = False                                 # line: take the count line away


class ZoneEditIn(BaseModel):
    changes: list[ZoneChangeIn] = Field(min_length=1, max_length=60)


class CELoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=1, max_length=200)


_login_fails: dict[str, list[float]] = {}     # client address -> times of recent failed sign-ins


def _login_guard(ip: str, failed: bool = False) -> None:
    """At most 5 failed Congruence Eye sign-ins per device in 5 minutes: this endpoint must not become a way
    to guess passwords of a server that is otherwise only reachable from this PC."""
    now = time.time()
    recent = [t for t in _login_fails.get(ip, []) if now - t < 300]
    if failed:
        recent.append(now)
    if recent:
        _login_fails[ip] = recent
    else:
        _login_fails.pop(ip, None)
    if len(_login_fails) > 500:
        _login_fails.clear()
    if not failed and len(recent) >= 5:
        raise HTTPException(429, "too many failed sign-ins from this device: wait 5 minutes and try again")


@app.post("/api/congruence/login")
async def congruence_login(request: Request, body: CELoginIn):
    """Sign in to Congruence Eye for zone editing. The token goes back to the page and is sent with each edit;
    this app does not keep the password or the token."""
    ip = request.client.host if request.client else "?"
    _login_guard(ip)
    try:
        return await run_in_threadpool(zone_edit.ce_login, rt.config_store.config.congruence.base_url,
                                       body.email.strip(), body.password)
    except zone_edit.ZoneEditError as e:
        if e.status == 401:
            _login_guard(ip, failed=True)
        raise HTTPException(e.status, e.message)


@app.get("/api/congruence/session")
async def congruence_session(x_ce_token: str | None = Header(None)):
    """Who the page's Congruence Eye token belongs to (401 once it has expired)."""
    try:
        return await run_in_threadpool(zone_edit.ce_me, rt.config_store.config.congruence.base_url, x_ce_token or "")
    except zone_edit.ZoneEditError as e:
        raise HTTPException(e.status, e.message)


@app.post("/api/cameras/{cam_id}/zones/edit")
async def edit_zones(cam_id: str, body: ZoneEditIn, x_ce_token: str | None = Header(None)):
    """Zone and count line edits from the camera view. A managed camera's zones and line are changed on
    Congruence Eye (admin token in X-CE-Token) and synced back; a hand-made camera's are changed here."""
    cam = rt.config_store.config.camera(cam_id)
    if cam is None:
        raise HTTPException(404, "no such camera")
    changes = [c.model_dump() for c in body.changes]
    try:
        if cam.managed:
            result = await run_in_threadpool(zone_edit.apply_managed, rt.config_store, rt.connector, rt.reconciler,
                                             cam_id, changes, x_ce_token or "")
        elif _managed():
            raise HTTPException(409, "this hand-made camera is switched off while Congruence Eye manages the cameras")
        else:
            result = await run_in_threadpool(zone_edit.apply_local, rt.config_store, rt.manager, cam_id, changes)
    except zone_edit.ZoneEditError as e:
        raise HTTPException(e.status, e.message)
    cam = rt.config_store.config.camera(cam_id)
    return {**result, "camera": _camera_out(cam) if cam else None}


@app.post("/api/cameras/test")
async def test_camera(body: TestCameraIn):
    return await run_in_threadpool(rt.manager.test_camera, build_rtsp_url(body.url, body.user, body.password))


# ---------------------------------------------------------------- config

@app.get("/api/config")
async def get_config():
    return rt.config_store.config.to_dict(mask=True)


def _merge(current, body: dict, cls):
    data = {**current.__dict__, **{k: v for k, v in body.items() if k in current.__dict__}}
    return _dc(cls, data)


@app.put("/api/config/store")
async def put_store(body: dict):
    cfg = rt.config_store.config
    for key in ("open_time", "close_time"):
        if key in body and not re.fullmatch(r"\d{2}:\d{2}", str(body[key])):
            raise HTTPException(422, f"{key} must be HH:MM")
    with rt.config_store.lock:
        cfg.store = _merge(cfg.store, body, StoreConfig)
        rt.config_store.save()
    return cfg.to_dict(mask=True)["store"]


@app.put("/api/config/thresholds")
async def put_thresholds(body: dict):
    thr = _merge(rt.config_store.config.thresholds, body, Thresholds)
    if thr.run_speed <= thr.walk_speed:
        raise HTTPException(422, "run_speed must be greater than walk_speed")
    for k in ("loiter_seconds", "engage_seconds", "zone_engage_seconds", "queue_seconds", "unattended_seconds",
              "phone_seconds", "staff_zone_seconds", "group_window"):
        if getattr(thr, k) <= 0:
            raise HTTPException(422, f"{k} must be positive")
    if thr.reentry_minutes < 0:
        raise HTTPException(422, "reentry_minutes must be 0 or more")
    if thr.line_margin < 0 or thr.line_cooldown_seconds < 0:
        raise HTTPException(422, "line_margin and line_cooldown_seconds must be 0 or more")
    if int(thr.line_min_frames) < 1:
        raise HTTPException(422, "line_min_frames must be at least 1")
    thr.line_min_frames = int(thr.line_min_frames)
    rt.manager.apply_thresholds(thr)
    return rt.config_store.config.to_dict()["thresholds"]


@app.put("/api/config/notifications")
async def put_notifications(body: dict):
    cfg = rt.config_store.config
    cur = cfg.notifications
    for secret in ("telegram_bot_token", "smtp_password"):
        if body.get(secret) == "***":
            body[secret] = getattr(cur, secret)
    with rt.config_store.lock:
        cfg.notifications = _merge(cur, body, NotifyConfig)
        rt.config_store.save()
    return cfg.to_dict(mask=True)["notifications"]


class CongruenceTestIn(BaseModel):
    base_url: str = ""
    api_key: str = ""


@app.post("/api/congruence/test")
async def congruence_test(body: CongruenceTestIn):
    """Fetch the store, cameras and zones the given server / key gives access to (drives the mapping UI)."""
    cur = rt.config_store.config.congruence
    url = (body.base_url or cur.base_url).strip().rstrip("/")
    key = cur.api_key if body.api_key in ("", "***") else body.api_key
    if not url or not key:
        raise HTTPException(422, "server URL and API key are required")
    try:
        data = await run_in_threadpool(fetch_config, url, key)
    except Exception as e:
        raise HTTPException(502, f"could not connect to {url}: {e}")
    return summarise_config(data)


@app.get("/api/congruence/status")
async def congruence_status():
    return rt.connector.status()


@app.post("/api/congruence/sync")
async def congruence_sync():
    """Managed mode: fetch the server's cameras / zones now and apply them (also runs every 5 minutes)."""
    cfg = rt.config_store.config.congruence
    if not cfg.managed:
        raise HTTPException(409, "managed mode is off (Settings > Congruence Eye)")
    if not (cfg.base_url and cfg.api_key):
        raise HTTPException(422, "server URL and API key are required")
    try:
        data = await run_in_threadpool(fetch_config, cfg.base_url, cfg.api_key)
    except Exception as e:
        raise HTTPException(502, f"could not connect to {cfg.base_url}: {e}")
    rt.connector.accept_config(data, dispatch=False)
    try:
        summary = await run_in_threadpool(rt.reconciler.run, data)
    except ValueError as e:
        raise HTTPException(502, str(e))
    return {"ok": True, **summary, "status": rt.reconciler.status()}


@app.put("/api/config/congruence")
async def put_congruence(body: dict):
    cfg = rt.config_store.config
    cur = cfg.congruence
    if body.get("api_key") in ("", "***", None):
        body["api_key"] = cur.api_key
    new = _merge(cur, body, CongruenceConfig)
    new.base_url = str(new.base_url or "").strip().rstrip("/")
    try:
        new.cameras = {str(k): int(v) for k, v in dict(new.cameras or {}).items() if v}
        new.zones = {str(k): int(v) for k, v in dict(new.zones or {}).items() if v}
    except (TypeError, ValueError):
        raise HTTPException(422, "camera and zone ids must be numbers")
    if new.managed:
        new.enabled = True                       # managed mode needs the connection
        new.cameras, new.zones = dict(cur.cameras), dict(cur.zones)   # regenerated by the sync, never edited by hand
    if new.enabled and not (new.base_url and new.api_key):
        raise HTTPException(422, "server URL and API key are required to enable the connector")
    with rt.config_store.lock:
        cfg.congruence = new
        rt.config_store.save()
    rt.connector.apply()                         # re-fetches the config on the next tick; managed mode then syncs
    return cfg.to_dict(mask=True)["congruence"]


@app.post("/api/notifications/test")
async def test_notifications():
    return await run_in_threadpool(rt.notifier.test)


# ---------------------------------------------------------------- evidence clips

@app.get("/api/clips")
async def list_clips(limit: int = Query(100, ge=1, le=1000)):
    return await run_in_threadpool(rt.clips.list, limit)


# ---------------------------------------------------------------- loss-prevention case export

@app.get("/api/cases/export.zip")
async def export_case(hours: float = Query(24, gt=0, le=24 * 31), types: str | None = None, camera: str | None = None):
    """One zip for an incident: the alerts of the period as CSV plus every evidence clip that belongs to them."""
    import csv, io, zipfile
    want = {t.strip() for t in types.split(",")} if types else None
    rows = await run_in_threadpool(rt.store.alerts, time.time() - hours * 3600, False, 5000)
    rows = [a for a in rows if (not want or a["type"] in want) and (not camera or a["camera_id"] == camera)]
    names = {c.id: c.name for c in rt.config_store.config.cameras}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["id", "started", "resolved", "type", "camera", "zone", "track_id", "message", "clip"])
        for a in rows:
            w.writerow([a["id"], datetime.fromtimestamp(a["started"]).isoformat(timespec="seconds"),
                        datetime.fromtimestamp(a["resolved"]).isoformat(timespec="seconds") if a.get("resolved") else "",
                        a["type"], names.get(a["camera_id"], a["camera_id"] or ""), a.get("zone") or "", a.get("track_id") or "",
                        a["message"], a.get("clip") or ""])
            clip = a.get("clip")
            if clip and (DATA_DIR / "clips" / clip).is_file():
                z.write(DATA_DIR / "clips" / clip, f"clips/{clip}")
        z.writestr("alerts.csv", out.getvalue())
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="case_{int(time.time())}.zip"'})


# ---------------------------------------------------------------- staffing (shopper-to-associate ratio)

@app.get("/api/staffing")
async def staffing(days: int = Query(7, ge=1, le=60), target_ratio: float = Query(4.0, gt=0)):
    """Per hour of day over the last N days: average customers, average staff on the floor, customers per staff
    (STAR) and the staff needed to hold target_ratio customers per staff."""
    import math
    rows = await run_in_threadpool(rt.store.minutes_since, time.time() - days * 86400)
    acc: dict[int, list] = {}
    for r in rows:
        h = datetime.fromtimestamp(r["ts"]).hour
        a = acc.setdefault(h, [0.0, 0.0, 0])
        a[0] += r["avg_occ"] or 0.0
        a[1] += r["staff_avg"] or 0.0
        a[2] += 1
    out = []
    for h in sorted(acc):
        cust, staff, n = acc[h][0] / acc[h][2], acc[h][1] / acc[h][2], acc[h][2]
        need = math.ceil(cust / target_ratio) if cust > 0.2 else 0
        out.append({"hour": h, "avg_customers": round(cust, 2), "avg_staff": round(staff, 2),
                    "customers_per_staff": round(cust / staff, 2) if staff > 0.05 else None,
                    "recommended_staff": need, "gap": round(need - staff, 1), "minutes": n})
    return {"days": days, "target_ratio": target_ratio, "hours": out,
            "understaffed_hours": [o["hour"] for o in out if o["gap"] >= 1], "overstaffed_hours": [o["hour"] for o in out if o["gap"] <= -1]}


# ---------------------------------------------------------------- counting accuracy audits

class AuditIn(BaseModel):
    camera_id: str
    start: float = Field(gt=0, description="unix seconds")
    end: float = Field(gt=0)
    manual_in: int = Field(ge=0)
    manual_out: int | None = Field(None, ge=0)
    note: str | None = Field(None, max_length=300)


def _accuracy(engine: int, manual: int | None) -> float | None:
    if manual is None:
        return None
    if manual == 0:
        return 100.0 if engine == 0 else 0.0
    return round(max(0.0, 100.0 - abs(engine - manual) / manual * 100.0), 1)


@app.get("/api/audits")
async def list_audits(limit: int = Query(50, ge=1, le=500)):
    names = {c.id: c.name for c in rt.config_store.config.cameras}
    rows = await run_in_threadpool(rt.store.audits, limit)
    for a in rows:
        a["camera"] = names.get(a["camera_id"], a["camera_id"])
    return rows


@app.get("/api/audits/live")
async def audit_live(camera_id: str, since: float = Query(gt=0)):
    """What the engine has counted on a camera since a manual count started (poll while counting by hand)."""
    _runtime(camera_id)
    c = await run_in_threadpool(rt.store.crossing_counts, camera_id, since, time.time())
    return {"camera_id": camera_id, "since": since, "now": time.time(), **c}


@app.post("/api/audits", status_code=201)
async def add_audit(body: AuditIn):
    """Close a manual count: compare it with the engine's crossings in the same window and keep the result."""
    _runtime(body.camera_id)
    if body.end <= body.start:
        raise HTTPException(422, "end must be after start")
    c = await run_in_threadpool(rt.store.crossing_counts, body.camera_id, body.start, body.end)
    a = {"camera_id": body.camera_id, "start_ts": body.start, "end_ts": body.end, "manual_in": body.manual_in, "manual_out": body.manual_out,
         "engine_in": c["in"], "engine_out": c["out"], "accuracy_in": _accuracy(c["in"], body.manual_in),
         "accuracy_out": _accuracy(c["out"], body.manual_out), "note": body.note}
    a["id"] = await run_in_threadpool(rt.store.add_audit, a)
    cam = rt.config_store.config.camera(body.camera_id)
    rt.connector.count_audit(a, cam.name if cam else body.camera_id)
    return a


# ---------------------------------------------------------------- live POS

@app.post("/api/pos/transactions")
async def pos_transactions(request: Request, forward: bool = True):
    """Bills from the POS (webhook). Any of: {"transactions": [...]}, a list, or one bill; see app/pos.py for field aliases."""
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(422, "body must be JSON")
    try:
        return await run_in_threadpool(rt.pos.ingest, body, "webhook", forward and rt.config_store.config.pos.forward)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/pos/status")
async def pos_status():
    return await run_in_threadpool(rt.pos.status)


@app.get("/api/pos/transactions")
async def pos_list(hours: float = Query(24, gt=0, le=24 * 92), limit: int = Query(500, ge=1, le=5000)):
    return await run_in_threadpool(rt.store.transactions, time.time() - hours * 3600, time.time() + 60, limit)


@app.post("/api/pos/scan")
async def pos_scan():
    """Process the CSV drop folder now (it is also scanned every 30 s)."""
    return {"bills": await run_in_threadpool(rt.pos.scan_inbox)}


@app.put("/api/config/pos")
async def put_pos(body: dict):
    cfg = rt.config_store.config
    with rt.config_store.lock:
        cfg.pos = _merge(cfg.pos, body, PosConfig)
        rt.config_store.save()
    return cfg.to_dict()["pos"]


@app.put("/api/config/report")
async def put_report(body: dict):
    cfg = rt.config_store.config
    if "daily_time" in body and not re.fullmatch(r"\d{2}:\d{2}", str(body["daily_time"])):
        raise HTTPException(422, "daily_time must be HH:MM")
    with rt.config_store.lock:
        cfg.report = _merge(cfg.report, body, ReportConfig)
        rt.config_store.save()
    return cfg.to_dict()["report"]


@app.put("/api/config/demographics")
async def put_demographics(body: dict):
    cfg = rt.config_store.config
    new = _merge(cfg.demographics, body, DemographicsConfig)
    if new.min_face < 8 or new.sample_every < 1 or new.max_samples < 1:
        raise HTTPException(422, "invalid demographics settings")
    needs_restart = new.enabled and not cfg.demographics.enabled
    with rt.config_store.lock:
        cfg.demographics = new
        rt.config_store.save()
    if needs_restart:   # pipelines only load the face models when started with demographics on
        for cam in list(cfg.cameras):
            if cam.enabled:
                await run_in_threadpool(rt.manager._stop_runtime, cam.id)
                await run_in_threadpool(rt.manager._start_runtime, cam)
    return cfg.to_dict()["demographics"]


class UniformIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    h: int = Field(ge=0, le=179)
    s: int = Field(ge=0, le=255)
    v: int = Field(ge=0, le=255)
    logo_hues: list[int] = Field([], max_length=4)       # OpenCV hues (0-179) of a printed logo; staff only when it is visible
    logo_min_pct: float = Field(0.4, ge=0.05, le=20)      # share of the chest crop the logo colours must cover
    logo_white: bool = False                              # the print is white on a dark garment
    logo_white_min_pct: float = Field(5, ge=1, le=30)     # share of the chest crop the white print must cover

    def model_post_init(self, __context) -> None:
        if any(h < 0 or h > 179 for h in self.logo_hues):
            raise ValueError("logo hues must be between 0 and 179")


class BrandIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    hue: int = Field(ge=0, le=179)
    tol: int = Field(6, ge=1, le=20)


class StaffIn(BaseModel):
    after_minutes: float = Field(45, ge=0, le=24 * 60)
    uniforms: list[UniformIn] = []
    brands: list[BrandIn] = []


@app.put("/api/config/staff")
async def put_staff(body: StaffIn):
    cfg = StaffConfig(after_minutes=body.after_minutes, uniforms=[u.model_dump() for u in body.uniforms],
                      brands=[b.model_dump() for b in body.brands])
    await run_in_threadpool(rt.manager.apply_staff, cfg)
    return rt.config_store.config.to_dict()["staff"]


class TrackStaffIn(BaseModel):
    staff: bool = True
    learn_uniform: bool = True
    name: str = Field("", max_length=40)


@app.post("/api/cameras/{cam_id}/tracks/{track_id}/staff")
async def tag_track_staff(cam_id: str, track_id: int, body: TrackStaffIn):
    r = _runtime(cam_id)
    res = r.analytics.set_staff(track_id, body.staff)
    if res is None:
        raise HTTPException(404, "that person is no longer in view")
    learned = None
    staff = rt.config_store.config.staff
    generic_only = body.staff and body.learn_uniform and res.get("torso") and any(u.get("logo_hues") or u.get("logo_white") for u in staff.uniforms)         and uniform_kind(*(int(x) for x in res["torso"])) != "distinct"
    if generic_only:
        # a logo uniform is configured: a plain dark / white colour would mark every similar garment as staff
        return {"ok": True, "torso": res.get("torso"), "learned": None,
                "note": "colour not learned: the staff uniform is recognised by its logo; this person is tagged for now"}
    if body.staff and body.learn_uniform and res.get("torso"):
        h, s, v = res["torso"]
        learned = {"name": body.name or f"Uniform {len(staff.uniforms) + 1}", "h": int(h), "s": int(s), "v": int(v)}
        learned["kind"] = uniform_kind(int(h), int(s), int(v))
        cfg = StaffConfig(after_minutes=staff.after_minutes, uniforms=staff.uniforms + [{k: learned[k] for k in ("name", "h", "s", "v")}])
        await run_in_threadpool(rt.manager.apply_staff, cfg)
    return {"ok": True, "torso": res.get("torso"), "learned": learned}


@app.post("/api/reset")
async def reset():
    rt.manager.reset_counters()
    return {"ok": True}


# ---------------------------------------------------------------- analytics

@app.get("/api/traffic")
async def traffic(hours: float | None = None, day: str | None = None, camera: str | None = None):
    since, until = _range(hours, day)
    return await run_in_threadpool(rt.store.traffic, since, until, rt.config_store.config.thresholds.engage_seconds, camera)


@app.get("/api/demographics")
async def demographics(hours: float | None = None, day: str | None = None, camera: str | None = None):
    since, until = _range(hours, day)
    return await run_in_threadpool(rt.store.demographics, since, until, camera)


@app.get("/api/queues")
async def queues(hours: float | None = None, day: str | None = None):
    since, until = _range(hours, day)
    data = await run_in_threadpool(rt.store.queues, since, until, _zone_roles())
    names = {c.id: c.name for c in rt.config_store.config.cameras}
    for q in data["queues"]:
        q["camera"] = names.get(q["camera_id"], q["camera_id"])
    return data


@app.get("/api/zones")
async def zones(hours: float | None = None, day: str | None = None):
    since, until = _range(hours, day)
    data = await run_in_threadpool(rt.store.zones, since, until, _zone_roles(),
                                   rt.config_store.config.thresholds.zone_engage_seconds)
    names = {c.id: c.name for c in rt.config_store.config.cameras}
    for z in data["zones"]:
        z["camera"] = names.get(z["camera_id"], z["camera_id"])
    for t in data["transitions"]:
        t["camera"] = names.get(t["camera_id"], t["camera_id"])
    return data


@app.get("/api/timeline")
async def timeline(hours: float = Query(24, gt=0, le=24 * 31), camera: str | None = None):
    return await run_in_threadpool(rt.store.minutes_since, time.time() - hours * 3600, camera)


@app.get("/api/events")
async def events(hours: float = Query(24, gt=0, le=24 * 31), limit: int = Query(200, ge=1, le=2000),
                 types: str | None = None, camera: str | None = None):
    type_list = [t.strip() for t in types.split(",") if t.strip()] if types else None
    return await run_in_threadpool(rt.store.events, time.time() - hours * 3600, limit, type_list, camera)


@app.get("/api/visits")
async def visits(hours: float | None = None, day: str | None = None, limit: int = Query(300, ge=1, le=5000)):
    since, until = _range(hours, day)
    return await run_in_threadpool(rt.store.visits, since, until, limit)


# ---------------------------------------------------------------- alerts

@app.get("/api/alerts")
async def alerts(hours: float = Query(24, gt=0, le=24 * 31), open: bool = False, limit: int = Query(300, ge=1, le=5000)):
    rows = await run_in_threadpool(rt.store.alerts, time.time() - hours * 3600, open, limit)
    names = {c.id: c.name for c in rt.config_store.config.cameras}
    for a in rows:
        a["camera"] = names.get(a["camera_id"], a["camera_id"] or "")
    return rows


@app.get("/api/alerts/summary")
async def alerts_summary(hours: float | None = None, day: str | None = None):
    since, until = _range(hours, day)
    counts = await run_in_threadpool(rt.store.alert_counts, since, until)
    return {"counts": counts, "open": len(rt.alerts.list_open()), "since": since, "until": until}


@app.post("/api/alerts/{alert_id}/ack")
async def ack_alert(alert_id: int):
    rt.alerts.ack(alert_id)
    return {"ok": True}


@app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int):
    rt.alerts.resolve_id(alert_id)
    return {"ok": True}


# ---------------------------------------------------------------- sales + reports

class SalesIn(BaseModel):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    bills: int = Field(ge=0)
    revenue: float = Field(0, ge=0)


@app.get("/api/sales")
async def get_sales():
    return await run_in_threadpool(rt.store.sales_rows)


@app.post("/api/sales")
async def post_sales(body: SalesIn):
    await run_in_threadpool(rt.store.set_sales, body.date, body.bills, body.revenue)
    return {"ok": True}


@app.get("/api/report/daily.xlsx")
async def daily_report(day: str | None = None):
    d = date.today()
    if day:
        try:
            d = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(422, "day must be YYYY-MM-DD")
    name, data = await run_in_threadpool(build_daily_report, rt.store, rt.config_store.config, d)
    return Response(data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/api/report/email")
async def email_report(day: str | None = None):
    d = date.today() if not day else datetime.strptime(day, "%Y-%m-%d").date()
    name, data = await run_in_threadpool(build_daily_report, rt.store, rt.config_store.config, d)
    cfg = rt.config_store.config
    to = cfg.report.email_to or cfg.notifications.email_to
    try:
        result = await run_in_threadpool(rt.notifier.send_email, f"Daily store report {d}", "Attached: footfall, zones and alerts.", to, (name, data))
    except Exception as e:
        raise HTTPException(502, f"email failed: {e}")
    return {"result": result}
