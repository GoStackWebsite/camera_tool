"""Zone and count line editing from the camera view (reshape, rename, change the role, add, remove; move
the count line's ends and choose its IN side).

* Hand-made cameras: the change is applied to data/config.json through the manager.
* Managed cameras: Congruence Eye stays the source of truth. The change is written to the server's zone
  API, then the server configuration is pulled and applied, so both sides always show the same zones.
  That API needs an admin sign-in (the edge key cannot change master data); the page signs in through
  ``ce_login`` and sends the token with every edit. The token is never stored by this app.

The count line travels as ``{"op": "line", "points": [A, B]}`` with this app's orientation (the + side of
A->B is IN), or ``{"op": "line", "remove": true}``. On Congruence Eye it is the camera's ENTRY_LINE zone,
whose IN side is a screen direction (``inDirection``): the direction nearest to the line's IN normal.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import replace

import requests

from .config import ZONE_ROLES, ZoneConfig
from .congruence import fetch_config
from .sync import DIRECTIONS, ROLE_TO_TYPE, frame_to_stage

log = logging.getLogger("zone_edit")

TIMEOUT = 15
ADMIN_ROLES = ("SUPER_ADMIN", "ADMIN")
MAX_NAME = 40
MAX_POINTS = 64
MIN_LINE = 0.02          # shortest count line (fraction of the frame): a shorter one is a slip of the mouse
LINE_NAME = "Entrance"   # name of a count line created from the camera view (never "Pass...": that is the pass-by line)
LINE_LABEL = "count line"


class ZoneEditError(Exception):
    """A problem the page can show as is; ``status`` becomes the HTTP status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------- validation

def clean_name(name) -> str:
    n = re.sub(r"\s+", " ", str(name or "")).strip()
    if not n:
        raise ZoneEditError(422, "a zone needs a name")
    if len(n) > MAX_NAME:
        raise ZoneEditError(422, f"zone names can be at most {MAX_NAME} characters")
    return n


def clean_points(points) -> list[list[float]]:
    """Clamp to the frame, round, drop repeated corners; needs a real area."""
    out: list[list[float]] = []
    for p in points or []:
        try:
            q = [round(min(1.0, max(0.0, float(p[0]))), 4), round(min(1.0, max(0.0, float(p[1]))), 4)]
        except (TypeError, ValueError, IndexError):
            raise ZoneEditError(422, "zone corners must be [x, y] numbers")
        if not out or q != out[-1]:
            out.append(q)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    if len(out) < 3:
        raise ZoneEditError(422, "a zone needs at least 3 corners")
    if len(out) > MAX_POINTS:
        raise ZoneEditError(422, f"a zone can have at most {MAX_POINTS} corners")
    n = len(out)
    area = abs(sum(out[i][0] * out[(i + 1) % n][1] - out[(i + 1) % n][0] * out[i][1] for i in range(n))) / 2
    if area < 1e-4:
        raise ZoneEditError(422, "the zone is too small: drag its corners apart")
    return out


def clean_line(points) -> list[list[float]]:
    """Two ends, clamped to the frame and rounded, far enough apart to be a line."""
    pts = list(points or [])
    if len(pts) != 2:
        raise ZoneEditError(422, "the count line needs exactly 2 ends")
    try:
        out = [[round(min(1.0, max(0.0, float(p[0]))), 4), round(min(1.0, max(0.0, float(p[1]))), 4)] for p in pts]
    except (TypeError, ValueError, IndexError):
        raise ZoneEditError(422, "the count line's ends must be [x, y] numbers")
    if math.hypot(out[1][0] - out[0][0], out[1][1] - out[0][1]) < MIN_LINE:
        raise ZoneEditError(422, "the count line is too short: drag its ends apart")
    return out


def in_direction(line) -> str:
    """Congruence Eye's inDirection for a line whose + side of A->B is IN: the screen direction (y down)
    nearest to the IN normal (-dy, dx). Its dot product with the normal is always positive, so the server's
    orient_line gives back exactly this line, also after the 16:9 stage mapping (it scales x and y by
    positive factors, which keeps the sign)."""
    (ax, ay), (bx, by) = line
    nx, ny = -(by - ay), bx - ax
    return max(DIRECTIONS, key=lambda d: DIRECTIONS[d][0] * nx + DIRECTIONS[d][1] * ny)


def same_points(a, b) -> bool:
    return len(a) == len(b) and all(abs(float(p[0]) - float(q[0])) < 5e-5 and abs(float(p[1]) - float(q[1])) < 5e-5
                                    for p, q in zip(a, b))


def _find(zones: list[ZoneConfig], name) -> ZoneConfig:
    want = str(name or "")
    z = next((z for z in zones if z.name == want), None) or \
        next((z for z in zones if z.name.lower() == want.strip().lower()), None)
    if z is None:
        raise ZoneEditError(404, f"no zone named '{want}' on this camera (it may have been changed elsewhere: reopen the camera)")
    return z


def _check_free(zones: list[ZoneConfig], name: str, skip: ZoneConfig | None = None) -> None:
    if any(z is not skip and z.name.lower() == name.lower() for z in zones):
        raise ZoneEditError(422, f"another zone on this camera is already called '{name}'")


def _label(ch: dict) -> str:
    if ch.get("op") == "line":
        return LINE_LABEL
    return str(ch.get("zone") or ch.get("name") or "zone")


def _wanted_line(ch: dict) -> list[list[float]] | None:
    """The line a "line" change asks for (None = remove it)."""
    if ch.get("remove"):
        return None
    if ch.get("points") is None:
        raise ZoneEditError(422, "send the count line's 2 ends, or remove: true")
    return clean_line(ch["points"])


# ---------------------------------------------------------------- hand-made cameras

def _apply_local_one(zones: list[ZoneConfig], ch: dict, renames: dict[str, str]) -> None:
    op = ch.get("op")
    if op == "create":
        name = clean_name(ch.get("name"))
        _check_free(zones, name)
        role = ch.get("role") or "display"
        if role not in ZONE_ROLES:
            raise ZoneEditError(422, f"unknown zone role '{role}'")
        zones.append(ZoneConfig(name, clean_points(ch.get("points")), role))
        return
    z = _find(zones, ch.get("zone"))
    if op == "delete":
        zones.remove(z)
        return
    if op != "update":
        raise ZoneEditError(422, f"unknown operation '{op}'")
    name, role, points = z.name, z.role, z.points
    if ch.get("name") is not None:
        name = clean_name(ch["name"])
        if name != z.name:
            _check_free(zones, name, skip=z)
    if ch.get("role") is not None:
        if ch["role"] not in ZONE_ROLES:
            raise ZoneEditError(422, f"unknown zone role '{ch['role']}'")
        role = ch["role"]
    if ch.get("points") is not None:
        points = clean_points(ch["points"])
    if name != z.name:                      # A -> B then B -> C in one save is A -> C
        origin = next((o for o, n in renames.items() if n == z.name), z.name)
        renames[origin] = name
    zones[zones.index(z)] = replace(z, name=name, role=role, points=points)


def apply_local(config_store, manager, cam_id: str, changes: list[dict]) -> dict:
    with config_store.lock:
        cam = config_store.config.camera(cam_id)
        if cam is None:
            raise ZoneEditError(404, "no such camera")
        if cam.managed:
            raise ZoneEditError(409, "this camera is managed by Congruence Eye")
        zones = list(cam.zones)
        line = cam.line
    renames: dict[str, str] = {}
    applied, errors = 0, []
    for ch in changes:
        try:
            if ch.get("op") == "line":
                want = _wanted_line(ch)
                if want == line or (want and line and same_points(want, line)):
                    continue
                line = want
            else:
                _apply_local_one(zones, ch, renames)
            applied += 1
        except ZoneEditError as e:
            errors.append({"zone": _label(ch), "error": e.message})
    if applied:
        manager.update_camera(replace(cam, zones=zones, line=line), renames=renames)
        if renames:                          # a hand-made mapping to Congruence Eye follows the new name
            with config_store.lock:
                m = dict(config_store.config.congruence.zones or {})
                moved = False
                for old, new in renames.items():
                    if f"{cam_id}/{old}" in m:
                        m[f"{cam_id}/{new}"] = m.pop(f"{cam_id}/{old}")
                        moved = True
                if moved:
                    config_store.config.congruence.zones = m
                    config_store.save()
    return {"applied": applied, "errors": errors, "synced": True}


# ---------------------------------------------------------------- Congruence Eye admin API

def _server_message(r) -> str:
    try:
        m = r.json().get("message")
    except Exception:
        m = None
    if isinstance(m, list):
        m = "; ".join(str(x) for x in m)
    return str(m or r.text or f"HTTP {r.status_code}")[:300]


def _user_view(u: dict) -> dict:
    role = str(u.get("role") or "")
    return {"name": str(u.get("name") or u.get("email") or ""), "email": str(u.get("email") or ""), "role": role,
            "can_edit": role in ADMIN_ROLES}


def ce_login(base_url: str, email: str, password: str) -> dict:
    """Sign in to Congruence Eye; returns {"token", "user"}. Nothing is stored or logged here."""
    base = str(base_url or "").rstrip("/")
    if not base:
        raise ZoneEditError(409, "the Congruence Eye server is not configured (Settings > Congruence Eye)")
    try:
        r = requests.post(base + "/api/auth/login", json={"email": email, "password": password}, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise ZoneEditError(502, f"could not reach Congruence Eye at {base}: {type(e).__name__}")
    if r.status_code == 401:
        raise ZoneEditError(401, "wrong email or password")
    if r.status_code == 400:
        raise ZoneEditError(422, _server_message(r))
    if not r.ok:
        raise ZoneEditError(502, f"Congruence Eye answered {r.status_code}: {_server_message(r)}")
    try:
        body = r.json()
    except ValueError:
        raise ZoneEditError(502, "Congruence Eye did not return JSON")
    token = body.get("accessToken") or body.get("token") or body.get("access_token")
    if not token:
        raise ZoneEditError(502, "the sign-in response had no token")
    return {"token": token, "user": _user_view(body.get("user") or {})}


class CEAdmin:
    """The few admin calls zone editing needs, made with the signed-in person's token."""

    def __init__(self, base_url: str, token: str):
        self.base = str(base_url or "").rstrip("/")
        self.token = token

    def request(self, method: str, path: str, json: dict | None = None):
        try:
            r = requests.request(method, self.base + path, json=json, timeout=TIMEOUT,
                                 headers={"Authorization": f"Bearer {self.token}"})
        except requests.RequestException as e:
            raise ZoneEditError(502, f"could not reach Congruence Eye at {self.base}: {type(e).__name__}")
        if r.status_code == 401:
            raise ZoneEditError(401, "your Congruence Eye sign-in has expired: sign in again")
        if r.status_code == 403:
            raise ZoneEditError(403, "this Congruence Eye account cannot change zones: an Admin account is needed")
        if r.status_code == 404:
            raise ZoneEditError(404, f"Congruence Eye no longer has this zone ({_server_message(r)}): press Sync now in Settings")
        if 400 <= r.status_code < 500:
            raise ZoneEditError(422, f"Congruence Eye refused the change: {_server_message(r)}")
        if not r.ok:
            raise ZoneEditError(502, f"Congruence Eye answered {r.status_code}: {_server_message(r)}")
        try:
            return r.json()
        except ValueError:
            return {}


def ce_me(base_url: str, token: str) -> dict:
    """Who a token belongs to (401 once it has expired)."""
    if not token:
        raise ZoneEditError(401, "not signed in")
    return {"user": _user_view(CEAdmin(base_url, token).request("GET", "/api/auth/me") or {})}


# ---------------------------------------------------------------- managed cameras

def _managed_type(role) -> str:
    if role not in ROLE_TO_TYPE:
        raise ZoneEditError(422, f"Congruence Eye has no '{role}' zone type: use display, queue or staff")
    return ROLE_TO_TYPE[role]


def _apply_managed_line(api: CEAdmin, cam, ch: dict, store_id) -> bool:
    """The count line on the server: the camera's ENTRY_LINE zone (the pass-by line is a different one)."""
    want = _wanted_line(ch)
    zid = (cam.remote_line or {}).get("zone_id")
    if want is None:
        if not zid:
            return False
        api.request("DELETE", f"/api/zones/{int(zid)}")             # the server keeps it, marked inactive
        return True
    if zid and cam.line and same_points(want, [[min(1.0, max(0.0, float(q[0]))), min(1.0, max(0.0, float(q[1])))]
                                               for q in cam.line]):
        return False
    body = {"polygon": [frame_to_stage(p, cam.frame_w, cam.frame_h) for p in want], "inDirection": in_direction(want)}
    if zid:
        api.request("PATCH", f"/api/zones/{int(zid)}", body)
        return True
    if not store_id:
        raise ZoneEditError(409, "the store is not known yet: press Sync now in Settings > Congruence Eye")
    api.request("POST", "/api/zones", {"storeId": int(store_id), "cameraId": int(cam.remote_id), "name": LINE_NAME,
                                       "type": "ENTRY_LINE", **body})
    return True


def _apply_managed_one(api: CEAdmin, cam, zones: list[ZoneConfig], ch: dict, store_id) -> bool:
    """One change on the server. Returns False when there was nothing to change."""
    op = ch.get("op")
    if op == "line":
        return _apply_managed_line(api, cam, ch, store_id)

    def stage(pts):
        return [frame_to_stage(p, cam.frame_w, cam.frame_h) for p in pts]

    if op == "create":
        name = clean_name(ch.get("name"))
        _check_free(zones, name)
        role = ch.get("role") or "display"
        ztype = _managed_type(role)
        pts = clean_points(ch.get("points"))
        if not store_id:
            raise ZoneEditError(409, "the store is not known yet: press Sync now in Settings > Congruence Eye")
        api.request("POST", "/api/zones", {"storeId": int(store_id), "cameraId": int(cam.remote_id), "name": name,
                                           "type": ztype, "polygon": stage(pts)})
        zones.append(ZoneConfig(name, pts, role))
        return True
    z = _find(zones, ch.get("zone"))
    if not z.remote_id:
        raise ZoneEditError(409, f"'{z.name}' is not linked to a Congruence Eye zone: press Sync now in Settings")
    if op == "delete":
        api.request("DELETE", f"/api/zones/{int(z.remote_id)}")     # the server keeps it, marked inactive
        zones.remove(z)
        return True
    if op != "update":
        raise ZoneEditError(422, f"unknown operation '{op}'")
    payload: dict = {}
    name, role, points = z.name, z.role, z.points
    if ch.get("name") is not None:
        name = clean_name(ch["name"])
        if name != z.name:
            _check_free(zones, name, skip=z)
            payload["name"] = name
    if ch.get("role") is not None and ch["role"] != z.role:
        payload["type"] = _managed_type(ch["role"])
        role = ch["role"]
    if ch.get("points") is not None:
        pts = clean_points(ch["points"])
        # compare against the stored shape as the editor sees it (clamped to the frame): a zone the server
        # drew past the edge of a non-16:9 frame must not look edited when nobody touched it
        seen = [[min(1.0, max(0.0, float(q[0]))), min(1.0, max(0.0, float(q[1])))] for q in z.points]
        if not same_points(pts, seen):
            payload["polygon"] = stage(pts)
            points = pts
    if not payload:
        return False
    api.request("PATCH", f"/api/zones/{int(z.remote_id)}", payload)
    zones[zones.index(z)] = replace(z, name=name, role=role, points=points)
    return True


def apply_managed(config_store, connector, reconciler, cam_id: str, changes: list[dict], token: str) -> dict:
    cc = config_store.config.congruence
    cam = config_store.config.camera(cam_id)
    if cam is None:
        raise ZoneEditError(404, "no such camera")
    if not cam.managed or not cam.remote_id:
        raise ZoneEditError(409, "this camera is not managed by Congruence Eye")
    if not (cc.base_url and cc.api_key):
        raise ZoneEditError(409, "the Congruence Eye connection is not configured (Settings > Congruence Eye)")
    if not token:
        raise ZoneEditError(401, "sign in to Congruence Eye to change zones")
    remote = {}
    if connector is not None:
        remote = connector.remote or getattr(connector, "remote_cached", None) or {}
    store_id = (remote.get("store") or {}).get("id")
    api = CEAdmin(cc.base_url, token)
    zones = list(cam.zones)
    applied, errors = 0, []
    for ch in changes:
        try:
            if _apply_managed_one(api, cam, zones, ch, store_id):
                applied += 1
        except ZoneEditError as e:
            if e.status in (401, 403):
                if not applied:
                    raise                    # nothing written yet: the page asks for a sign-in and retries
                errors.append({"zone": _label(ch), "error": e.message})
                break
            errors.append({"zone": _label(ch), "error": e.message})
    result = {"applied": applied, "errors": errors, "synced": True}
    if applied:
        log.info("zone edit on %s (Congruence Eye camera %s): %d change(s) written to the server",
                 cam_id, cam.remote_id, applied)
        try:                                 # pull the server's truth back so this app matches it exactly
            data = fetch_config(cc.base_url, cc.api_key)
            connector.accept_config(data, dispatch=False)
            reconciler.run(data)
        except Exception as e:
            log.warning("zone edit saved on the server but the sync back failed: %s", e)
            result.update(synced=False, sync_error=f"{type(e).__name__}: {e}"[:200])
    return result
