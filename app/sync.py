"""Managed mode: mirror Congruence Eye's cameras, count line and zones into this app.

With Settings -> Congruence Eye -> "managed" on, the server's ``GET /api/ingest/config`` is the source of
truth and nothing is drawn or mapped by hand here:

  * every active server camera becomes a local camera ``ce<id>``; its RTSP URL comes from the server when
    the camera is linked to a recorder there, otherwise from a local override entered in Settings;
  * the camera's ENTRY_LINE zone becomes the count line, oriented from the server's ``inDirection``;
  * SECTION / QUEUE / STAFF_ONLY zones become display / queue / staff zones (SHELF has no local equivalent);
  * the id mapping used by the connector is regenerated from the server ids.

Coordinates: the server's zone editor draws on a 16:9 stage over the snapshots this app uploads, which the
connector letterboxes to 16:9, so the drawn points are mapped back onto the real frame here
(``stage_to_frame``). For 16:9 cameras that mapping is the identity.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from dataclasses import replace

from .config import CameraConfig, ZoneConfig

log = logging.getLogger("sync")

STAGE_ASPECT = 16 / 9
ROLE_MAP = {"SECTION": "display", "QUEUE": "queue", "STAFF_ONLY": "staff"}   # ENTRY_LINE -> count line, SHELF skipped


def is_passby_name(name) -> bool:
    n = str(name or "").strip().lower().replace("-", "").replace(" ", "").replace("_", "")
    return n.startswith("pass")
DIRECTIONS = {"UP": (0.0, -1.0), "DOWN": (0.0, 1.0), "LEFT": (-1.0, 0.0), "RIGHT": (1.0, 0.0)}


# ---------------------------------------------------------------- geometry

def local_id(remote_id: int) -> str:
    return f"ce{int(remote_id)}"


def stage_to_frame(p, fw: int, fh: int) -> list[float]:
    """Editor (16:9 stage) coordinates -> frame coordinates: the inverse of the connector's letterboxing.
    Unknown frame size -> identity (re-derived once the first frame has been decoded)."""
    x, y = float(p[0]), float(p[1])
    if not fw or not fh:
        return [x, y]
    ai = fw / fh
    if ai >= STAGE_ASPECT:                                   # wide frame: bars top and bottom
        return [x, (y - 0.5) * (ai / STAGE_ASPECT) + 0.5]
    return [(x - 0.5) * (STAGE_ASPECT / ai) + 0.5, y]       # narrow frame: bars left and right


def orient_line(pts, in_direction: str | None) -> list[list[float]]:
    """Order the two points so this app's IN side (the + side of A->B) is the side reached by moving in the
    server's inDirection (screen directions, y down). Same rule as Congruence Eye's own engine."""
    ax, ay = float(pts[0][0]), float(pts[0][1])
    bx, by = float(pts[1][0]), float(pts[1][1])
    dx, dy = DIRECTIONS.get(str(in_direction or "UP").upper(), DIRECTIONS["UP"])
    in_sign = (bx - ax) * dy - (by - ay) * dx
    return [[ax, ay], [bx, by]] if in_sign >= 0 else [[bx, by], [ax, ay]]


def clean_zone_name(name) -> str:
    return re.sub(r"\s+", " ", str(name or "")).strip()[:40] or "Zone"


def usable_url(raw) -> str:
    """The server masks recorder passwords it cannot decrypt as ****: such a URL cannot be opened."""
    u = str(raw or "").strip()
    return "" if (not u or "****" in u) else u


def plan_cameras(remote: dict, frame_sizes: dict[str, tuple[int, int]]) -> tuple[list[CameraConfig], list[str]]:
    """Desired local cameras from the server's config (active cameras and zones only), in server id order."""
    notes: list[str] = []
    zones_by_cam: dict[int, list[dict]] = {}
    for z in remote.get("zones") or []:
        if not isinstance(z, dict) or z.get("isActive", True) is False or z.get("cameraId") is None:
            continue
        zones_by_cam.setdefault(int(z["cameraId"]), []).append(z)
    desired: list[CameraConfig] = []
    cams = [c for c in (remote.get("cameras") or []) if isinstance(c, dict) and c.get("id") is not None]
    for c in sorted(cams, key=lambda c: int(c["id"])):
        if c.get("isActive", True) is False:
            continue
        rid = int(c["id"])
        lid = local_id(rid)
        fw, fh = frame_sizes.get(lid, (0, 0))
        zl = sorted(zones_by_cam.get(rid, []), key=lambda z: int(z.get("id", 0)))
        all_lines = [z for z in zl if z.get("type") == "ENTRY_LINE" and len(z.get("polygon") or []) == 2]
        # an ENTRY_LINE whose name starts with "pass" (e.g. "Pass-by") is the street-side line: everyone crossing
        # it is a passer-by and walk-ins / passers-by is the capture rate
        pb = [z for z in all_lines if is_passby_name(z.get("name"))]
        lines = [z for z in all_lines if not is_passby_name(z.get("name"))]
        line = remote_line = passby_line = remote_passby = None
        if lines:
            z = lines[0]
            line = orient_line([stage_to_frame(p, fw, fh) for p in z["polygon"]], z.get("inDirection"))
            remote_line = {"zone_id": int(z["id"]), "points": [[float(p[0]), float(p[1])] for p in z["polygon"]],
                           "in_direction": str(z.get("inDirection") or "UP").upper()}
            if len(lines) > 1:
                notes.append(f"{c.get('name')}: {len(lines)} entry lines on the server, using '{z.get('name')}'")
        if pb:
            z = pb[0]
            passby_line = [stage_to_frame(p, fw, fh) for p in z["polygon"]]
            remote_passby = {"zone_id": int(z["id"]), "points": [[float(p[0]), float(p[1])] for p in z["polygon"]]}
        zones: list[ZoneConfig] = []
        used: set[str] = set()
        for z in zl:
            role = ROLE_MAP.get(z.get("type"))
            poly = z.get("polygon") or []
            if role is None or len(poly) < 3:
                continue
            name = base = clean_zone_name(z.get("name"))
            n = 2
            while name.lower() in used:
                name = f"{base[:34]} ({n})"
                n += 1
            used.add(name.lower())
            zones.append(ZoneConfig(name, [stage_to_frame(p, fw, fh) for p in poly], role,
                                    remote_id=int(z["id"]), remote_points=[[float(p[0]), float(p[1])] for p in poly]))
        desired.append(CameraConfig(id=lid, name=str(c.get("name") or lid)[:60], url=usable_url(c.get("rtspUrl")),
                                    line=line, zones=zones, managed=True, remote_id=rid, remote_line=remote_line,
                                    passby_line=passby_line, remote_passby=remote_passby, frame_w=fw, frame_h=fh))
    return desired, notes


def _signature(cam: CameraConfig) -> str:
    return json.dumps(cam.to_dict(), sort_keys=True)


# ---------------------------------------------------------------- reconciler

class Reconciler:
    """Applies the server's configuration to the local cameras on its own thread (starting a camera loads
    the detector, which takes seconds, so this never runs on the connector or health threads)."""

    def __init__(self, config_store, manager, store, connector):
        self.config_store = config_store
        self.manager = manager
        self.store = store
        self.connector = connector
        self._lock = threading.Lock()          # one reconcile / re-derive / local edit at a time
        self._pending: dict | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="ce-sync", daemon=True)
        self.last_sync = 0.0
        self.last_error = ""
        self.summary: dict = {}

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    @property
    def cfg(self):
        return self.config_store.config.congruence

    def request(self, remote: dict) -> None:
        """Queue a server config for the sync thread (cheap; called from the connector thread)."""
        self._pending = remote
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                break
            remote, self._pending = self._pending, None
            if remote is None:
                continue
            try:
                self.run(remote)
            except Exception:
                log.exception("Congruence Eye sync failed")

    def run(self, remote: dict) -> dict:
        """Apply a server config now (Sync now); returns the summary. Raises on a bad config."""
        with self._lock:
            if not isinstance(remote, dict) or not isinstance(remote.get("cameras"), list) \
                    or not isinstance(remote.get("zones"), list):
                self.last_error = "the server config has no cameras / zones lists"
                raise ValueError(self.last_error)
            try:
                self.summary = self._reconcile(remote)
                self.last_sync = time.time()
                self.last_error = ""
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                raise
            return self.summary

    def status(self) -> dict:
        return {"managed": bool(self.cfg.managed), "last_sync": self.last_sync, "last_error": self.last_error,
                "summary": self.summary, "needs_url": list(self.summary.get("needs_url", []))}

    # ---- the sync itself --------------------------------------------------
    def _frame_sizes(self) -> dict[str, tuple[int, int]]:
        sizes: dict[str, tuple[int, int]] = {}
        with self.config_store.lock:
            for c in self.config_store.config.cameras:
                if c.frame_w and c.frame_h:
                    sizes[c.id] = (c.frame_w, c.frame_h)
        with self.manager.lock:
            for cid, rt in self.manager.runtimes.items():
                if rt.camera.width and rt.camera.height:
                    sizes[cid] = (rt.camera.width, rt.camera.height)
        return sizes

    def _reconcile(self, remote: dict) -> dict:
        cs = self.config_store
        # ---- 1. plan (fast) under the config lock
        with cs.lock:
            cfg = cs.config
            if not cfg.congruence.managed:
                return {"skipped": "managed mode is off"}
            current = {c.id: c for c in cfg.cameras if c.managed}
            unmanaged = [c for c in cfg.cameras if not c.managed]
            old_map = {str(k): int(v) for k, v in (cfg.congruence.cameras or {}).items() if v}
            overrides = {str(k): str(v) for k, v in (cfg.congruence.url_overrides or {}).items() if v}
            if not current and cs.path.exists():          # first switch: keep the hand-made config around
                try:
                    shutil.copyfile(cs.path, cs.path.with_name("config.pre-managed.json"))
                except OSError:
                    log.exception("could not snapshot config.json before switching to managed mode")
            frame_sizes = self._frame_sizes()
            for u in unmanaged:                           # a hand-made camera about to be replaced knows its frame size
                rid = old_map.get(u.id)
                if rid and local_id(rid) not in current and u.id in frame_sizes:
                    frame_sizes.setdefault(local_id(rid), frame_sizes[u.id])
            desired, notes = plan_cameras(remote, frame_sizes)
        # ---- 2. apply, outside the lock (camera starts are slow and lock on their own)
        added = updated = removed = disabled = 0
        needs_url: list[dict] = []
        donors_used: set[str] = set()
        for d in desired:
            cur = current.get(d.id)
            donor = None
            if cur is None:
                donor = next((u for u in unmanaged if u.id not in donors_used and old_map.get(u.id) == d.remote_id), None)
            if donor is not None:                         # migration of a hand-made camera
                donors_used.add(donor.id)
                d.url_override = "" if d.url else donor.url
                d.target_fps, d.blur_faces = donor.target_fps, donor.blur_faces
                try:
                    self.manager.remove_camera(donor.id)
                except KeyError:
                    pass
                try:
                    self.store.rename_camera(donor.id, d.id)
                except Exception:
                    log.exception("could not move the history of %s to %s", donor.id, d.id)
                log.info("managed: '%s' (%s) replaces the hand-made camera '%s'", d.name, d.id, donor.id)
            elif cur is not None:                         # keep the local-only settings
                d.url_override, d.target_fps, d.blur_faces = cur.url_override, cur.target_fps, cur.blur_faces
                if not (d.frame_w and d.frame_h):
                    d.frame_w, d.frame_h = cur.frame_w, cur.frame_h
            elif not d.url and overrides.get(str(d.remote_id)):   # re-added after a removal: remembered local URL
                d.url_override = overrides[str(d.remote_id)]
            d.url = d.url_override or d.url
            if d.url_override:
                overrides[str(d.remote_id)] = d.url_override
            d.enabled = bool(d.url)
            if not d.enabled:
                needs_url.append({"id": d.id, "name": d.name, "remote_id": d.remote_id})
            if cur is None:
                try:
                    self.manager.add_camera(d)
                    added += 1
                except ValueError:                        # id already taken by a non-managed camera: take it over
                    self.manager.update_camera(d)
                    updated += 1
            elif _signature(d) != _signature(cur):
                self.manager.update_camera(d)
                updated += 1
        wanted = {d.id for d in desired}
        for cid in [c for c in current if c not in wanted]:
            try:
                self.manager.remove_camera(cid)
                removed += 1
            except KeyError:
                pass
        for u in unmanaged:                               # whatever is left by hand must not count twice
            if u.id in donors_used or not u.enabled:
                continue
            self.manager.update_camera(replace(u, enabled=False))
            disabled += 1
        # ---- 3. mapping and store hours, one save
        with cs.lock:
            cfg = cs.config
            cfg.congruence.cameras = {c.id: int(c.remote_id) for c in cfg.cameras if c.managed and c.remote_id}
            cfg.congruence.zones = {f"{c.id}/{z.name}": int(z.remote_id)
                                    for c in cfg.cameras if c.managed for z in c.zones if z.remote_id}
            cfg.congruence.url_overrides = overrides
            st = remote.get("store") or {}
            if st.get("name"):
                cfg.store.name = str(st["name"])[:150]
            for src, dst in (("openTime", "open_time"), ("closeTime", "close_time")):
                v = str(st.get(src) or "")
                if re.fullmatch(r"\d{2}:\d{2}", v):
                    setattr(cfg.store, dst, v)
            cs.save()
        summary = {"added": added, "updated": updated, "removed": removed, "disabled": disabled,
                   "cameras": len(desired), "needs_url": needs_url, "notes": notes, "at": time.time()}
        log.info("Congruence Eye sync: %d camera(s): +%d ~%d -%d, %d hand-made disabled, %d without RTSP URL",
                 len(desired), added, updated, removed, disabled, len(needs_url))
        return summary

    # ---- hot re-derive and the one local edit allowed in managed mode -----------
    def rederive(self, cam_id: str) -> None:
        """The decoded frame size changed: map the server's editor geometry onto it again (no restart)."""
        with self._lock:
            with self.config_store.lock:
                cam = self.config_store.config.camera(cam_id)
                if cam is None or not cam.managed or not (cam.frame_w and cam.frame_h):
                    return
                fw, fh = cam.frame_w, cam.frame_h
                line = cam.line
                rl = cam.remote_line or {}
                if len(rl.get("points") or []) == 2:
                    line = orient_line([stage_to_frame(p, fw, fh) for p in rl["points"]], rl.get("in_direction"))
                zones = [replace(z, points=[stage_to_frame(p, fw, fh) for p in z.remote_points]) if z.remote_points else z
                         for z in cam.zones]
                passby = cam.passby_line
                rp = cam.remote_passby or {}
                if len(rp.get("points") or []) == 2:
                    passby = [stage_to_frame(p, fw, fh) for p in rp["points"]]
                new = replace(cam, line=line, zones=zones, passby_line=passby)
            self.manager.update_camera(new)

    def set_local(self, cam_id: str, url_override: str | None = None, target_fps: float | None = None,
                  blur_faces: bool | None = None) -> CameraConfig:
        """Local settings of a managed camera: an RTSP URL that wins over the server's, fps, face blur.
        url_override: None = unchanged, "" = clear (follow the server), otherwise the new URL."""
        with self._lock:
            with self.config_store.lock:
                cam = self.config_store.config.camera(cam_id)
                if cam is None:
                    raise KeyError(cam_id)
                if not cam.managed:
                    raise ValueError("not a managed camera")
                remote_url = ""
                rem = None
                if self.connector is not None:
                    rem = self.connector.remote or self.connector.remote_cached
                for c in (rem or {}).get("cameras") or []:
                    if isinstance(c, dict) and c.get("id") is not None and int(c["id"]) == cam.remote_id:
                        remote_url = usable_url(c.get("rtspUrl"))
                        break
                ov = cam.url_override if url_override is None else str(url_override).strip()
                if ov:
                    url = ov
                elif remote_url:
                    url = remote_url
                elif url_override == "":
                    url = ""                              # override cleared and the server has no URL either
                else:
                    url = cam.url                         # server unreachable: keep what runs today
                new = replace(cam, url_override=ov, url=url, enabled=bool(url),
                              target_fps=cam.target_fps if target_fps is None else float(target_fps),
                              blur_faces=cam.blur_faces if blur_faces is None else bool(blur_faces))
                mem = dict(self.config_store.config.congruence.url_overrides or {})
                if ov:
                    mem[str(cam.remote_id)] = ov
                else:
                    mem.pop(str(cam.remote_id), None)
                self.config_store.config.congruence.url_overrides = mem
            self.manager.update_camera(new)                # saves the config as well
            return new
