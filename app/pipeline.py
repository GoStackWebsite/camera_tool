"""Per-camera worker: newest frame -> detect/track -> analytics -> (blur) -> annotate -> JPEG."""
from __future__ import annotations

import logging
import math
import threading
from collections import deque
import time

import cv2
import numpy as np

from .analytics import Analytics, Track
from .camera import RTSPCamera
from .config import CameraConfig, ConfigStore, Settings
from .demographics import Demographics
from .detector import PersonTracker

log = logging.getLogger("pipeline")

COLORS = {  # BGR
    "visitor": (245, 245, 127),    # cyan: customer, not engaged yet
    "engaged": (90, 174, 46),      # green: customer engaged (dwell >= engage_seconds)
    "lingering": (0, 165, 255),    # orange
    "staff": (205, 90, 106),       # purple
    "identifying": (190, 190, 190),  # grey: not judged yet (neither staff nor customer)
    "phone": (0, 0, 255),
}
ROLE_COLORS = {  # BGR, zone outlines by role
    "display": (255, 160, 0),
    "entrance": (90, 220, 90),
    "checkout": (255, 230, 0),
    "queue": (0, 200, 255),
    "staff": (255, 120, 160),
}
LINE_COLOR = (255, 220, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _dur(s: float) -> str:
    s = int(s)
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60:02d}s"


def _text_color(bgr: tuple) -> tuple:
    b, g, r = bgr
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255)


def _label(img: np.ndarray, text: str, x: int, y: int, color: tuple, scale: float) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    y = max(y, th + base + 2)
    x = max(0, min(x, img.shape[1] - tw - 6))
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y - base // 2), FONT, scale, _text_color(color), 1, cv2.LINE_AA)


def _label_block(img: np.ndarray, lines: list[str], x: int, y: int, color: tuple, scale: float) -> None:
    """Filled multi-line block whose bottom edge sits at y (just above a person box)."""
    pad = int(6 * scale / 0.6)
    sizes = [cv2.getTextSize(ln, FONT, scale, 1) for ln in lines]
    lh = max(s[0][1] + s[1] for s in sizes) + pad
    bw = max(s[0][0] for s in sizes) + 2 * pad
    bh = lh * len(lines) + pad // 2
    x = max(0, min(x, img.shape[1] - bw))
    y0 = max(0, y - bh)
    cv2.rectangle(img, (x, y0), (x + bw, y0 + bh), color, -1)
    tc = _text_color(color)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (x + pad, y0 + lh * (i + 1) - pad // 2), FONT, scale, tc, 1, cv2.LINE_AA)


def blur_heads(img: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Pixelate the head of every tracked person: the detected face box (padded) plus the top of the
    person box, so nobody is identifiable in streams or snapshots even when the face box lags a frame."""
    h, w = img.shape[:2]
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8:
            continue
        regions = [(x1 + int(bw * 0.12), y1 - int(bh * 0.03), x2 - int(bw * 0.12), y1 + max(12, int(bh * 0.24)))]
        if t.face_box:
            fx1, fy1, fx2, fy2 = t.face_box
            fw, fh = fx2 - fx1, fy2 - fy1
            regions.append((fx1 - int(fw * 0.4), fy1 - int(fh * 0.4), fx2 + int(fw * 0.4), fy2 + int(fh * 0.4)))
        for rx1, ry1, rx2, ry2 in regions:
            rx1, ry1, rx2, ry2 = max(0, rx1), max(0, ry1), min(w, rx2), min(h, ry2)
            rw, rh = rx2 - rx1, ry2 - ry1
            if rw < 4 or rh < 4:
                continue
            block = max(3, int(max(rw, rh) / 6))     # ~6 coarse blocks across the head
            small = cv2.resize(img[ry1:ry2, rx1:rx2], (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_LINEAR)
            img[ry1:ry2, rx1:rx2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    return img


def draw_overlay(frame: np.ndarray, tracks: list[Track], cam: CameraConfig, stats: dict,
                 infer_fps: float, cam_fps: float, now: float, engage_seconds: float = 30.0) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    scale = max(0.45, min(1.0, h / 1080 * 0.75))
    thick = max(1, round(h / 540))

    # zones: translucent fill + outline + live count, coloured by role
    if cam.zones:
        ov = img.copy()
        polys = [np.array([[int(x * w), int(y * h)] for x, y in z.points], np.int32) for z in cam.zones]
        for z, pts in zip(cam.zones, polys):
            cv2.fillPoly(ov, [pts], ROLE_COLORS.get(z.role, ROLE_COLORS["display"]))
        img = cv2.addWeighted(ov, 0.16, img, 0.84, 0)
        zstats = {z["name"]: z for z in stats.get("zones", [])}
        for z, pts in zip(cam.zones, polys):
            color = ROLE_COLORS.get(z.role, ROLE_COLORS["display"])
            zs = zstats.get(z.name, {})
            cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)
            if z.role == "queue":
                txt = f"{z.name}: {zs.get('count', 0)} waiting"
                if zs.get("max_wait"):
                    txt += f" - {_dur(zs['max_wait'])}"
            else:
                txt = f"{z.name} ({z.role}): {zs.get('count', 0)}"
            if zs.get("staff"):
                txt += f" +{zs['staff']} staff"
            if zs.get("queue_breach"):
                txt += "  BREACH - open another till"
            if zs.get("unattended"):
                txt += "  UNATTENDED!"
            _label(img, txt, int(pts[0][0]), int(pts[0][1]) - 4, color, scale)

    # counting line with IN / OUT sides
    if cam.passby_line:
        (px, py), (qx, qy) = cam.passby_line
        cv2.line(img, (int(px * w), int(py * h)), (int(qx * w), int(qy * h)), (255, 200, 0), 2)
        cv2.putText(img, f"pass-by {stats.get('passby', 0)}", (int(px * w) + 4, int(py * h) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    if cam.line:
        (ax, ay), (bx, by) = cam.line
        a = (int(ax * w), int(ay * h))
        b = (int(bx * w), int(by * h))
        cv2.line(img, a, b, LINE_COLOR, thick + 1, cv2.LINE_AA)
        mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / n, dx / n          # unit normal pointing to the + side (IN)
        off = int(30 * scale / 0.75)
        _label(img, "IN", int(mx + nx * off) - 10, int(my + ny * off) + 6, LINE_COLOR, scale)
        _label(img, "OUT", int(mx - nx * off) - 14, int(my - ny * off) + 6, LINE_COLOR, scale)

    # tracked people: TangoEye-style label block above each box
    engaged_now = 0
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        dwell = t.dwell(now)
        entry = time.strftime("%I:%M %p", time.localtime(t.first_seen)).lstrip("0").lower()
        # the same person carries one id on every camera of the store; someone only this (second) camera sees keeps
        # its own number with the camera's letter, so a bare number always means the overview camera's person
        pid = getattr(t, "store_id", None) or (f"{stats['id_tag']}{t.id}" if stats.get("id_tag") else t.id)
        if t.is_staff:
            color = COLORS["staff"]
            lines = [f"Staff ID: {pid}", f"On floor: {_dur(dwell)}" + (f" - {t.staff_reason}" if t.staff_reason else "")]
        elif not getattr(t, "judged", True):
            color = COLORS["identifying"]       # just came into view: staff or customer is decided within a second or two
            lines = [f"ID: {pid}", "identifying..."]
        else:
            engaged = dwell >= engage_seconds
            engaged_now += int(engaged)
            color = COLORS["lingering"] if t.loitering else (COLORS["engaged"] if engaged else COLORS["visitor"])
            lines = [f"Visitor ID: {pid}", f"Entry time: {entry}",
                     f"{_dur(dwell)} - {t.activity}" + (" - engaged" if engaged else "") + (" - lingering" if t.loitering else "")]
            if t.gender:
                lines.append(f"{'Man' if t.gender == 'M' else 'Woman'} - {t.age_band}")
        if t.phone_since:
            lines.append("Using phone")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
        _label_block(img, lines, x1, y1 - 2, color, scale * 0.85)
        cv2.circle(img, ((x1 + x2) // 2, y2), thick + 2, color, -1, cv2.LINE_AA)
        if t.phone_box:
            px1, py1, px2, py2 = t.phone_box
            cv2.rectangle(img, (px1, py1), (px2, py2), COLORS["phone"], thick, cv2.LINE_AA)
        trail = [(int(fx * w), int(fy * h)) for _, fx, fy in list(t.history)[-40:]]
        if len(trail) > 1:
            cv2.polylines(img, [np.array(trail, np.int32)], False, color, 1, cv2.LINE_AA)

    # translucent status bar
    bar_h = int(36 * scale / 0.75)
    img[:bar_h] = (img[:bar_h] * 0.35).astype(np.uint8)
    text = (f"{cam.name}   Customers {stats.get('occupancy', 0)}   Engaged {engaged_now}   Staff {stats.get('staff', 0)}"
            f"   In {stats.get('entries', 0)} ({stats.get('groups_in', 0)} groups)   Out {stats.get('exits', 0)}"
            f"   Cam {cam_fps:.0f} fps | AI {infer_fps:.1f} fps   {time.strftime('%H:%M:%S', time.localtime(now))}")
    cv2.putText(img, text, (10, int(bar_h * 0.68)), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def apply_heatmap(img: np.ndarray, heat: np.ndarray) -> np.ndarray:
    peak = float(heat.max())
    if peak <= 0:
        return img
    h, w = img.shape[:2]
    norm = np.sqrt(heat / peak)
    hm = cv2.resize((norm * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    hm = cv2.GaussianBlur(hm, (0, 0), max(3.0, w / 120))
    color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    alpha = (hm.astype(np.float32) / 255.0 * 0.65)[..., None]
    out = img.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    return out.astype(np.uint8)


def letterbox(img: np.ndarray, aspect: float) -> np.ndarray:
    """Pad (never crop) with black bars so the image gets the given width/height ratio, content centred.
    Used for snapshots sent to Congruence Eye, whose zone editor shows camera images on a 16:9 stage."""
    h, w = img.shape[:2]
    cur = w / h
    if abs(cur - aspect) < 1e-3:
        return img
    if cur < aspect:                      # too narrow: bars left and right
        pad = int(round(h * aspect)) - w
        return cv2.copyMakeBorder(img, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    pad = int(round(w / aspect)) - h      # too wide: bars top and bottom
    return cv2.copyMakeBorder(img, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def placeholder_frame(message: str, size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    img = np.full((size[1], size[0], 3), 22, np.uint8)
    cv2.putText(img, "NO SIGNAL", (size[0] // 2 - 150, size[1] // 2 - 10), FONT, 1.6, (90, 90, 90), 3, cv2.LINE_AA)
    cv2.putText(img, message, (40, size[1] // 2 + 50), FONT, 0.7, (170, 170, 170), 1, cv2.LINE_AA)
    return img


RING_FPS = 4            # evidence-clip buffer: frames per second kept in memory
RING_FRAMES = 100       # ~25 s at RING_FPS
RING_WIDTH = 640


class Pipeline(threading.Thread):
    def __init__(self, camera: RTSPCamera, tracker: PersonTracker, analytics: Analytics,
                 settings: Settings, target_fps: float = 0.0, demographics: Demographics | None = None,
                 config_store: ConfigStore | None = None):
        super().__init__(name=f"pipeline-{analytics.camera_id}", daemon=True)
        self.camera = camera
        self.tracker = tracker
        self.analytics = analytics
        self.settings = settings
        self.demographics = demographics
        self.config_store = config_store
        self._frame_no = 0
        self._face_job: tuple | None = None
        self._face_cond = threading.Condition()
        self.face_ms = 0.0
        self.target_dt = 1.0 / max(target_fps or settings.target_fps, 1.0)
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._seq = 0
        self._raw: np.ndarray | None = None
        self._annot: np.ndarray | None = None
        self._ring: deque = deque(maxlen=RING_FRAMES)   # (ts, small annotated JPEG): the evidence-clip buffer
        self._ring_ts = 0.0
        self._jpeg_cache: dict[tuple, tuple[int, bytes]] = {}
        self._placeholder: tuple[str, bytes] | None = None
        self.infer_fps = 0.0
        self.infer_ms = 0.0
        self.last_stats: dict = {}
        self.frame_size = (0, 0)
        self._last_seq = -1

    def stop(self) -> None:
        self._stop.set()

    def replace_camera(self, camera: RTSPCamera) -> None:
        """Swap the RTSP reader (stall watchdog) without touching the tracker, analytics or counters."""
        with self._cond:
            self.camera = camera
            self._last_seq = -1

    # ---- thread body ------------------------------------------------------
    def run(self) -> None:
        if self.demographics is not None:
            threading.Thread(target=self._face_loop, name=f"faces-{self.analytics.camera_id}", daemon=True).start()
        self._last_seq = -1
        tick = time.time()
        idle_ts = 0.0
        n = 0
        ms_acc = 0.0
        while not self._stop.is_set():
            item = self.camera.latest()
            if item is None or item[0] == self._last_seq:
                now = time.time()
                if not self.camera.connected and now - idle_ts >= 1.0:
                    # stalled / offline: keep the analytics clock running so lost tracks close as
                    # visits, occupancy drops to zero and the series / minute stats stay continuous
                    idle_ts = tick = now
                    n, ms_acc = 0, 0.0
                    self.infer_fps = 0.0
                    self._idle_update(now)
                time.sleep(0.004)
                continue
            seq, frame = item
            self._last_seq = seq
            t0 = time.time()
            try:
                dets = self.tracker.track(frame)
            except Exception:
                log.exception("inference failed on %s", self.analytics.camera_id)
                dets = []
            t1 = time.time()
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            self._frame_no += 1
            self.analytics.update(dets, w, h, t0, frame=frame)
            self._queue_faces(frame, dets)
            stats = self.analytics.snapshot(t0)
            tracks = self.analytics.active_tracks(t0)
            cam = self.analytics.cam
            out = blur_heads(frame.copy(), tracks) if cam.blur_faces else frame
            annotated = draw_overlay(out, tracks, cam, stats, self.infer_fps, self.camera.fps, t0,
                                     self.analytics.thr.engage_seconds)
            with self._cond:
                self._raw = out
                self._annot = annotated
                self._seq += 1
                self._jpeg_cache.clear()
                self.last_stats = stats
                self._cond.notify_all()
            if t0 - self._ring_ts >= 1.0 / RING_FPS:
                self._ring_ts = t0
                try:
                    self._ring.append((t0, self._encode(annotated, RING_WIDTH)))
                except Exception:
                    pass
            n += 1
            ms_acc += (t1 - t0) * 1000
            if t0 - tick >= 2.0:
                self.infer_fps = n / (t0 - tick)
                self.infer_ms = ms_acc / n
                n = 0
                ms_acc = 0.0
                tick = t0
            time.sleep(max(0.0, self.target_dt - (time.time() - t0)))

    def _idle_update(self, now: float) -> None:
        """Analytics step with no detections (camera offline): expires tracks and refreshes stats."""
        w, h = self.frame_size
        if not w or not h:
            return
        self.analytics.update([], w, h, now)
        with self._cond:
            self.last_stats = self.analytics.snapshot(now)

    def _queue_faces(self, frame: np.ndarray, dets) -> None:
        """Hand the newest frame to the face worker (drops the job if the worker is still busy)."""
        if self.demographics is None or self.config_store is None:
            return
        cfg = self.config_store.config.demographics
        if not cfg.enabled or self._frame_no % max(1, cfg.sample_every):
            return
        boxes = {}
        uniforms = bool(self.config_store.config.staff.uniforms)
        now = time.time()
        for d in dets:
            if not d.is_person:
                continue
            t = self.analytics.tracks.get(d.track_id)
            # sample until demographics are settled; keep refreshing faces slowly for uniform anchoring
            if t is None or t.face_samples < cfg.max_samples or (uniforms and now - t.face_ts > 1.5):
                boxes[d.track_id] = (d.x1, d.y1, d.x2, d.y2)
        if not boxes:
            return
        with self._face_cond:
            self._face_job = (frame, boxes, cfg.min_face)
            self._face_cond.notify()

    def _face_loop(self) -> None:
        while not self._stop.is_set():
            with self._face_cond:
                self._face_cond.wait_for(lambda: self._face_job is not None or self._stop.is_set(), 1.0)
                job, self._face_job = self._face_job, None
            if job is None:
                continue
            t0 = time.time()
            try:
                faces = self.demographics.detect(*job)
                if faces:
                    self.analytics.apply_faces(faces)
            except Exception:
                log.exception("face analysis failed")
            self.face_ms = 0.8 * self.face_ms + 0.2 * (time.time() - t0) * 1000

    # ---- consumers --------------------------------------------------------
    def wait_for_frame(self, last_seq: int, timeout: float = 1.0) -> int:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout)
            return self._seq

    def recent_frames(self, t0: float, t1: float) -> list[tuple[float, bytes]]:
        """Buffered annotated frames with t0 <= ts <= t1 (for evidence clips)."""
        return [(ts, j) for ts, j in list(self._ring) if t0 <= ts <= t1]

    def jpeg(self, mode: str = "annotated", heat: bool = False, width: int | None = None,
             aspect: float | None = None) -> bytes | None:
        """Latest frame as JPEG. `aspect` pads the image with black bars to that width/height ratio."""
        key = (mode, heat, width, aspect)
        with self._cond:
            cached = self._jpeg_cache.get(key)
            if cached and cached[0] == self._seq:
                return cached[1]
            img = self._annot if mode == "annotated" else self._raw
            if img is None:
                return None
            seq = self._seq
            heat_grid = self.analytics.heat.copy() if heat else None
        if heat_grid is not None:
            img = apply_heatmap(img, heat_grid)
        data = self._encode(img, width, aspect)
        with self._cond:
            self._jpeg_cache[key] = (seq, data)
        return data

    def placeholder_jpeg(self) -> bytes:
        msg = self.camera.last_error or ("Connecting to camera..." if not self.camera.connected
                                         else "Starting analysis...")
        if self._placeholder is None or self._placeholder[0] != msg:
            self._placeholder = (msg, self._encode(placeholder_frame(msg), None))
        return self._placeholder[1]

    def stats(self) -> dict:
        s = {"camera_id": self.analytics.camera_id, "name": self.analytics.cam.name, "occupancy": 0, "staff": 0,
             "entries": 0, "exits": 0, "groups_in": 0, "unique": 0, "loitering": 0, "phones": 0, "avg_dwell": 0.0,
             "cfg_rev": self.analytics.cfg_rev, "zones": [], "tracks": [], "events": [], "series": [],
             "has_line": bool(self.analytics.cam.line), "has_staff_zone": self.analytics.has_staff_zone}
        s.update(self.last_stats)
        s["camera"] = {
            "connected": self.camera.connected,
            "width": self.camera.width,
            "height": self.camera.height,
            "fps": round(self.camera.fps, 1),
            "error": self.camera.last_error,
            "stalls": self.camera.stalls,
            "reconnects": self.camera.reconnects,
        }
        s["inference"] = {
            "fps": round(self.infer_fps, 1),
            "ms": round(self.infer_ms, 1),
            "face_ms": round(self.face_ms, 1),
            "model": self.tracker.model_name,
            "device": self.tracker.device_label,
        }
        return s

    def _encode(self, img: np.ndarray, width: int | None, aspect: float | None = None) -> bytes:
        sw = width or self.settings.stream_width
        if sw and img.shape[1] > sw:
            scale = sw / img.shape[1]
            img = cv2.resize(img, (sw, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        if aspect:
            img = letterbox(img, aspect)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])
        return buf.tobytes() if ok else b""
