"""Per-camera people analytics: occupancy, dwell, activity, staff, line counting with groups,
zones with journeys, heatmap, and SOP checks (queue, unattended customer, phone usage)."""
from __future__ import annotations

import math
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .alerts import AlertManager
from .config import CameraConfig, StaffConfig, Thresholds, ZoneConfig, age_band
from .detector import Detection
from .store import Store

log = logging.getLogger("analytics")

LOST_TIMEOUT = 1.5           # seconds without a detection before a track is considered gone
SPEED_WINDOW = 0.5           # seconds of history used to estimate speed
HEAT_H, HEAT_W = 36, 64      # heatmap grid resolution
HEAT_DECAY = 0.99998         # per-frame decay (half-life ~38 min at 15 fps)
PHONE_GAP = 2.0              # seconds without a phone detection before "holding phone" resets
ACTIVITIES = ("stationary", "walking", "running")


# ---------------------------------------------------------------- geometry

def point_in_polygon(x: float, y: float, pts: list) -> bool:
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def line_side(x: float, y: float, line: list) -> int:
    (ax, ay), (bx, by) = line
    cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
    return 1 if cross > 0 else (-1 if cross < 0 else 0)



def hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


BRAND_MIN_PX = 40            # logo pixels of one brand colour on the chest (summed over samples) before the brand is known
MODEL_MIN_SAMPLES = 6        # learned classifier: samples (0.5 s apart) before a person can become staff
MODEL_STAFF_PROB = 0.70      # mean P(staff) at or above this -> staff
MODEL_CUSTOMER_PROB = 0.25   # a "uniform" staff whose mean drops to this (10+ samples) becomes a customer again


def logo_mask(hsv: np.ndarray, hues: list, tol: int = 4) -> np.ndarray:
    """Pixels (rows of an N x 3 OpenCV HSV array) that are bright, saturated and close to one of the hues."""
    h = hsv[:, 0].astype(np.int16)
    hit = np.zeros(len(hsv), bool)
    for lh in hues:
        d = np.abs(h - int(lh))
        hit |= np.minimum(d, 180 - d) <= tol
    return hit & (hsv[:, 1] > 120) & (hsv[:, 2] > 120)


def has_logo(u: dict) -> bool:
    return bool(u.get("logo_hues")) or bool(u.get("logo_white"))


def logo_fraction(hsv: np.ndarray, u: dict) -> float:
    """Share (0-1) of the chest crop covered by the uniform's print: pixels of the logo hues ("logo_hues",
    OpenCV hue 0-179) and / or, with "logo_white", white print on the dark garment. 0 when the crop is not
    mostly dark cloth (decoration or goods held in front of the chest would otherwise count)."""
    if len(hsv) == 0 or not has_logo(u):
        return 0.0
    n = len(hsv)
    dark = np.count_nonzero((hsv[:, 2] < 100) & (hsv[:, 1] < 110)) / n
    if dark < 0.45:
        return 0.0
    hues = u.get("logo_hues") or []
    frac = float(np.count_nonzero(logo_mask(hsv, hues))) / n if hues else 0.0
    if u.get("logo_white"):
        white = float(np.count_nonzero((hsv[:, 1] < 60) & (hsv[:, 2] > 180))) / n
        lo = float(u.get("logo_white_min_pct", 5)) / 100.0
        if lo <= white <= 0.30 and dark >= 0.55:      # a print, not a striped or mostly white garment
            frac = max(frac, white)
    return frac


def logo_present(hsv: np.ndarray, u: dict) -> bool:
    """The print must cover at least "logo_min_pct" percent of the chest crop; uniforms without one always pass."""
    if not has_logo(u):
        return True
    return logo_fraction(hsv, u) >= float(u.get("logo_min_pct", 0.4)) / 100.0


def match_uniform(hsv: tuple, u: dict) -> bool:
    """Does a torso colour sample (OpenCV HSV: h 0-179, s/v 0-255) look like this uniform?"""
    h, s, v = hsv
    uh, us, uv = u["h"], u["s"], u["v"]
    if uv < 70:                                   # dark uniform (black / navy): hue is unreliable
        # with a logo required, the colour test can be looser (black cloth under bright shop lights reads 90-110)
        return (v < 125 and s < 120) if has_logo(u) else (v < 80 and s < 110)
    if us < 55 and uv > 160:                      # white uniform: nearly no saturation, bright even in shadow
        return s < 35 and v > 115
    return hue_distance(h, uh) <= 10 and abs(s - us) <= 60 and abs(v - uv) <= 70


# ---------------------------------------------------------------- tracks

@dataclass
class Track:
    id: int
    first_seen: float
    last_seen: float
    bbox: tuple
    conf: float
    history: deque = field(default_factory=lambda: deque(maxlen=150))  # (t, fx, fy) foot point
    speed: float = 0.0
    activity: str = "stationary"
    zones: dict = field(default_factory=dict)       # zone name -> entered_at
    zone_seq: list = field(default_factory=list)    # ordered zones visited
    side: int = 0                                   # count line: +1 IN side, -1 OUT side, 0 unknown
    side_pt: tuple | None = None                    # foot point of the last clear observation of that side
    pending_side: int = 0                           # opposite side seen while a crossing is being confirmed
    pending_n: int = 0
    last_cross_ts: float = 0.0
    pb_side: int = 0                                # pass-by line: side of the last clear observation
    pb_pt: tuple | None = None
    pb_pending: int = 0
    pb_n: int = 0
    pb_counted: bool = False                        # a track is one passer-by however often it wanders over the line
    brand_px: dict = field(default_factory=dict)    # staff: logo pixels seen per brand colour
    brand: str | None = None                        # staff: brand of the uniform (from the logo colour)
    served: dict = field(default_factory=dict)      # customer: seconds attended per staff brand (resolved at visit end)
    served_by: dict = field(default_factory=dict)   # customer: seconds attended per staff track id
    phone_near: float = 0.0                         # staff phone episode: seconds with a customer within attend_distance
    phone_total: float = 0.0
    loitering: bool = False
    run_alerted: bool = False
    path_len: float = 0.0
    is_staff: bool = False
    staff_zone_time: float = 0.0
    group_id: str | None = None
    crossed_in: bool = False
    phone_since: float | None = None
    phone_last: float = 0.0
    phone_alerted: bool = False
    phone_box: tuple | None = None
    gender_votes: list = field(default_factory=lambda: [0.0, 0.0])   # [female, male], weighted by face size
    age_samples: list = field(default_factory=list)
    face_samples: int = 0
    gender: str | None = None
    age: int | None = None
    face_box: tuple | None = None
    face_ts: float = 0.0                              # when the face box was last refreshed
    torso: tuple | None = None                       # latest torso colour sample (h, s, v)
    torso_ts: float = 0.0
    torso_anchored: bool = False                     # sample taken below a detected face
    uniform_votes: deque = field(default_factory=lambda: deque(maxlen=20))
    model_votes: deque = field(default_factory=lambda: deque(maxlen=20))   # learned classifier: P(staff) per sample
    manual_staff: bool | None = None                 # True/False once a person tagged it by hand
    staff_reason: str = ""
    logo_frac: float = 0.0                           # last measured logo coverage of the chest crop (uniform logos)

    @property
    def age_band(self) -> str | None:
        return age_band(self.age)

    def dwell(self, now: float) -> float:
        return now - self.first_seen

    @property
    def foot(self) -> tuple[float, float]:
        return (self.history[-1][1], self.history[-1][2]) if self.history else (0.0, 0.0)


def _fresh_minute(bucket: int) -> dict:
    return {"ts": bucket, "sum": 0.0, "n": 0, "max": 0, "entries": 0, "exits": 0, "new": 0, "staff_sum": 0.0, "queues": {}}


class Analytics:
    def __init__(self, cam: CameraConfig, thr: Thresholds, store: Store | None, alerts: AlertManager | None,
                 staff_cfg: StaffConfig | None = None, staff_model=None):
        self.cam = cam
        self.thr = thr
        self.staff_cfg = staff_cfg or StaffConfig()
        self.staff_model = staff_model            # app.staffcls.StaffClassifier (learned uniform), optional
        self.store = store
        self.alerts = alerts
        self._lock = threading.RLock()
        self.tracks: dict[int, Track] = {}
        self.heat = np.zeros((HEAT_H, HEAT_W), np.float32)
        self.minutes: deque = deque(maxlen=24 * 60)
        self._minute = _fresh_minute(int(time.time() // 60) * 60)
        self._last_series_ts = 0.0
        self._recent_ins: deque = deque(maxlen=50)      # (ts, fx, fy, group_id)
        self._zone_state: dict[str, dict] = {}
        self.passby = 0                               # people who crossed the pass-by (street) line today
        self.served_today: dict[str, int] = {}        # walk-ins attributed per brand today
        self._staff_brand: dict[int, str] = {}        # staff track id -> brand once the logo colour settled
        self._prev_update = 0.0
        self.cfg_rev = 0
        self.ce_heat = np.zeros((18, 32), np.int32)   # per-window foot-point counts for Congruence Eye
        self.emitter = None                           # Congruence Eye connector (attached by the connector)
        self.reset()

    # ---- public ----------------------------------------------------------
    @property
    def camera_id(self) -> str:
        return self.cam.id

    def reset(self) -> None:
        with self._lock:
            self.session_start = time.time()
            self.events: deque = deque(maxlen=500)
            self.entries = 0
            self.exits = 0
            self.passby = 0
            self.groups_in = 0
            self.unique_ids: set[int] = set(self.tracks.keys())
            self.peak = 0
            self.completed_dwells: deque = deque(maxlen=2000)
            self.zone_stats: dict[str, dict] = {}
            self.occupancy_series: deque = deque(maxlen=600)   # (ts, customers, staff)
            self.heat[:] = 0
            for t in self.tracks.values():
                t.loitering = False
                t.run_alerted = False

    def apply_camera_config(self, cam: CameraConfig) -> None:
        with self._lock:
            self.cam = cam
            self.cfg_rev += 1
            names = {z.name for z in cam.zones}
            now = time.time()
            for t in self.tracks.values():
                for zname in [k for k in t.zones if k not in names]:   # zone removed or renamed: close its visit
                    self._emit_ce("ZONE_EXIT", t, now, zone=zname, dwell=now - t.zones[zname])
                    self._leave_zone(t, zname, now)
                t.pending_side = t.pending_n = 0
                if t.history and cam.line:
                    _, fx, fy = t.history[-1]
                    t.side = self._observe_line(fx, fy)
                    t.side_pt = (fx, fy) if t.side else None
                else:
                    t.side = 0
                    t.side_pt = None
            for name in [n for n in self._zone_state if n not in names]:
                del self._zone_state[name]
                if self.alerts:
                    self.alerts.resolve(f"queue:{cam.id}:{name}")
                    self.alerts.resolve(f"unattended:{cam.id}:{name}")

    def apply_thresholds(self, thr: Thresholds) -> None:
        with self._lock:
            self.thr = thr
            self.cfg_rev += 1

    def apply_staff(self, cfg: StaffConfig) -> None:
        with self._lock:
            self.staff_cfg = cfg
            self.cfg_rev += 1
            for t in self.tracks.values():    # re-evaluate uniform decisions under the new settings
                if t.staff_reason == "uniform" and t.manual_staff is None:
                    t.is_staff = False
                    t.staff_reason = ""
                t.uniform_votes.clear()

    def set_staff(self, track_id: int, flag: bool) -> dict | None:
        """Manual tag from the dashboard. Returns the track's torso colour so the uniform can be learned."""
        with self._lock:
            t = self.tracks.get(track_id)
            if t is None:
                return None
            t.manual_staff = flag
            if flag and not t.is_staff:
                self._set_staff(t, time.time(), "manual")
            elif not flag:
                t.is_staff = False
                t.staff_reason = ""
            return {"torso": t.torso if t.torso_anchored else None, "anchored": t.torso_anchored}

    @property
    def staff_detection_active(self) -> bool:
        return self.staff_model is not None or self.has_staff_zone or bool(self.staff_cfg.uniforms) or self.staff_cfg.after_minutes > 0

    def active_tracks(self, now: float) -> list[Track]:
        with self._lock:
            return [t for t in self.tracks.values() if now - t.last_seen <= LOST_TIMEOUT]

    @property
    def has_staff_zone(self) -> bool:
        return any(z.role == "staff" for z in self.cam.zones)

    # ---- frame update ----------------------------------------------------
    def update(self, dets: list[Detection], width: int, height: int, now: float, faces: list | None = None,
               frame: np.ndarray | None = None) -> None:
        aspect = width / height if height else 1.0
        with self._lock:
            seen: set[int] = set()
            persons = [d for d in dets if d.is_person]
            phones = [d for d in dets if not d.is_person]
            for d in persons:
                fx = ((d.x1 + d.x2) / 2) / width
                fy = d.y2 / height
                t = self.tracks.get(d.track_id)
                if t is None:
                    t = Track(d.track_id, now, now, (d.x1, d.y1, d.x2, d.y2), d.conf)
                    self.tracks[d.track_id] = t
                    self.unique_ids.add(d.track_id)
                    self._minute["new"] += 1
                    t.side = self._observe_line(fx, fy) if self.cam.line else 0
                    t.side_pt = (fx, fy) if t.side else None
                    self._event(now, "enter", d.track_id, f"Person #{d.track_id} appeared")
                    dt = 0.0
                else:
                    _, px, py = t.history[-1]
                    t.path_len += math.hypot((fx - px) * aspect, fy - py)
                    dt = now - t.last_seen
                t.last_seen = now
                t.bbox = (d.x1, d.y1, d.x2, d.y2)
                t.conf = d.conf
                t.phone_box = None
                t.history.append((now, fx, fy))
                seen.add(d.track_id)
                if frame is not None:
                    self._sample_torso(t, frame, now)
                self._update_motion(t, now, aspect)
                self._update_line(t, now, fx, fy)
                if self.cam.passby_line:
                    self._update_passby(t, now, fx, fy)
                self._update_zones(t, now, fx, fy, dt)
                self._heat_add(fx, fy)
                self.ce_heat[min(17, max(0, int(fy * 18))), min(31, max(0, int(fx * 32)))] += 1

            self._update_phones(phones, now)
            if faces:
                self._update_faces(faces, now)

            # expire lost tracks -> visit records
            for tid, t in list(self.tracks.items()):
                if tid in seen or now - t.last_seen <= LOST_TIMEOUT:
                    continue
                for zname, entered in list(t.zones.items()):     # lost inside a zone: close the zone visit
                    self._emit_ce("ZONE_EXIT", t, t.last_seen, zone=zname, dwell=t.last_seen - entered)
                self._close_track(t, now)
                del self.tracks[tid]

            active = [t for t in self.tracks.values() if now - t.last_seen <= LOST_TIMEOUT]
            customers = [t for t in active if not t.is_staff]
            staff = [t for t in active if t.is_staff]
            fdt = min(1.0, now - self._prev_update) if self._prev_update else 0.0
            self._prev_update = now
            self._attend(customers, staff, fdt)
            occ = len(customers)
            self.peak = max(self.peak, occ)
            self._check_crowd(occ)
            self._check_zones(customers, staff, now)

            self.heat *= HEAT_DECAY
            if now - self._last_series_ts >= 1.0:
                self.occupancy_series.append((round(now, 1), occ, len(staff)))
                self._last_series_ts = now
            self._minute_tick(now, occ, len(staff))

    # ---- snapshot --------------------------------------------------------
    def snapshot(self, now: float) -> dict:
        with self._lock:
            active = [t for t in self.tracks.values() if now - t.last_seen <= LOST_TIMEOUT]
            customers = [t for t in active if not t.is_staff]
            act = {a: 0 for a in ACTIVITIES}
            for t in customers:
                act[t.activity] += 1
            dwells = [t.dwell(now) for t in customers]
            zones = []
            for z in self.cam.zones:
                zs = self._zone(z.name)
                st = self._zone_state.get(z.name, {})
                waits = [now - t.zones[z.name] for t in customers if z.name in t.zones]
                zones.append({
                    "name": z.name, "role": z.role,
                    "count": len(waits),
                    "avg_wait": round(sum(waits) / len(waits), 1) if waits else 0.0,
                    "max_wait": round(max(waits), 1) if waits else 0.0,
                    "breach_for": round(now - st["breach_since"], 1) if st.get("breach_since") else 0.0,
                    "staff": sum(1 for t in active if t.is_staff and z.name in t.zones),
                    "visits": zs["visits"],
                    "avg_dwell": round(zs["dwell_total"] / zs["visits"], 1) if zs["visits"] else 0.0,
                    "queue_breach": bool(st.get("breach_since")),
                    "unattended": bool(st.get("unattended")),
                })
            return {
                "ts": now,
                "camera_id": self.cam.id,
                "name": self.cam.name,
                "session_start": self.session_start,
                "cfg_rev": self.cfg_rev,
                "occupancy": len(customers),
                "staff": len(active) - len(customers),
                "peak": self.peak,
                "unique": len(self.unique_ids),
                "entries": self.entries,
                "exits": self.exits,
                "passby": self.passby,
                "served_today": dict(self.served_today),
                "staff_brands": {b: sum(1 for t in active if t.is_staff and t.brand == b) for b in {t.brand for t in active if t.is_staff and t.brand}},
                "capture_rate_pct": round(100.0 * self.entries / self.passby, 1) if self.passby else None,
                "has_passby": bool(self.cam.passby_line),
                "groups_in": self.groups_in,
                "loitering": sum(1 for t in customers if t.loitering),
                "phones": sum(1 for t in active if t.phone_since),
                "avg_dwell": round(sum(dwells) / len(dwells), 1) if dwells else 0.0,
                "avg_visit": round(sum(self.completed_dwells) / len(self.completed_dwells), 1)
                if self.completed_dwells else 0.0,
                "activity": act,
                "zones": zones,
                "has_line": bool(self.cam.line),
                "has_staff_zone": self.has_staff_zone,
                "staff_detection": self.staff_detection_active,
                "tracks": [
                    {
                        "id": t.id, "activity": t.activity, "speed": round(t.speed, 3),
                        "dwell": round(t.dwell(now), 1), "zones": sorted(t.zones), "loitering": t.loitering,
                        "staff": t.is_staff, "phone": bool(t.phone_since), "group": t.group_id,
                        "gender": t.gender, "age": t.age, "age_band": t.age_band,
                        "staff_reason": t.staff_reason, "torso": t.torso, "manual": t.manual_staff,
                        "conf": round(t.conf, 2),
                        "anchored": t.torso_anchored, "votes": [int(sum(t.uniform_votes)), len(t.uniform_votes)],
                        "logo_pct": round(t.logo_frac * 100, 2),
                        "model_pct": round(100 * sum(t.model_votes) / len(t.model_votes), 1) if t.model_votes else None,
                        "brand": t.brand, "served": self._served_live(t) or None,
                        "phone_demo_pct": round(100 * t.phone_near / t.phone_total) if t.phone_total else None,
                        "foot": [round(t.foot[0], 3), round(t.foot[1], 3)],
                    }
                    for t in sorted(active, key=lambda t: t.id)
                ],
                "events": list(self.events)[-40:],
                "series": list(self.occupancy_series),
            }

    # ---- internals -------------------------------------------------------
    def _zone(self, name: str) -> dict:
        return self.zone_stats.setdefault(name, {"visits": 0, "dwell_total": 0.0})

    def _event(self, now: float, etype: str, track_id: int | None, message: str, zone: str | None = None) -> None:
        ev = {"ts": round(now, 2), "camera_id": self.cam.id, "type": etype, "track_id": track_id,
              "zone": zone, "message": message}
        self.events.append(ev)
        if self.store:
            self.store.add_event(ev)

    def _emit_ce(self, etype: str, t: Track, now: float, zone: str | None = None, dwell: float | None = None) -> None:
        """Forward a visitor event (ENTRY / EXIT / ZONE_ENTER / ZONE_EXIT) to the Congruence Eye connector."""
        em = self.emitter
        if em is None:
            return
        try:
            em.visitor_event(self.cam.id, etype, t, now, zone, dwell)
        except Exception:
            log.exception("Congruence Eye connector rejected %s", etype)

    def _update_motion(self, t: Track, now: float, aspect: float) -> None:
        ref = None
        for s in reversed(t.history):
            if now - s[0] >= SPEED_WINDOW:
                ref = s
                break
        if ref is not None:
            dt = now - ref[0]
            _, fx, fy = t.history[-1]
            inst = math.hypot((fx - ref[1]) * aspect, fy - ref[2]) / dt
            t.speed = 0.7 * t.speed + 0.3 * inst if t.speed else inst
        if t.speed < self.thr.walk_speed:
            act = "stationary"
        elif t.speed < self.thr.run_speed:
            act = "walking"
        else:
            act = "running"
        if act != t.activity:
            t.activity = act
            if act == "running" and not t.run_alerted:
                t.run_alerted = True
                self._event(now, "running", t.id, f"Person #{t.id} is running")
        if (not t.is_staff and t.manual_staff is None and self.staff_cfg.after_minutes > 0
                and t.dwell(now) >= self.staff_cfg.after_minutes * 60):
            self._set_staff(t, now, "long presence")
        if not t.loitering and not t.is_staff and t.dwell(now) >= self.thr.loiter_seconds:
            t.loitering = True
            self._event(now, "loiter", t.id, f"Person #{t.id} lingering for {t.dwell(now):.0f}s")
            if self.alerts:
                self.alerts.fire(f"loiter:{self.cam.id}:{t.id}", "loiter",
                                 f"{self.cam.name}: visitor #{t.id} lingering for {t.dwell(now) / 60:.0f} min",
                                 self.cam.id, None, t.id)

    def _vote_brand(self, t: Track, frame: np.ndarray, anchored: bool) -> None:
        """Count logo-coloured pixels on the chest for each configured brand; the dominant colour names the brand."""
        x1, y1, x2, y2 = t.bbox
        bw, bh = x2 - x1, y2 - y1
        if anchored and t.face_box:
            fx1, fy1, fx2, fy2 = t.face_box
            fw, fh = fx2 - fx1, fy2 - fy1
            cx = (fx1 + fx2) // 2
            lx1, lx2, ly1, ly2 = cx - int(fw * 1.1), cx + int(fw * 1.1), fy2 + int(fh * 0.2), fy2 + int(fh * 1.6)
        else:
            lx1, lx2, ly1, ly2 = x1 + int(bw * 0.15), x2 - int(bw * 0.15), y1 + int(bh * 0.12), y1 + int(bh * 0.55)
        lx1, lx2 = max(x1, lx1, 0), min(x2, lx2, frame.shape[1])
        ly1, ly2 = max(y1, ly1, 0), min(y2, ly2, frame.shape[0])
        if lx2 - lx1 < 4 or ly2 - ly1 < 4:
            return
        hsv = cv2.cvtColor(frame[ly1:ly2, lx1:lx2], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        for b in self.staff_cfg.brands:
            name, hue = str(b.get("name") or "").strip(), b.get("hue")
            if not name or hue is None:
                continue
            t.brand_px[name] = t.brand_px.get(name, 0) + int(np.count_nonzero(logo_mask(hsv, [int(hue)], int(b.get("tol", 6)))))
        if t.brand_px:
            best = max(t.brand_px, key=t.brand_px.get)
            rest = max([v for k, v in t.brand_px.items() if k != best] or [0])
            if t.brand_px[best] >= BRAND_MIN_PX and t.brand_px[best] >= 2 * rest:
                t.brand = best

    def _attend(self, customers: list[Track], staff: list[Track], dt: float) -> None:
        """Per frame: which brand's staff is near each customer (walk-in attribution) and whether a staff member
        on the phone has a customer in front (product demo rather than personal use)."""
        if dt <= 0 or not staff:
            return
        near = self.thr.attend_distance
        for s in staff:
            sx, sy = s.foot
            any_near = False
            for c in customers:
                if math.hypot(sx - c.foot[0], sy - c.foot[1]) <= near:
                    any_near = True
                    c.served_by[s.id] = c.served_by.get(s.id, 0.0) + dt
            if s.brand:
                self._staff_brand[s.id] = s.brand
            if s.phone_since:
                s.phone_total += dt
                if any_near:
                    s.phone_near += dt

    def _served_live(self, t: Track) -> dict:
        out: dict[str, float] = {}
        for sid, secs in t.served_by.items():
            key = self._staff_brand.get(sid) or "staff"
            out[key] = round(out.get(key, 0.0) + secs, 1)
        return out

    def _served_brand(self, t: Track) -> tuple[str | None, float]:
        # resolve per-staff seconds into brands now: a staff member's brand is often only known after a few samples
        t.served = {}
        for sid, secs in t.served_by.items():
            key = self._staff_brand.get(sid) or "staff"
            t.served[key] = t.served.get(key, 0.0) + secs
        if not t.served:
            return None, 0.0
        best = max(t.served, key=t.served.get)
        secs = t.served[best]
        return (best if secs >= self.thr.serve_seconds else None), round(sum(t.served.values()), 1)

    def _update_passby(self, t: Track, now: float, fx: float, fy: float) -> None:
        """Count a track once when it crosses the pass-by line (either direction, inside the drawn segment).
        Capture rate = walk-ins / passers-by."""
        if t.pb_counted:
            return
        (ax, ay), (bx, by) = self.cam.passby_line
        dx, dy = bx - ax, by - ay
        cross = dx * (fy - ay) - dy * (fx - ax)
        if abs(cross) / (math.hypot(dx, dy) or 1e-9) < self.thr.line_margin:
            return
        s = 1 if cross > 0 else -1
        if t.pb_side == 0:
            t.pb_side, t.pb_pt = s, (fx, fy)
            return
        if s == t.pb_side:
            t.pb_pt, t.pb_pending, t.pb_n = (fx, fy), 0, 0
            return
        if t.pb_pending != s:
            t.pb_pending, t.pb_n = s, 0
        t.pb_n += 1
        if t.pb_n < max(1, int(self.thr.line_min_frames)):
            return
        px, py = t.pb_pt or (fx, fy)
        c0 = dx * (py - ay) - dy * (px - ax)
        c1 = cross
        ta = c0 / (c0 - c1) if (c0 - c1) else 0.0
        qx, qy = px + ta * (fx - px), py + ta * (fy - py)
        u = ((qx - ax) * dx + (qy - ay) * dy) / ((dx * dx + dy * dy) or 1e-12)
        t.pb_side, t.pb_pt, t.pb_pending, t.pb_n = s, (fx, fy), 0, 0
        if -0.1 <= u <= 1.1:
            t.pb_counted = True
            self.passby += 1
            self._event(now, "pass_by", t.id, f"Person #{t.id} passed by")
            self._emit_ce("PASS_BY", t, now)
            if self.store:
                self.store.add_crossing({"camera_id": self.cam.id, "ts": now, "direction": "pass", "track_id": t.id,
                                         "group_id": t.group_id, "is_staff": t.is_staff})

    def _observe_line(self, fx: float, fy: float) -> int:
        """Side of the count line a foot point is on: +1 (IN side), -1 (OUT side), or 0 when it is within
        line_margin of the line, which is too close to call."""
        (ax, ay), (bx, by) = self.cam.line
        dx, dy = bx - ax, by - ay
        cross = dx * (fy - ay) - dy * (fx - ax)
        if abs(cross) / (math.hypot(dx, dy) or 1e-9) < self.thr.line_margin:
            return 0
        return 1 if cross > 0 else -1

    def _update_line(self, t: Track, now: float, fx: float, fy: float) -> None:
        """Count a crossing only when the person clearly changes side (outside the margin band), stays on the
        new side for line_min_frames frames, crossed within the drawn segment, and did not just cross."""
        line = self.cam.line
        if not line:
            t.side = 0
            t.pending_side = t.pending_n = 0
            return
        s = self._observe_line(fx, fy)
        if s == 0:                                   # on the line: no decision this frame
            return
        if t.side == 0 or s == t.side:               # first clear observation, or still on the known side
            t.side = s
            t.side_pt = (fx, fy)
            t.pending_side = t.pending_n = 0
            return
        if t.pending_side != s:
            t.pending_side, t.pending_n = s, 0
        t.pending_n += 1
        if t.pending_n < max(1, int(self.thr.line_min_frames)):
            return
        # confirmed side change: where did the path from the last clear point cut the line?
        (ax, ay), (bx, by) = line
        dx, dy = bx - ax, by - ay
        px, py = t.side_pt if t.side_pt else (fx, fy)
        c0 = dx * (py - ay) - dy * (px - ax)
        c1 = dx * (fy - ay) - dy * (fx - ax)
        ta = c0 / (c0 - c1) if (c0 - c1) else 0.0
        qx, qy = px + ta * (fx - px), py + ta * (fy - py)
        u = ((qx - ax) * dx + (qy - ay) * dy) / ((dx * dx + dy * dy) or 1e-12)
        within = -0.1 <= u <= 1.1                    # inside the drawn segment (with a little slack)
        if within and now - t.last_cross_ts >= self.thr.line_cooldown_seconds:
            if s > 0:
                self.entries += 1
                self._minute["entries"] += 1
                t.crossed_in = True
                t.group_id = self._assign_group(now, fx, fy, t)
                self._event(now, "line_in", t.id, f"Person #{t.id} crossed IN")
                self._record_crossing(now, "in", t)
                self._emit_ce("ENTRY", t, now)
            else:
                self.exits += 1
                self._minute["exits"] += 1
                self._event(now, "line_out", t.id, f"Person #{t.id} crossed OUT")
                self._record_crossing(now, "out", t)
                self._emit_ce("EXIT", t, now, dwell=now - t.first_seen)
            t.last_cross_ts = now
        t.side = s
        t.side_pt = (fx, fy)
        t.pending_side = t.pending_n = 0

    def _assign_group(self, now: float, fx: float, fy: float, t: Track) -> str:
        """People crossing IN within group_window seconds and close together count as one group."""
        for ts, gx, gy, gid in reversed(self._recent_ins):
            if now - ts > self.thr.group_window:
                break
            if math.hypot(fx - gx, fy - gy) <= self.thr.group_distance:
                self._recent_ins.append((now, fx, fy, gid))
                return gid
        gid = f"{self.cam.id}-{int(now * 1000)}-{t.id}"
        self._recent_ins.append((now, fx, fy, gid))
        self.groups_in += 1
        return gid

    def _record_crossing(self, now: float, direction: str, t: Track) -> None:
        if self.store:
            self.store.add_crossing({"camera_id": self.cam.id, "ts": now, "direction": direction, "track_id": t.id,
                                     "group_id": t.group_id or f"{self.cam.id}-{t.id}", "is_staff": t.is_staff})

    def _update_zones(self, t: Track, now: float, fx: float, fy: float, dt: float) -> None:
        for z in self.cam.zones:
            inside = point_in_polygon(fx, fy, z.points)
            if inside and z.name not in t.zones:
                t.zones[z.name] = now
                if not t.zone_seq or t.zone_seq[-1] != z.name:
                    t.zone_seq.append(z.name)
                self._zone(z.name)["visits"] += 1
                self._event(now, "zone_enter", t.id, f"Person #{t.id} entered '{z.name}'", z.name)
                self._emit_ce("ZONE_ENTER", t, now, zone=z.name)
            elif not inside and z.name in t.zones:
                self._emit_ce("ZONE_EXIT", t, now, zone=z.name, dwell=now - t.zones[z.name])
                self._leave_zone(t, z.name, now)
            if inside and z.role == "staff" and not t.is_staff and t.manual_staff is None:
                t.staff_zone_time += dt
                if t.staff_zone_time >= self.thr.staff_zone_seconds:
                    self._set_staff(t, now, "staff zone", z.name)

    def _set_staff(self, t: Track, now: float, reason: str, zone: str | None = None) -> None:
        t.is_staff = True
        t.loitering = False
        t.staff_reason = reason
        self._event(now, "staff", t.id, f"Person #{t.id} classified as staff ({reason})", zone)

    def _sample_torso(self, t: Track, frame: np.ndarray, now: float) -> None:
        if now - t.torso_ts < 0.5:            # one sample every half second spreads votes over time
            return
        t.torso_ts = now
        x1, y1, x2, y2 = t.bbox
        bw, bh = x2 - x1, y2 - y1
        anchored = bool(t.face_box) and now - t.face_ts < 3.0
        if anchored:                          # chest sits just below the face: much more reliable than the box
            fx1, fy1, fx2, fy2 = t.face_box
            fw, fh = fx2 - fx1, fy2 - fy1
            cx = (fx1 + fx2) // 2
            rx1, rx2 = cx - int(fw * 0.8), cx + int(fw * 0.8)
            ry1, ry2 = fy2 + int(fh * 0.3), fy2 + int(fh * 1.1)      # upper chest only: below that is often the desk
        else:                                 # fallback: narrow central band of the person box
            rx1, rx2 = x1 + int(bw * 0.35), x2 - int(bw * 0.35)
            ry1, ry2 = y1 + int(bh * 0.28), y1 + int(bh * 0.50)
        rx1, rx2 = max(x1, rx1, 0), min(x2, rx2, frame.shape[1])
        ry1, ry2 = max(y1, ry1, 0), min(y2, ry2, frame.shape[0])
        if rx2 - rx1 < 4 or ry2 - ry1 < 4:
            return
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        med = np.median(hsv, axis=0)
        t.torso = (int(med[0]), int(med[1]), int(med[2]))
        # a small bright mark (a printed logo) must not make a plain garment look "mixed"
        marks = (hsv[:, 1] > 90) & (hsv[:, 2] > 110)
        plain = hsv[~marks] if 0 < np.count_nonzero(marks) <= 0.25 * len(hsv) else hsv
        mixed = float(plain[:, 2].std()) > 55        # desk + laptop + paper, not a single garment
        t.torso_anchored = anchored and not mixed
        if self.staff_cfg.brands:
            self._vote_brand(t, frame, anchored)
        if t.manual_staff is not None or t.staff_reason not in ("", "uniform"):
            return
        if self.staff_model is not None:
            # learned uniform classifier (models/staff_cls.pt): decides on its own, the colour rules below are
            # only used when no model is installed
            prob = self.staff_model.staff_prob(frame, t.bbox)
            if prob is not None:
                t.model_votes.append(prob)
                n = len(t.model_votes)
                mean = sum(t.model_votes) / n
                if not t.is_staff and n >= MODEL_MIN_SAMPLES and mean >= MODEL_STAFF_PROB:
                    self._set_staff(t, now, "uniform")
                elif t.is_staff and t.staff_reason == "uniform" and n >= 10 and mean <= MODEL_CUSTOMER_PROB:
                    t.is_staff = False
                    t.staff_reason = ""
                    self._event(now, "staff", t.id, f"Person #{t.id} no longer looks like staff, counted as customer")
            return
        if not self.staff_cfg.uniforms:
            return
        plain = [u for u in self.staff_cfg.uniforms if not has_logo(u)]
        logos = [u for u in self.staff_cfg.uniforms if has_logo(u)]
        vote = False
        # plain colour votes only from face-anchored samples; the fallback band too often lands on desks or walls
        if plain and t.torso_anchored:
            vote = any(match_uniform(t.torso, u) for u in plain)
        if logos and not vote:
            # a printed logo is specific enough to be looked for on every sample: below the face when one is
            # visible, otherwise across the upper half of the person box
            if anchored:
                fx1, fy1, fx2, fy2 = t.face_box
                fw, fh = fx2 - fx1, fy2 - fy1
                cx = (fx1 + fx2) // 2
                lx1, lx2 = cx - int(fw * 1.1), cx + int(fw * 1.1)
                ly1, ly2 = fy2 + int(fh * 0.2), fy2 + int(fh * 1.6)
            else:
                lx1, lx2 = x1 + int(bw * 0.15), x2 - int(bw * 0.15)
                ly1, ly2 = y1 + int(bh * 0.12), y1 + int(bh * 0.55)
            lx1, lx2 = max(x1, lx1, 0), min(x2, lx2, frame.shape[1])
            ly1, ly2 = max(y1, ly1, 0), min(y2, ly2, frame.shape[0])
            if lx2 - lx1 >= 4 and ly2 - ly1 >= 4:
                chest = cv2.cvtColor(frame[ly1:ly2, lx1:lx2], cv2.COLOR_BGR2HSV).reshape(-1, 3)
                t.logo_frac = max(logo_fraction(chest, u) for u in logos)
                vote = any(match_uniform(t.torso, u) and logo_present(chest, u) for u in logos)
        if logos or t.torso_anchored:
            t.uniform_votes.append(vote)
            if len(t.uniform_votes) >= 15:
                ratio = sum(t.uniform_votes) / len(t.uniform_votes)
                if not t.is_staff and ratio >= 0.8:
                    self._set_staff(t, now, "uniform")
                elif t.is_staff and t.staff_reason == "uniform" and ratio <= 0.3:
                    t.is_staff = False
                    t.staff_reason = ""
                    self._event(now, "staff", t.id, f"Person #{t.id} no longer matches a uniform, counted as customer")

    def _leave_zone(self, t: Track, name: str, now: float) -> None:
        entered = t.zones.pop(name)
        stay = now - entered
        self._zone(name)["dwell_total"] += stay
        self._event(now, "zone_exit", t.id, f"Person #{t.id} left '{name}' after {stay:.0f}s", name)
        if self.store:
            self.store.add_zone_visit({"camera_id": self.cam.id, "zone": name, "track_id": t.id,
                                       "enter_ts": entered, "exit_ts": now, "dwell": stay, "is_staff": t.is_staff})

    def _update_phones(self, phones: list[Detection], now: float) -> None:
        active = [t for t in self.tracks.values() if now - t.last_seen <= LOST_TIMEOUT]
        for p in phones:
            cx, cy = (p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2
            best = None
            for t in active:
                x1, y1, x2, y2 = t.bbox
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if best is None or area < best[0]:
                        best = (area, t)
            if best:
                t = best[1]
                t.phone_last = now
                t.phone_box = (p.x1, p.y1, p.x2, p.y2)
                if t.phone_since is None:
                    t.phone_since = now
        for t in active:
            if t.phone_since and now - t.phone_last > PHONE_GAP:
                t.phone_since = None
                t.phone_alerted = False
                t.phone_near = t.phone_total = 0.0
            if (t.phone_since and not t.phone_alerted and now - t.phone_since >= self.thr.phone_seconds
                    and (t.is_staff or self.thr.phone_everyone)):
                t.phone_alerted = True
                if t.is_staff and t.phone_total > 0 and t.phone_near / t.phone_total >= self.thr.demo_share:
                    # a customer was in front for most of the episode: a product demo, not personal use
                    share = int(100 * t.phone_near / t.phone_total)
                    self._event(now, "phone_demo", t.id, f"Staff #{t.id} showing a phone to a customer ({share}% of the time with a customer)")
                    em = self.emitter
                    if em is not None and hasattr(em, "compliance"):
                        try:
                            em.compliance(self.cam.id, "MOBILE_USAGE", f"{self.cam.name}: staff #{t.id} phone demo to a customer ({share}% attended)", now, t.id)
                        except Exception:
                            log.exception("compliance emit failed")
                    continue
                self._event(now, "phone", t.id, f"Person #{t.id} using a phone")
                if self.alerts:
                    who = "staff" if t.is_staff else "person"
                    self.alerts.fire(f"phone:{self.cam.id}:{t.id}", "phone_usage",
                                     f"{self.cam.name}: {who} #{t.id} on the phone for {now - t.phone_since:.0f}s",
                                     self.cam.id, None, t.id)

    def apply_faces(self, faces: list) -> None:
        with self._lock:
            self._update_faces(faces, time.time())

    def _update_faces(self, faces: list, now: float) -> None:
        for f in faces:
            t = self.tracks.get(f["track_id"])
            if t is None or t.face_samples >= 200:
                continue
            t.gender_votes[f["gender"]] += f["size"]
            t.age_samples.append(f["age"])
            t.face_samples += 1
            t.face_box = f["box"]
            t.face_ts = now
            t.gender = "M" if t.gender_votes[1] > t.gender_votes[0] else "F"
            t.age = int(round(float(np.median(t.age_samples))))

    def _close_track(self, t: Track, now: float) -> None:
        dwell = t.last_seen - t.first_seen
        if not t.is_staff:
            self.completed_dwells.append(dwell)
        for zname in list(t.zones):
            self._leave_zone(t, zname, t.last_seen)
        self._event(now, "leave", t.id, f"{'Staff' if t.is_staff else 'Person'} #{t.id} left after {dwell:.0f}s")
        brand, served = (None, 0.0) if t.is_staff else self._served_brand(t)
        if brand:
            self.served_today[brand] = self.served_today.get(brand, 0) + 1
            self._event(t.last_seen, "served", t.id, f"Person #{t.id} served by {brand} staff for {t.served[brand]:.0f}s", f"Served by {brand}")
            # Congruence Eye: a SECTION zone named "Served by <brand>" on this camera receives the visit, so its
            # zone analytics (visits, dwell, gender) become the brand's walk-ins
            self._emit_ce("ZONE_ENTER", t, max(t.first_seen, t.last_seen - t.served[brand]), zone=f"Served by {brand}")
            self._emit_ce("ZONE_EXIT", t, t.last_seen, zone=f"Served by {brand}", dwell=t.served[brand])
        if self.store:
            self.store.add_visit({"camera_id": self.cam.id, "track_id": t.id, "start_ts": t.first_seen,
                                  "end_ts": t.last_seen, "dwell": dwell, "engaged": dwell >= self.thr.engage_seconds,
                                  "is_staff": t.is_staff, "zones": t.zone_seq, "crossed_in": t.crossed_in,
                                  "group_id": t.group_id, "gender": t.gender, "age": t.age,
                                  "age_band": t.age_band, "brand": brand, "served_seconds": served})

    def _check_crowd(self, occ: int) -> None:
        if not self.alerts:
            return
        key = f"crowd:{self.cam.id}"
        if occ >= self.thr.crowd_threshold:
            self.alerts.open(key, "crowd", f"{self.cam.name}: crowd of {occ} people", self.cam.id)
        else:
            self.alerts.resolve(key)

    def _check_zones(self, customers: list[Track], staff: list[Track], now: float) -> None:
        if not self.alerts:
            return
        staff_active = self.staff_detection_active
        for z in self.cam.zones:
            st = self._zone_state.setdefault(z.name, {"breach_since": None, "unattended": False})
            in_zone = [t for t in customers if z.name in t.zones]
            if z.role == "queue":
                key = f"queue:{self.cam.id}:{z.name}"
                waits = [now - t.zones[z.name] for t in in_zone]
                q = self._minute["queues"].setdefault(z.name, {"sum_len": 0, "n": 0, "max_len": 0, "sum_wait": 0.0, "max_wait": 0.0})
                q["sum_len"] += len(in_zone)
                q["n"] += 1
                q["max_len"] = max(q["max_len"], len(in_zone))
                if waits:
                    q["sum_wait"] += sum(waits) / len(waits)
                    q["max_wait"] = max(q["max_wait"], max(waits))
                if len(in_zone) >= self.thr.queue_max:
                    st["breach_since"] = st["breach_since"] or now
                    waited = now - st["breach_since"]
                    if waited >= self.thr.queue_seconds:
                        self.alerts.open(key, "queue_breach",
                                         f"{self.cam.name}: {len(in_zone)} waiting at '{z.name}' for {waited / 60:.0f} min",
                                         self.cam.id, z.name)
                else:
                    st["breach_since"] = None
                    self.alerts.resolve(key)
            elif z.role in ("display", "checkout") and staff_active:
                key = f"unattended:{self.cam.id}:{z.name}"
                waiting = [now - t.zones[z.name] for t in in_zone]
                attended = False
                if in_zone:
                    for s in staff:
                        if z.name in s.zones:
                            attended = True
                            break
                        sx, sy = s.foot
                        if any(math.hypot(sx - t.foot[0], sy - t.foot[1]) <= self.thr.attend_distance for t in in_zone):
                            attended = True
                            break
                if in_zone and not attended and max(waiting) >= self.thr.unattended_seconds:
                    st["unattended"] = True
                    self.alerts.open(key, "unattended",
                                     f"{self.cam.name}: customer waiting at '{z.name}' for {max(waiting):.0f}s with no staff",
                                     self.cam.id, z.name)
                else:
                    st["unattended"] = False
                    self.alerts.resolve(key)

    def _heat_add(self, fx: float, fy: float) -> None:
        gx = min(HEAT_W - 1, max(0, int(fx * HEAT_W)))
        gy = min(HEAT_H - 1, max(0, int(fy * HEAT_H)))
        self.heat[gy, gx] += 1.0
        if gx > 0:
            self.heat[gy, gx - 1] += 0.35
        if gx < HEAT_W - 1:
            self.heat[gy, gx + 1] += 0.35
        if gy > 0:
            self.heat[gy - 1, gx] += 0.35
        if gy < HEAT_H - 1:
            self.heat[gy + 1, gx] += 0.35

    def _minute_tick(self, now: float, occ: int, staff: int) -> None:
        bucket = int(now // 60) * 60
        m = self._minute
        if bucket != m["ts"]:
            row = {
                "ts": m["ts"],
                "avg_occ": round(m["sum"] / m["n"], 2) if m["n"] else 0.0,
                "max_occ": m["max"],
                "entries": m["entries"],
                "exits": m["exits"],
                "new_visitors": m["new"],
                "staff_avg": round(m["staff_sum"] / m["n"], 2) if m["n"] else 0.0,
            }
            self.minutes.append(row)
            if self.store:
                self.store.add_minute(self.cam.id, row)
                for zname, q in m["queues"].items():
                    if q["n"]:
                        self.store.add_queue_minute(self.cam.id, zname, {
                            "ts": m["ts"], "avg_len": round(q["sum_len"] / q["n"], 2), "max_len": q["max_len"],
                            "avg_wait": round(q["sum_wait"] / q["n"], 1), "max_wait": round(q["max_wait"], 1)})
            self._minute = m = _fresh_minute(bucket)
        m["sum"] += occ
        m["staff_sum"] += staff
        m["n"] += 1
        m["max"] = max(m["max"], occ)
