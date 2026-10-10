"""Owns every camera runtime (reader + tracker + analytics + pipeline), camera health,
store open/close detection, and the daily report schedule."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2

from .alerts import AlertManager
from .analytics import Analytics
from .camera import RTSPCamera
from .config import MODELS_DIR, AppConfig, CameraConfig, ConfigStore, Settings, StaffConfig, Thresholds, mask_url
from .demographics import Demographics
from .staffcls import StaffClassifier
from .detector import PersonTracker
from .notify import Notifier
from .pipeline import Pipeline
from .store import Store, day_bounds

log = logging.getLogger("manager")
OFFLINE_ALERT_AFTER = 60.0   # seconds a camera must be down before alerting
WATCHDOG_AFTER = 45.0        # seconds without frames before the reader is replaced
WATCHDOG_GRACE = 60.0        # never replace the reader of a runtime younger than this
WATCHDOG_GAP = 60.0          # minimum time between two watchdog restarts of one camera


@dataclass
class CameraRuntime:
    cfg: CameraConfig
    camera: RTSPCamera
    tracker: PersonTracker
    analytics: Analytics
    pipeline: Pipeline
    started_at: float
    was_connected: bool = False
    offline_since: float | None = None
    last_watchdog: float = 0.0
    watchdog_restarts: int = 0


class Manager:
    def __init__(self, settings: Settings, config_store: ConfigStore, store: Store,
                 alerts: AlertManager, notifier: Notifier):
        self.settings = settings
        self.config_store = config_store
        self.store = store
        self.alerts = alerts
        self.notifier = notifier
        self.runtimes: dict[str, CameraRuntime] = {}
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._day: dict = {"date": None, "opened_at": None, "last_presence": None, "open_late_fired": False,
                           "report_sent": False, "closed_fired": False}
        self._health_thread = threading.Thread(target=self._health_loop, name="health", daemon=True)
        self.report_builder = None   # set by server (avoids circular import)
        self.connector = None        # Congruence Eye connector (set by server)
        self.reconciler = None       # Congruence Eye managed-mode sync (set by server)

    @property
    def config(self) -> AppConfig:
        return self.config_store.config

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> None:
        for cam in self.config.cameras:
            if cam.enabled:
                try:
                    self._start_runtime(cam)
                except Exception:
                    log.exception("Could not start camera %s", cam.id)
        self._health_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for rt in list(self.runtimes.values()):
            rt.pipeline.stop()
            rt.camera.stop()

    def _start_runtime(self, cam: CameraConfig) -> CameraRuntime:
        log.info("Starting camera %s (%s): %s", cam.id, cam.name, mask_url(cam.url))
        camera = RTSPCamera(cam.url)
        tracker = PersonTracker(self.settings.model, self.settings.device, self.settings.conf, self.settings.imgsz)
        staff_model = StaffClassifier.create(MODELS_DIR)      # learned uniform, when models/staff_cls.pt exists
        analytics = Analytics(cam, self.config.thresholds, self.store, self.alerts, self.config.staff, staff_model)
        demographics = Demographics.create(MODELS_DIR) if self.config.demographics.enabled else None
        pipeline = Pipeline(camera, tracker, analytics, self.settings, cam.target_fps, demographics, self.config_store)
        rt = CameraRuntime(cam, camera, tracker, analytics, pipeline, time.time())
        camera.start()
        pipeline.start()
        with self.lock:
            self.runtimes[cam.id] = rt
        return rt

    def _stop_runtime(self, cam_id: str) -> None:
        with self.lock:
            rt = self.runtimes.pop(cam_id, None)
        if rt:
            rt.pipeline.stop()
            rt.camera.stop()
            for a in self.alerts.list_open():
                if a.get("camera_id") == cam_id:
                    self.alerts.resolve(a["key"])

    # ---- config changes ---------------------------------------------------
    def add_camera(self, cam: CameraConfig) -> None:
        with self.config_store.lock:
            if self.config.camera(cam.id):
                raise ValueError(f"camera id '{cam.id}' already exists")
            self.config.cameras.append(cam)
            self.config_store.save()
        if cam.enabled:
            self._start_runtime(cam)

    def update_camera(self, cam: CameraConfig) -> None:
        with self.config_store.lock:
            old = self.config.camera(cam.id)
            if old is None:
                raise KeyError(cam.id)
            self.config.cameras[self.config.cameras.index(old)] = cam
            self.config_store.save()
        rt = self.runtimes.get(cam.id)
        needs_restart = (old.url != cam.url or old.enabled != cam.enabled or old.target_fps != cam.target_fps)
        if rt and not needs_restart:
            rt.cfg = cam
            rt.analytics.apply_camera_config(cam)
            return
        if rt:
            self._stop_runtime(cam.id)
        if cam.enabled:
            self._start_runtime(cam)

    def remove_camera(self, cam_id: str) -> None:
        with self.config_store.lock:
            cam = self.config.camera(cam_id)
            if cam is None:
                raise KeyError(cam_id)
            self.config.cameras.remove(cam)
            self.config_store.save()
        self._stop_runtime(cam_id)

    def apply_thresholds(self, thr: Thresholds) -> None:
        with self.config_store.lock:
            self.config.thresholds = thr
            self.config_store.save()
        for rt in self.runtimes.values():
            rt.analytics.apply_thresholds(thr)

    def apply_staff(self, cfg: StaffConfig) -> None:
        with self.config_store.lock:
            self.config.staff = cfg
            self.config_store.save()
        for rt in self.runtimes.values():
            rt.analytics.apply_staff(cfg)

    def reset_counters(self) -> None:
        for rt in self.runtimes.values():
            rt.analytics.reset()

    @staticmethod
    def test_camera(url: str, seconds: float = 6.0) -> dict:
        t0 = time.time()
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return {"ok": False, "error": "could not open stream (check URL, credentials, network)"}
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = 0
        start = time.time()
        while time.time() - start < min(seconds, 3.0):
            ok, _ = cap.read()
            if not ok:
                break
            frames += 1
        cap.release()
        el = max(time.time() - start, 1e-6)
        return {"ok": frames > 0, "width": w, "height": h, "fps": round(frames / el, 1),
                "open_seconds": round(time.time() - t0, 1), "error": "" if frames else "stream opened but no frames"}

    # ---- stats ------------------------------------------------------------
    def stats(self) -> dict:
        cams = {}
        totals = {"occupancy": 0, "staff": 0, "entries": 0, "exits": 0, "groups_in": 0, "unique": 0,
                  "loitering": 0, "phones": 0, "cameras_online": 0, "cameras_total": 0}
        dwell_sum, dwell_n = 0.0, 0
        with self.lock:
            runtimes = list(self.runtimes.values())
        for rt in runtimes:
            s = rt.pipeline.stats()
            cams[rt.cfg.id] = s
            totals["cameras_total"] += 1
            totals["cameras_online"] += int(s["camera"]["connected"])
            for k in ("occupancy", "staff", "entries", "exits", "groups_in", "unique", "loitering", "phones"):
                totals[k] += s.get(k, 0)
            if s.get("occupancy"):
                dwell_sum += s["avg_dwell"] * s["occupancy"]
                dwell_n += s["occupancy"]
        totals["avg_dwell"] = round(dwell_sum / dwell_n, 1) if dwell_n else 0.0
        disabled = [c.id for c in self.config.cameras if not c.enabled]
        return {
            "ts": time.time(),
            "store": {"name": self.config.store.name, **totals,
                      "opened_at": self._day.get("opened_at"), "last_presence": self._day.get("last_presence")},
            "cameras": cams,
            "disabled_cameras": disabled,
            "open_alerts": self.alerts.list_open(),
            "notifier": {"sent": self.notifier.sent, "last_error": self.notifier.last_error},
            "congruence": self.connector.status() if self.connector else None,
            "staff_model": next((rt.analytics.staff_model.info() for rt in runtimes if rt.analytics.staff_model), {"loaded": False}),
            "demographics": {"enabled": self.config.demographics.enabled,
                             "available": any(rt.pipeline.demographics is not None for rt in runtimes),
                             "faces_seen": sum(rt.pipeline.demographics.faces_seen for rt in runtimes if rt.pipeline.demographics)},
        }

    # ---- health / store hours / reports ----------------------------------
    def _health_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick_health()
                self._tick_store_day()
                self._tick_report()
            except Exception:
                log.exception("health loop error")
            self._stop.wait(5.0)

    def _tick_health(self) -> None:
        now = time.time()
        with self.lock:
            runtimes = list(self.runtimes.values())
        for rt in runtimes:
            connected = rt.camera.connected
            if connected != rt.was_connected or (now - rt.started_at < 6 and connected and rt.offline_since is None and not rt.was_connected):
                self.store.add_health(rt.cfg.id, "online" if connected else "offline", rt.camera.last_error)
                rt.was_connected = connected
            if connected:
                rt.offline_since = None
                self.alerts.resolve(f"camera_offline:{rt.cfg.id}")
                self._track_frame_size(rt)
            else:
                rt.offline_since = rt.offline_since or now
                if now - rt.offline_since >= OFFLINE_ALERT_AFTER:
                    self.alerts.open(f"camera_offline:{rt.cfg.id}", "camera_offline",
                                     f"{rt.cfg.name} offline for {(now - rt.offline_since) / 60:.0f} min: {rt.camera.last_error or 'no frames'}",
                                     rt.cfg.id)
                # the watchdog is for a reader that is stuck; a reader that keeps retrying an unreachable camera
                # (recent open attempt) is doing its job and is left alone
                if (rt.cfg.enabled and now - rt.offline_since >= WATCHDOG_AFTER
                        and now - rt.started_at >= WATCHDOG_GRACE and now - rt.last_watchdog >= WATCHDOG_GAP
                        and now - rt.camera.last_attempt >= WATCHDOG_AFTER):
                    self._watchdog_restart(rt, now)

    def _watchdog_restart(self, rt: CameraRuntime, now: float) -> None:
        """The reader delivered nothing for a long time although it should have reconnected by itself:
        replace it. Tracker, analytics and counters stay; the old reader thread ends at its next read return."""
        rt.watchdog_restarts += 1
        rt.last_watchdog = now
        log.warning("watchdog: camera %s silent for %.0fs (%s): replacing the RTSP reader (restart #%d)",
                    rt.cfg.id, now - (rt.offline_since or now), rt.camera.last_error or "no frames", rt.watchdog_restarts)
        try:
            rt.camera.stop()
        except Exception:
            log.exception("watchdog: stopping the old reader of %s failed", rt.cfg.id)
        camera = RTSPCamera(rt.cfg.url)
        camera.stalls, camera.reconnects = rt.camera.stalls, rt.camera.reconnects   # keep the counters
        camera.start()
        rt.camera = camera
        rt.pipeline.replace_camera(camera)
        self.store.add_health(rt.cfg.id, "offline", f"watchdog restart #{rt.watchdog_restarts}")

    def _track_frame_size(self, rt: CameraRuntime) -> None:
        """Remember the decoded frame size; managed cameras map the server's editor coordinates with it."""
        w, h = rt.camera.width, rt.camera.height
        if not w or not h or (w, h) == (rt.cfg.frame_w, rt.cfg.frame_h):
            return
        with self.config_store.lock:
            cam = self.config.camera(rt.cfg.id)
            if cam is None:
                return
            cam.frame_w, cam.frame_h = w, h
            rt.cfg = cam
            self.config_store.save()
        if cam.managed and self.reconciler is not None:
            try:
                self.reconciler.rederive(cam.id)
            except Exception:
                log.exception("re-mapping the zones of %s failed", cam.id)

    @staticmethod
    def _hm_to_ts(day_start: float, hm: str) -> float:
        try:
            h, m = (int(x) for x in hm.split(":"))
        except Exception:
            h, m = 0, 0
        return day_start + h * 3600 + m * 60

    def _tick_store_day(self) -> None:
        now = time.time()
        day_start, _ = day_bounds(now)
        date = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        d = self._day
        if d["date"] != date:
            saved = self.store.daily_store(date) or {}
            d.update({"date": date, "opened_at": saved.get("opened_at"), "last_presence": saved.get("closed_at"),
                      "open_late_fired": False, "report_sent": False, "closed_fired": False})
            # if the app itself started after opening time, do not complain about today
            if now > self._hm_to_ts(day_start, self.config.store.open_time) and d["opened_at"] is None:
                d["open_late_fired"] = True
        present = any((rt.pipeline.last_stats.get("occupancy", 0) + rt.pipeline.last_stats.get("staff", 0)) > 0
                      for rt in self.runtimes.values())
        st = self.config.store
        open_ts = self._hm_to_ts(day_start, st.open_time)
        close_ts = self._hm_to_ts(day_start, st.close_time)
        grace = st.grace_minutes * 60
        if present:
            if d["opened_at"] is None:
                d["opened_at"] = now
                self.store.set_daily_store(date, opened_at=now)
                self._ce_store("STORE_OPENED", now)
                self.alerts.resolve(f"store_open_late:{date}")
            if d["last_presence"] is None or now - d["last_presence"] >= 60:
                self.store.set_daily_store(date, closed_at=now)
            d["last_presence"] = now
        if d["opened_at"] is None and now > open_ts + grace and not d["open_late_fired"] and self.runtimes:
            d["open_late_fired"] = True
            self.alerts.open(f"store_open_late:{date}", "store_open_late",
                             f"{st.name}: nobody seen by {st.open_time} + {st.grace_minutes} min")
        if now > close_ts + grace and present:
            self.alerts.open(f"store_not_closed:{date}", "store_not_closed",
                             f"{st.name}: people still present after closing time {st.close_time}")
        elif not present:
            self.alerts.resolve(f"store_not_closed:{date}")
            if now > close_ts and d["opened_at"] and not d.get("closed_fired"):
                d["closed_fired"] = True
                self._ce_store("STORE_CLOSED", now)

    def _ce_store(self, etype: str, now: float) -> None:
        if self.connector is not None:
            try:
                self.connector.store_event(etype, now, f"{self.config.store.name}: {etype.replace('_', ' ').lower()}")
            except Exception:
                log.exception("Congruence Eye store event failed")

    def _tick_report(self) -> None:
        rep = self.config.report
        if not rep.enabled or self.report_builder is None or self._day.get("report_sent"):
            return
        now = datetime.now()
        if now.strftime("%H:%M") != rep.daily_time:
            return
        self._day["report_sent"] = True
        try:
            name, data = self.report_builder(now.date())
            to = rep.email_to or self.config.notifications.email_to
            result = self.notifier.send_email(f"Daily store report {now:%Y-%m-%d}", "Attached: today's footfall, zones and alerts.", to, (name, data))
            log.info("Daily report: %s", result)
        except Exception:
            log.exception("Daily report failed")
