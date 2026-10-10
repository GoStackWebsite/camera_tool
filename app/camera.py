"""Threaded RTSP reader that always exposes the newest frame and reconnects on failure."""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from .config import mask_url

log = logging.getLogger("camera")

STALL_SECONDS = 10.0     # no frame for this long -> treat the stream as dead and reconnect
OPEN_TIMEOUT_MS = 10000  # FFmpeg: give up opening after this long
READ_TIMEOUT_MS = 5000   # FFmpeg: a single read may block at most this long (a silent socket then returns)


class RTSPCamera(threading.Thread):
    def __init__(self, url: str, reconnect_delay: float = 3.0):
        super().__init__(name="rtsp-camera", daemon=True)
        self.url = url
        self.reconnect_delay = reconnect_delay
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._last_frame_ts = 0.0
        self._connected = False
        self._fps = 0.0
        self.width = 0
        self.height = 0
        self.source_fps = 0.0
        self.last_error = ""
        self.stalls = 0          # times the stream went silent and was dropped
        self.reconnects = 0      # times the reader re-opened the stream
        self.last_attempt = 0.0  # when the reader last tried to open the stream (the watchdog leaves a retrying reader alone)
        self._fail_logged = 0.0

    # ---- public API -------------------------------------------------------
    def latest(self) -> tuple[int, np.ndarray] | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._seq, self._frame

    @property
    def connected(self) -> bool:
        return self._connected and (time.time() - self._last_frame_ts) < STALL_SECONDS

    @property
    def fps(self) -> float:
        return self._fps if self.connected else 0.0

    def stop(self) -> None:
        self._stop.set()

    # ---- thread body ------------------------------------------------------
    def run(self) -> None:
        delay = self.reconnect_delay
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self._connected = False
                time.sleep(delay)
                delay = min(delay * 1.5, 30.0)
                continue
            delay = self.reconnect_delay
            self._read_loop(cap)
            cap.release()
            self._connected = False
            if not self._stop.is_set():
                self.reconnects += 1
                log.warning("Stream lost (%s), reconnecting in %.0fs", self.last_error or "read failed", self.reconnect_delay)
                time.sleep(self.reconnect_delay)

    def _open(self) -> cv2.VideoCapture | None:
        self.last_attempt = time.time()
        if self.last_attempt - self._fail_logged > 60:    # while a camera is unreachable: one line a minute, not one per try
            log.info("Connecting to %s", mask_url(self.url))
        params: list[int] = []
        for name, ms in (("CAP_PROP_OPEN_TIMEOUT_MSEC", OPEN_TIMEOUT_MS), ("CAP_PROP_READ_TIMEOUT_MSEC", READ_TIMEOUT_MS)):
            prop = getattr(cv2, name, None)
            if prop is not None:
                params += [int(prop), int(ms)]
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        if not cap.isOpened():
            self.last_error = "could not open stream (check URL / credentials / network)"
            now = time.time()
            if now - self._fail_logged > 60:
                log.error("Open failed: %s (retrying every few seconds, logged once a minute)", self.last_error)
                self._fail_logged = now
            cap.release()
            return None
        self._fail_logged = 0.0
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        log.info("Connected: %dx%d @ %.1f fps", self.width, self.height, self.source_fps)
        self.last_error = ""
        return cap

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        """Publish frames until the stream goes silent for STALL_SECONDS (then return so run() reconnects)."""
        tick = time.time()
        frames = 0
        last = tick                      # the successful open counts as activity
        while not self._stop.is_set():
            ok, frame = cap.read()
            now = time.time()
            if not ok or frame is None:
                if now - last >= STALL_SECONDS:
                    self.last_error = f"no frames for {now - last:.0f}s"
                    self.stalls += 1
                    return
                time.sleep(0.02)
                continue
            last = now
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._last_frame_ts = now
                self._connected = True
            frames += 1
            if now - tick >= 2.0:
                self._fps = frames / (now - tick)
                frames = 0
                tick = now
