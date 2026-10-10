"""Evidence clips: a short MP4 around every alert (loss prevention, SOP breaches, camera events).

Each camera pipeline keeps the last PRE_SECONDS of annotated frames in memory (small JPEGs, a few per second).
When an alert opens, the recorder waits POST_SECONDS, takes the frames around the alert and writes
data/clips/<started>_<camera>_<type>.mp4. The file name is known at alert time, so the alert record, the
notification and the Congruence Eye detection can carry the link before the file exists."""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("clips")

PRE_SECONDS = 10.0
POST_SECONDS = 8.0
FPS = 5.0
KEEP_DAYS = 7
KEEP_FILES = 500


def clip_name(started: float, camera_id: str, atype: str) -> str:
    cam = re.sub(r"[^A-Za-z0-9_-]", "_", camera_id or "cam")
    return f"{int(started)}_{cam}_{re.sub(r'[^a-z0-9_]', '_', (atype or 'alert').lower())}.mp4"


class ClipRecorder:
    def __init__(self, clip_dir: Path, manager, store):
        self.dir = Path(clip_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manager = manager
        self.store = store
        self.written = 0
        self.last_error = ""
        self._lock = threading.Lock()
        self._pending: set[str] = set()

    # ---- public ------------------------------------------------------------
    def request(self, alert: dict) -> str | None:
        """Schedule a clip for an alert that has a camera. Returns the clip file name (relative to the clip dir)."""
        cam_id = alert.get("camera_id")
        if not cam_id or self.manager is None:
            return None
        rt = self.manager.runtimes.get(cam_id)
        if rt is None:
            return None
        name = clip_name(alert.get("started") or time.time(), cam_id, alert.get("type") or "alert")
        with self._lock:
            if name in self._pending:
                return name
            self._pending.add(name)
        threading.Thread(target=self._record, args=(rt, alert, name), name=f"clip-{cam_id}", daemon=True).start()
        return name

    def list(self, limit: int = 100) -> list[dict]:
        files = sorted(self.dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        out = []
        for p in files:
            m = re.match(r"(\d+)_(.+?)_([a-z0-9_]+)\.mp4$", p.name)
            out.append({"file": p.name, "url": f"/clips/{p.name}", "size": p.stat().st_size, "mtime": p.stat().st_mtime,
                        "started": int(m.group(1)) if m else None, "camera_id": m.group(2) if m else None, "type": m.group(3) if m else None})
        return out

    # ---- worker -------------------------------------------------------------
    def _record(self, rt, alert: dict, name: str) -> None:
        started = float(alert.get("started") or time.time())
        try:
            time.sleep(max(0.0, started + POST_SECONDS - time.time()))
            frames = rt.pipeline.recent_frames(started - PRE_SECONDS, started + POST_SECONDS)
            if len(frames) < 2:
                self.last_error = f"{name}: no frames buffered"
                return
            path = self.dir / name
            first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
            h, w = first.shape[:2]
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            if not vw.isOpened():
                self.last_error = f"{name}: VideoWriter could not open"
                return
            # frames are irregularly spaced: repeat / drop so the clip plays in real time at FPS
            t_prev = None
            for ts, jpeg in frames:
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                n = 1 if t_prev is None else max(1, min(int(round((ts - t_prev) * FPS)), int(FPS * 2)))
                for _ in range(n):
                    vw.write(img)
                t_prev = ts
            vw.release()
            self.written += 1
            if self.store is not None and alert.get("id"):
                try:
                    self.store.set_alert_clip(int(alert["id"]), name)
                except Exception as e:
                    log.debug("could not attach clip to alert: %s", e)
            log.info("evidence clip written: %s (%d frames)", name, len(frames))
            self._prune()
        except Exception as e:
            self.last_error = f"{name}: {e}"
            log.warning("evidence clip failed: %s", e)
        finally:
            with self._lock:
                self._pending.discard(name)

    def _prune(self) -> None:
        files = sorted(self.dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        cutoff = time.time() - KEEP_DAYS * 86400
        for i, p in enumerate(files):
            if i >= KEEP_FILES or p.stat().st_mtime < cutoff:
                try:
                    os.remove(p)
                except OSError:
                    pass
