"""Anonymous visitor re-identification: the same person seen twice in a day (or within the repeat window) is one
unique visitor. Appearance only (clothing / build), no face recognition: an ImageNet-trained YOLO11 classification
network embeds the upper body of each track, the mean embedding is compared (cosine) with the day's gallery.

Limits, by design: people in near-identical clothing can merge, a changed outfit splits; staff are never matched.
One instance per camera pipeline (the ultralytics predictor is not thread-safe); the gallery is shared per store.

In use: Embedder + RevisitGate (one walk-in per person, app/analytics.py). VisitorGallery is the whole-day prototype
and is not wired in: on 10 Oct its MATCH_THRESHOLD of 0.82 also matched about two thirds of pairs of different people."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("reid")

EMBED_MODEL = "yolo11n-cls.pt"      # ImageNet features, 1280-d after global pooling
IMGSZ = 128
MATCH_THRESHOLD = 0.82               # cosine similarity: same person
SAMPLES_PER_TRACK = 6                # embeddings averaged per track (0.5 s apart after the first 1 s)
GALLERY_HOURS = 14.0                 # a visitor is matchable this long after last seen (a shop day)
CROP_BOTTOM = 0.85                   # head to knees: clothing is the signal


def upper_body(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = y2 - y1, x2 - x1
    if h < 40 or w < 16:
        return None
    y2 = min(frame.shape[0], int(y1 + h * CROP_BOTTOM))
    crop = frame[max(0, y1):y2, max(0, x1):min(frame.shape[1], x2)]
    return crop if crop.size and crop.shape[0] >= 24 else None


class Embedder:
    def __init__(self, models_dir: Path):
        from ultralytics import YOLO
        path = Path(models_dir) / EMBED_MODEL
        self.model = YOLO(str(path) if path.exists() else EMBED_MODEL)   # ultralytics downloads it once
        self.samples = 0

    @classmethod
    def create(cls, models_dir: Path) -> "Embedder | None":
        try:
            return cls(models_dir)
        except Exception as e:
            log.warning("re-identification disabled (%s)", e)
            return None

    def embed(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        crop = upper_body(frame, bbox)
        if crop is None:
            return None
        try:
            v = self.model.embed(crop, imgsz=IMGSZ, verbose=False)[0]
            v = v.float().cpu().numpy().reshape(-1)
            n = float(np.linalg.norm(v))
            if n == 0:
                return None
            self.samples += 1
            return v / n
        except Exception as e:
            log.debug("embed failed: %s", e)
            return None

    @staticmethod
    def colour(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        return colour_signature(frame, bbox)


class VisitorGallery:
    """Day gallery of visitor embeddings shared by all cameras of the store (thread-safe)."""

    def __init__(self, store=None):
        self.store = store
        self._lock = threading.Lock()
        self._visitors: dict[str, dict] = {}        # visitor_id -> {"emb", "first", "last", "visits", "n"}
        self._next = int(time.time())
        self._bound: dict[str, float] = {}           # visitor_id -> ts while a live track holds it
        self.matches = 0
        self.created = 0
        if store is not None:
            try:
                for v in store.load_visitors(time.time() - GALLERY_HOURS * 3600):
                    self._visitors[v["visitor_id"]] = {"emb": np.frombuffer(v["embedding"], np.float32).copy(), "first": v["first_seen"],
                                                       "last": v["last_seen"], "visits": v["visits"], "n": 1}
            except Exception as e:
                log.debug("gallery not loaded: %s", e)

    def assign(self, emb: np.ndarray, now: float, current: str | None = None) -> tuple[str, bool]:
        """Match a (mean) embedding; returns (visitor_id, is_repeat). `current` keeps a track on its own id."""
        with self._lock:
            self._expire(now)
            if current and current in self._visitors:
                self._update(current, emb, now, new_visit=False)
                return current, self._visitors[current]["visits"] > 1
            best, best_s = None, 0.0
            for vid, v in self._visitors.items():
                if vid in self._bound and now - self._bound[vid] < 5.0:     # held by another live track right now
                    continue
                s = float(np.dot(v["emb"], emb))
                if s > best_s:
                    best, best_s = vid, s
            if best and best_s >= MATCH_THRESHOLD:
                self._update(best, emb, now, new_visit=True)
                self._bound[best] = now
                self.matches += 1
                return best, True
            self._next += 1
            vid = f"v{self._next}"
            self._visitors[vid] = {"emb": emb.astype(np.float32), "first": now, "last": now, "visits": 1, "n": 1}
            self._bound[vid] = now
            self.created += 1
            self._persist(vid)
            return vid, False

    def touch(self, vid: str, now: float) -> None:
        with self._lock:
            self._bound[vid] = now
            if vid in self._visitors:
                self._visitors[vid]["last"] = now

    def release(self, vid: str) -> None:
        with self._lock:
            self._bound.pop(vid, None)

    def stats(self) -> dict:
        with self._lock:
            return {"visitors": len(self._visitors), "matches": self.matches, "created": self.created,
                    "repeat": sum(1 for v in self._visitors.values() if v["visits"] > 1)}

    # ---- internals ---------------------------------------------------------
    def _update(self, vid: str, emb: np.ndarray, now: float, new_visit: bool) -> None:
        v = self._visitors[vid]
        n = v["n"]
        mixed = (v["emb"] * n + emb) / (n + 1)
        v["emb"] = (mixed / (np.linalg.norm(mixed) or 1.0)).astype(np.float32)
        v["n"] = min(n + 1, 20)
        v["last"] = now
        if new_visit:
            v["visits"] += 1
        self._persist(vid)

    def _expire(self, now: float) -> None:
        old = [vid for vid, v in self._visitors.items() if now - v["last"] > GALLERY_HOURS * 3600]
        for vid in old:
            self._visitors.pop(vid, None)
            self._bound.pop(vid, None)

    def _persist(self, vid: str) -> None:
        if self.store is None:
            return
        v = self._visitors[vid]
        try:
            self.store.save_visitor(vid, v["emb"].astype(np.float32).tobytes(), v["first"], v["last"], v["visits"])
        except Exception as e:
            log.debug("visitor not saved: %s", e)


# ---------------------------------------------------------------- one walk-in per person
# Measured on 17 Oct with 281 shop snapshots of 12-13 Oct: 145 pairs of people in the same frame (certainly two people)
# against 159 same-person pairs (next snapshot, same spot). Embedding cosine alone at 0.90 matched 13 % of the
# different-people pairs; the colour histogram separates far better, and both together at 0.90 / 0.60 kept 95 % of
# the same-person pairs and matched none of the 145 different ones. Real returns (other pose, other light) score
# lower than those same-person pairs, so some returns are counted again: the rule errs towards counting.
REVISIT_SIMILARITY = 0.90            # cosine of the mean embeddings
REVISIT_COLOUR = 0.60                # and colour similarity of upper and lower body (1 - Bhattacharyya distance)
REVISIT_MARGIN = 0.03                # and clearly closer than the next leaver: two look-alikes leave it a new walk-in
COLOUR_BINS = [12, 4, 4]             # HSV histogram per body half


def colour_signature(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """Normalised HSV histograms of the upper and the lower half of a person box, stacked (2 x 192)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
    if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 8:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    half = hsv.shape[0] // 2
    out = []
    for part in (hsv[:half], hsv[half:]):
        h = cv2.calcHist([part], [0, 1, 2], None, COLOUR_BINS, [0, 180, 0, 256, 0, 256]).flatten()
        out.append(h / (h.sum() or 1.0))
    return np.stack(out).astype(np.float32)


def colour_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([1.0 - cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA) for x, y in zip(a, b)]))


class RevisitGate:
    """Customers who left through the door recently, shared by the store's cameras (thread-safe). Someone walking in
    who looks like one of them is that person coming back: the same walk-in, not a new one.

    Deliberately small and short-lived: a newcomer is compared with the few people who went out in the last
    reentry_minutes, not with a whole day's gallery, so the weak ImageNet features rarely meet a look-alike. Any doubt
    (below REVISIT_SIMILARITY, or two leavers nearly as close) counts the person as a new walk-in, as before."""

    def __init__(self):
        self._lock = threading.Lock()
        self._left: list[dict] = []
        self.remembered = 0
        self.matched = 0
        self.ambiguous = 0

    def remember(self, emb: np.ndarray, colour: np.ndarray, ts: float, cam: str, track: int, exited: bool,
                 visitor: str | None) -> None:
        with self._lock:
            self._left.append({"emb": emb.astype(np.float32), "colour": colour.astype(np.float32), "ts": ts,
                               "cam": cam, "track": track,
                               "exited": exited, "visitor": visitor})
            del self._left[:-200]
            self.remembered += 1

    def waiting(self, now: float, window: float) -> bool:
        """Is anybody who could be coming back right now on the list?"""
        with self._lock:
            return any(0.0 <= now - v["ts"] <= window for v in self._left)

    def match(self, emb: np.ndarray, colour: np.ndarray, now: float, window: float) -> dict | None:
        """The leaver this newcomer is, or None. A match is taken off the list (one return per departure)."""
        with self._lock:
            self._left = [v for v in self._left if now - v["ts"] <= window]
            scored = []
            for i, v in enumerate(self._left):
                if v["ts"] > now:
                    continue
                s, c = float(np.dot(v["emb"], emb)), colour_similarity(v["colour"], colour)
                if s >= REVISIT_SIMILARITY and c >= REVISIT_COLOUR:
                    scored.append((s + c, s, c, i))
            if not scored:
                return None
            scored.sort(reverse=True)
            if len(scored) > 1 and scored[0][0] - scored[1][0] < REVISIT_MARGIN:
                self.ambiguous += 1
                return None
            _, s, c, i = scored[0]
            v = self._left.pop(i)
            self.matched += 1
            return {**v, "similarity": s, "colour_similarity": c}

    def stats(self, now: float | None = None) -> dict:
        with self._lock:
            return {"waiting": len(self._left), "remembered": self.remembered, "matched": self.matched,
                    "ambiguous": self.ambiguous}
