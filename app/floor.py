"""Floor plan: where on the shop floor, in metres, a point of a camera picture is.

Each camera is calibrated with a few marks on the floor (data/floor.json): a point clicked on the picture, at floor
level (a tile corner, the foot of a fixture), and where that point is on the floor. The floor is flat, so four or
more such marks fix a homography from the picture to the floor. All cameras of a store share one set of floor
coordinates, so a person seen by two cameras stands at (nearly) the same place on the floor in both: that is how
store fusion links them anywhere the views overlap (app/fusion.py), not only in hand-drawn shared regions.

Floor coordinates are entered in *units* (usually floor tiles: count the tile corners from one corner both cameras
can see) and ``unit_m`` says how many metres one unit is; everything this module returns is in metres.

    {"unit_m": 0.6,
     "cameras": {"ce31": {"points": [{"img": [0.41, 0.80], "floor": [0, 0]}, ...]}, "ce32": {...}}}

Picture points are fractions of the frame (0-1), as zones are. Accuracy is reported as the distance between where a
mark was said to be and where the fitted mapping puts it (``error_m``); with exactly four marks that is always zero,
so a fifth mark is asked for before the calibration is called checked."""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("floor")

MIN_POINTS = 4
MAX_POINTS = 40
GOOD_ERROR_M = 0.25         # mean distance between a mark and where the mapping puts it: below this, usable
DEFAULT_UNIT_M = 0.6        # the common 600 mm floor tile


class FloorError(ValueError):
    """A calibration that cannot be used, with a reason the page can show as is."""


def _num(v, name: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise FloorError(f"{name} must be a number") from None
    if not math.isfinite(f):
        raise FloorError(f"{name} must be a number")
    return f


def clean_points(points) -> list[dict]:
    """[{img: [x, y] (0-1), floor: [X, Y] (units)}], checked; raises FloorError."""
    if not isinstance(points, list):
        raise FloorError("points must be a list")
    if len(points) > MAX_POINTS:
        raise FloorError(f"at most {MAX_POINTS} marks per camera")
    out = []
    for i, p in enumerate(points, 1):
        try:
            img, flo = p["img"], p["floor"]
        except (TypeError, KeyError):
            raise FloorError(f"mark {i} needs a picture point and a floor position") from None
        if not (isinstance(img, (list, tuple)) and len(img) == 2 and isinstance(flo, (list, tuple)) and len(flo) == 2):
            raise FloorError(f"mark {i} needs a picture point and a floor position")
        x, y = (_num(v, f"mark {i}") for v in img)
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise FloorError(f"mark {i} is outside the picture")
        fx, fy = (_num(v, f"mark {i} floor position") for v in flo)
        if abs(fx) > 1000 or abs(fy) > 1000:
            raise FloorError(f"mark {i}: floor position out of range")
        out.append({"img": [round(x, 5), round(y, 5)], "floor": [round(fx, 4), round(fy, 4)]})
    for i in range(len(out)):
        for j in range(i):
            if out[i]["floor"] == out[j]["floor"]:
                raise FloorError(f"marks {j + 1} and {i + 1} have the same floor position")
            if math.dist(out[i]["img"], out[j]["img"]) < 0.004:
                raise FloorError(f"marks {j + 1} and {i + 1} are on the same spot of the picture")
    return out


def _spread(pts: np.ndarray) -> float:
    """How far the points are from lying on one line: the smaller singular value of the centred points, relative
    to the larger. Near 0: (nearly) collinear, no plane can be fitted."""
    c = pts - pts.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    return float(s[1] / s[0]) if s[0] > 0 else 0.0


def fit(points: list[dict], unit_m: float, aspect: float = 16 / 9) -> dict:
    """Fit picture -> floor (metres) for one camera. Returns {"H": 3x3 picture->floor, "Hinv": floor->picture,
    "error_m": mean, "max_error_m", "worst": index of the worst mark, "checked": bool}; raises FloorError.

    `aspect` (frame width / height) only weighs the collinearity test: picture fractions stretch a wide frame."""
    if len(points) < MIN_POINTS:
        raise FloorError(f"{MIN_POINTS} marks are needed, there {'is' if len(points) == 1 else 'are'} {len(points)}")
    img = np.array([p["img"] for p in points], np.float64)
    flo = np.array([p["floor"] for p in points], np.float64) * unit_m
    if _spread(img * [aspect, 1.0]) < 0.05:
        raise FloorError("the marks lie (nearly) on one line in the picture: spread them over the floor")
    if _spread(flo) < 0.05:
        raise FloorError("the floor positions lie (nearly) on one line: use marks in at least two rows")
    H, _ = cv2.findHomography(img, flo, 0)
    if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        raise FloorError("these marks do not describe a flat floor: check the floor positions")
    H = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H
    w = img @ H[2, :2] + H[2, 2]
    if not (np.all(w > 0) or np.all(w < 0)):
        raise FloorError("some marks put the floor behind the camera: a floor position is probably wrong")
    proj = cv2.perspectiveTransform(img.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - flo, axis=1)
    return {"H": H, "Hinv": np.linalg.inv(H), "sign": 1.0 if w[0] > 0 else -1.0,
            "error_m": float(err.mean()), "max_error_m": float(err.max()), "worst": int(err.argmax()),
            "checked": len(points) > MIN_POINTS}


class FloorMap:
    """data/floor.json, re-read when it changes; thread-safe."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mtime = None
        self.unit_m = DEFAULT_UNIT_M
        self.points: dict[str, list[dict]] = {}
        self.fits: dict[str, dict] = {}
        self.errors: dict[str, str] = {}
        self.error = ""
        self.reload()

    def reload(self) -> None:
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            mtime = None
        if mtime == self._mtime:
            return
        self._mtime = mtime
        unit_m, points, error = DEFAULT_UNIT_M, {}, ""
        if mtime is not None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                unit_m = _num(raw.get("unit_m", DEFAULT_UNIT_M), "unit_m")
                for cam, c in (raw.get("cameras") or {}).items():
                    try:
                        points[str(cam)] = clean_points((c or {}).get("points") or [])
                    except FloorError as e:
                        error = f"{cam}: {e}"
            except Exception as e:
                error = f"{self.path.name}: {e}"
                log.warning("floor plan not usable: %s", error)
        self._apply(unit_m, points, error)

    def _apply(self, unit_m: float, points: dict, error: str) -> None:
        fits, errors = {}, {}
        for cam, pts in points.items():
            if not pts:
                continue
            try:
                fits[cam] = fit(pts, unit_m)
            except FloorError as e:
                errors[cam] = str(e)
        with self._lock:
            self.unit_m, self.points, self.fits, self.errors, self.error = unit_m, points, fits, errors, error
        if fits:
            log.info("floor plan: %s", ", ".join(f"{c} ({f['error_m'] * 100:.0f} cm)" for c, f in sorted(fits.items())))

    # ---- use -------------------------------------------------------------
    def calibrated(self, cam: str) -> bool:
        return cam in self.fits

    def to_floor(self, cam: str, x: float, y: float) -> tuple[float, float] | None:
        """Picture point (fractions) -> floor (metres), or None when this camera is not calibrated or the point is
        above the horizon (not on the floor)."""
        f = self.fits.get(cam)
        if f is None:
            return None
        H = f["H"]
        w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        if w * f["sign"] <= 1e-9:
            return None
        return ((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w)

    # ---- the page ---------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            cams = {}
            for cam in sorted(set(self.points) | set(self.fits)):
                f = self.fits.get(cam)
                cams[cam] = {"points": self.points.get(cam, []), "ok": f is not None, "message": self.errors.get(cam, ""),
                             **({"error_m": round(f["error_m"], 3), "max_error_m": round(f["max_error_m"], 3),
                                 "worst": f["worst"], "checked": f["checked"], "good": f["error_m"] <= GOOD_ERROR_M,
                                 "to_image": [[float(v) for v in row] for row in f["Hinv"]],
                                 "to_floor": [[float(v) for v in row] for row in f["H"]]} if f else {})}
            return {"unit_m": self.unit_m, "cameras": cams, "error": self.error}

    def save_camera(self, cam: str, points, unit_m=None) -> dict:
        """Store one camera's marks (and the unit size, which all cameras share). Marks that cannot be fitted yet
        are saved anyway (calibrating is done a mark at a time); the reply says what is missing."""
        pts = clean_points(points)
        unit = self.unit_m if unit_m is None else _num(unit_m, "tile size")
        if not 0.05 <= unit <= 10.0:
            raise FloorError("the tile size must be between 0.05 and 10 m")
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except Exception:
            raw = {}
        raw["unit_m"] = unit
        cams = raw.setdefault("cameras", {})
        if pts:
            cams[cam] = {"points": pts}
        else:
            cams.pop(cam, None)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(raw, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
        self._mtime = None
        self.reload()
        return self.status()
