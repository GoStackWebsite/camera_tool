"""Learned staff-uniform classifier: a small image classifier (YOLO11n-cls, trained by tools/staff_train.py on
labelled person crops) that tells the store's uniform from customers' clothing. The model file
models/staff_cls.pt is portable: copy it to another store of the same chain and no training is needed there.

One instance per camera pipeline (the ultralytics predictor is not thread-safe)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("staffcls")

MODEL_FILE = "staff_cls.pt"
CROP_BOTTOM = 0.62          # upper body: head to hips, same crop as the training set
IMGSZ = 128
HEAD_BAND = 0.39            # of the crop height: the head (24 % of the person box, as in pipeline.blur_heads)


def upper_body(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = y2 - y1, x2 - x1
    if h < 24 or w < 12:
        return None
    pad = int(w * 0.08)
    x1, x2 = max(0, x1 - pad), min(frame.shape[1], x2 + pad)
    y2 = min(frame.shape[0], int(y1 + h * CROP_BOTTOM))
    y1 = max(0, y1)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size and crop.shape[0] >= 16 and crop.shape[1] >= 12 else None


def mask_head(crop: np.ndarray) -> np.ndarray:
    """Pixelate the head band of an upper-body crop. The classifier judges the clothing, never the face, and
    the training crops come from face-blurred snapshots, so live crops get the same treatment."""
    h, w = crop.shape[:2]
    y2, x1, x2 = max(8, int(h * HEAD_BAND)), int(w * 0.17), int(w * 0.83)
    if x2 - x1 < 4 or y2 >= h:
        return crop
    out = crop.copy()
    roi = out[:y2, x1:x2]
    block = max(3, int(max(roi.shape[:2]) / 6))
    small = cv2.resize(roi, (max(1, roi.shape[1] // block), max(1, roi.shape[0] // block)), interpolation=cv2.INTER_LINEAR)
    out[:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return out


class StaffClassifier:
    def __init__(self, path: Path):
        from ultralytics import YOLO
        self.path = path
        self.mtime = os.path.getmtime(path)
        self.model = YOLO(str(path))
        names = self.model.names
        self.staff_idx = next((i for i, n in names.items() if str(n).lower() == "staff"), None)
        if self.staff_idx is None:
            raise ValueError(f"{path}: no 'staff' class in {names}")
        self.samples = 0

    @classmethod
    def create(cls, models_dir: Path) -> "StaffClassifier | None":
        path = Path(models_dir) / MODEL_FILE
        if not path.exists():
            return None
        try:
            c = cls(path)
            log.info("Staff classifier loaded: %s", path)
            return c
        except Exception as e:  # a broken file must not stop the camera
            log.warning("Staff classifier not loaded (%s): %s", path, e)
            return None

    def staff_prob(self, frame: np.ndarray, bbox: tuple) -> float | None:
        """Probability (0-1) that the person in bbox wears the staff uniform; None when the crop is unusable."""
        crop = upper_body(frame, bbox)
        if crop is None:
            return None
        try:
            r = self.model.predict(mask_head(crop), imgsz=IMGSZ, verbose=False)[0]
            self.samples += 1
            return float(r.probs.data[self.staff_idx])
        except Exception as e:
            log.debug("staff classifier failed: %s", e)
            return None

    def info(self) -> dict:
        return {"loaded": True, "file": str(self.path), "trained": self.mtime, "samples": self.samples}


BRAND_MODEL_FILE = "brand_cls.pt"


class BrandClassifier:
    """Learned brand of the uniform (models/brand_cls.pt, classes customer / gofix / gogizmo ..., trained by
    tools/brand_train.py). Asked only about people already recognised as staff; returns the best staff class."""

    def __init__(self, path: Path):
        from ultralytics import YOLO
        self.path = path
        self.mtime = os.path.getmtime(path)
        self.model = YOLO(str(path))
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        self.brands = [n for n in self.names.values() if n.lower() != "customer"]
        if not self.brands:
            raise ValueError(f"{path}: no brand classes in {self.names}")
        self.samples = 0

    @classmethod
    def create(cls, models_dir: Path) -> "BrandClassifier | None":
        path = Path(models_dir) / BRAND_MODEL_FILE
        if not path.exists():
            return None
        try:
            c = cls(path)
            log.info("Brand classifier loaded: %s (%s)", path, ", ".join(c.brands))
            return c
        except Exception as e:
            log.warning("Brand classifier not loaded (%s): %s", path, e)
            return None

    def brand(self, frame: np.ndarray, bbox: tuple) -> tuple[str, float] | None:
        """(brand name, probability) of the best staff class, or None when the crop is unusable."""
        crop = upper_body(frame, bbox)
        if crop is None:
            return None
        try:
            r = self.model.predict(mask_head(crop), imgsz=IMGSZ, verbose=False)[0]
            probs = r.probs.data.float().cpu().tolist()
            self.samples += 1
            best = max((i for i, n in self.names.items() if n.lower() != "customer"), key=lambda i: probs[i])
            return self.names[best], float(probs[best])
        except Exception as e:
            log.debug("brand classifier failed: %s", e)
            return None

    def read(self, frame: np.ndarray, bbox: tuple) -> tuple[float, str, float] | None:
        """One pass for everything the engine asks of this model: (P(customer), best brand, its probability)."""
        crop = upper_body(frame, bbox)
        if crop is None:
            return None
        try:
            r = self.model.predict(mask_head(crop), imgsz=IMGSZ, verbose=False)[0]
            probs = r.probs.data.float().cpu().tolist()
            self.samples += 1
            ci = next((i for i, n in self.names.items() if n.lower() == "customer"), None)
            best = max((i for i, n in self.names.items() if n.lower() != "customer"), key=lambda i: probs[i])
            return (float(probs[ci]) if ci is not None else 0.0), self.names[best], float(probs[best])
        except Exception as e:
            log.debug("brand classifier failed: %s", e)
            return None

    def customer_prob(self, frame: np.ndarray, bbox: tuple) -> float | None:
        """P(customer) from the brand model: a second, independently trained opinion on whether this is staff at all."""
        crop = upper_body(frame, bbox)
        if crop is None:
            return None
        try:
            r = self.model.predict(mask_head(crop), imgsz=IMGSZ, verbose=False)[0]
            probs = r.probs.data.float().cpu().tolist()
            idx = next((i for i, n in self.names.items() if n.lower() == "customer"), None)
            return float(probs[idx]) if idx is not None else None
        except Exception:
            return None

    def info(self) -> dict:
        return {"loaded": True, "file": str(self.path), "trained": self.mtime, "brands": self.brands, "samples": self.samples}
