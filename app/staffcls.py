"""Learned staff-uniform classifier: a small image classifier (YOLO11n-cls, trained by tools/staff_train.py on
labelled person crops) that tells the store's uniform from customers' clothing. The model file
models/staff_cls.pt is portable: copy it to another store of the same chain and no training is needed there.

One instance per camera pipeline (the ultralytics predictor is not thread-safe)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("staffcls")

MODEL_FILE = "staff_cls.pt"
CROP_BOTTOM = 0.62          # upper body: head to hips, same crop as the training set
IMGSZ = 128


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
            r = self.model.predict(crop, imgsz=IMGSZ, verbose=False)[0]
            self.samples += 1
            return float(r.probs.data[self.staff_idx])
        except Exception as e:
            log.debug("staff classifier failed: %s", e)
            return None

    def info(self) -> dict:
        return {"loaded": True, "file": str(self.path), "trained": self.mtime, "samples": self.samples}
