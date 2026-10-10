"""YOLO detection of people (tracked with ByteTrack) and hand-held phones."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from .config import BASE_DIR

log = logging.getLogger("detector")
TRACKER_CFG = BASE_DIR / "app" / "tracker.yaml"
PERSON_CLASS = 0
PHONE_CLASS = 67   # COCO "cell phone"


@dataclass(slots=True)
class Detection:
    track_id: int
    cls: int
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    @property
    def is_person(self) -> bool:
        return self.cls == PERSON_CLASS


class PersonTracker:
    def __init__(self, model_name: str, device: str = "auto", conf: float = 0.35, imgsz: int = 640):
        if device == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.device_label = torch.cuda.get_device_name(0) if device != "cpu" else "CPU"
        self.conf = conf
        self.imgsz = imgsz
        self.quantize = 16 if device != "cpu" else None  # fp16 on GPU
        self.model_name = model_name
        log.info("Loading %s on %s", model_name, self.device_label)
        self.model = YOLO(model_name)
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), device=self.device, imgsz=imgsz,
                           quantize=self.quantize, verbose=False)  # warm-up
        log.info("Model ready")

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS, PHONE_CLASS],
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            quantize=self.quantize,
            tracker=str(TRACKER_CFG),
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        return [
            Detection(int(i), int(k), int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(c))
            for b, i, k, c in zip(xyxy, ids, clss, confs)
        ]
