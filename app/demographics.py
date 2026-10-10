"""Anonymous age / gender estimation: YuNet face detector (OpenCV) + InsightFace gender-age model (ONNX).

Faces are only used in memory to produce an age band and gender per tracked visit; no face image
or embedding is ever stored.
"""
from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("demographics")

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_FILE
GENDERAGE_FILE = "genderage.onnx"
PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
INPUT = 96
CANVAS = 256   # detector input: each person region is letter-boxed into this square


def ensure_models(models_dir: Path) -> bool:
    """Download the two model files if they are missing. Returns True when both are present."""
    models_dir.mkdir(exist_ok=True)
    yunet = models_dir / YUNET_FILE
    ga = models_dir / GENDERAGE_FILE
    try:
        if not yunet.exists():
            log.info("Downloading face detector (%s)", YUNET_URL)
            urllib.request.urlretrieve(YUNET_URL, yunet)
        if not ga.exists():
            log.info("Downloading gender-age model pack (%s, ~120 MB)", PACK_URL)
            data = urllib.request.urlopen(PACK_URL, timeout=120).read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                ga.write_bytes(z.read(GENDERAGE_FILE))
    except Exception as e:
        log.warning("Could not download demographics models: %s", e)
    return yunet.exists() and ga.exists()


class Demographics:
    """One instance per camera (the face detector keeps the input size as state)."""

    def __init__(self, models_dir: Path):
        import onnxruntime as ort  # imported here so the rest of the app works without it

        cv2.setNumThreads(2)   # keep the face detector from starving the main detection loop
        self.det = cv2.FaceDetectorYN.create(str(models_dir / YUNET_FILE), "", (320, 320), 0.6, 0.3, 50)
        self.size = (0, 0)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(models_dir / GENDERAGE_FILE), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.faces_seen = 0

    @classmethod
    def create(cls, models_dir: Path) -> "Demographics | None":
        if not ensure_models(models_dir):
            log.warning("Demographics disabled: model files missing in %s", models_dir)
            return None
        try:
            d = cls(models_dir)
            log.info("Demographics ready (YuNet + gender-age)")
            return d
        except Exception as e:
            log.warning("Demographics disabled: %s", e)
            return None

    def detect(self, frame: np.ndarray, boxes: dict[int, tuple], min_face: int = 24) -> list[dict]:
        """Look for a face in the upper part of each person box {track_id: (x1, y1, x2, y2)}.

        Each region is letter-boxed into a fixed 256x256 canvas (constant detector input, small faces
        get upscaled). Returns [{"track_id", "gender" (0 = female, 1 = male), "age", "size", "score", "box"}].
        """
        out: list[dict] = []
        if not boxes:
            return out
        H, W = frame.shape[:2]
        if self.size != (CANVAS, CANVAS):
            self.det.setInputSize((CANVAS, CANVAS))
            self.size = (CANVAS, CANVAS)
        candidates: list[tuple] = []
        for tid, (x1, y1, x2, y2) in boxes.items():
            bw, bh = x2 - x1, y2 - y1
            if bw < min_face or bh < min_face:
                continue
            px, py = int(bw * 0.15), int(bh * 0.1)
            cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
            cx2, cy2 = min(W, x2 + px), min(H, y1 + int(bh * 0.6))
            cw, ch = cx2 - cx1, cy2 - cy1
            if cw < 16 or ch < 16:
                continue
            region = frame[cy1:cy2, cx1:cx2]
            s = min(CANVAS / cw, CANVAS / ch)
            rw, rh = max(1, int(cw * s)), max(1, int(ch * s))
            canvas = np.zeros((CANVAS, CANVAS, 3), np.uint8)
            canvas[:rh, :rw] = cv2.resize(region, (rw, rh), interpolation=cv2.INTER_LINEAR)
            _, faces = self.det.detect(canvas)
            if faces is None or len(faces) == 0:
                continue
            best = None
            for f in sorted(faces, key=lambda r: -r[14]):
                x, y, fw, fh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])
                if fw / s < min_face or fh / s < min_face:
                    continue
                # the face must belong to THIS person: upper 40% of the box, central 70%, plausible size
                fcx, fcy = cx1 + (x + fw / 2) / s, cy1 + (y + fh / 2) / s
                if not (x1 + bw * 0.15 <= fcx <= x2 - bw * 0.15 and y1 <= fcy <= y1 + bh * 0.4 and fw / s <= bw * 0.6):
                    continue
                best = (x, y, fw, fh, score, fcx, fcy)
                break
            if best is None:
                continue
            x, y, fw, fh, score, fcx, fcy = best
            candidates.append((tid, canvas, x, y, fw, fh, score, fcx, fcy, s, cx1, cy1, bw))
        # one face can only belong to one person: if two tracks picked the same face, keep the narrower box
        candidates.sort(key=lambda c: c[12])
        taken: list[tuple[float, float]] = []
        for tid, canvas, x, y, fw, fh, score, fcx, fcy, s, cx1, cy1, bw in candidates:
            if any(abs(fcx - ox) < 12 and abs(fcy - oy) < 12 for ox, oy in taken):
                continue
            taken.append((fcx, fcy))
            crop = self._crop(canvas, x, y, fw, fh)
            blob = cv2.dnn.blobFromImage(crop, 1.0, (INPUT, INPUT), (0, 0, 0), swapRB=True)
            pred = self.sess.run(None, {self.input_name: blob})[0][0]
            self.faces_seen += 1
            out.append({"track_id": tid, "gender": int(np.argmax(pred[:2])), "age": float(pred[2] * 100),
                        "size": fw / s, "score": score,
                        "box": (cx1 + int(x / s), cy1 + int(y / s), cx1 + int((x + fw) / s), cy1 + int((y + fh) / s))})
        return out

    @staticmethod
    def _crop(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
        """Same crop InsightFace uses for its attribute model: centred, 1.5x the face box, scaled to 96 px."""
        cx, cy = x + w / 2, y + h / 2
        s = INPUT / (max(w, h) * 1.5)
        m = np.array([[s, 0, -cx * s + INPUT / 2], [0, s, -cy * s + INPUT / 2]], np.float32)
        return cv2.warpAffine(img, m, (INPUT, INPUT), borderValue=0.0)
