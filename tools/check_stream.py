"""Quick RTSP connectivity test.

    python tools/check_stream.py            # uses RTSP_URL/RTSP_USER/RTSP_PASS from .env
    python tools/check_stream.py --url rtsp://user:pass@host:554/path --seconds 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: F401  (sets FFmpeg RTSP options before cv2 import)
from app.config import DATA_DIR, mask_url, settings

import cv2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=settings.rtsp_url)
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    print(f"Opening {mask_url(args.url)} ...")
    t0 = time.time()
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"FAILED to open after {time.time() - t0:.1f}s.")
        print("Check RTSP_USER / RTSP_PASS in .env, the stream path, and that port 554 is reachable.")
        return 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opened in {time.time() - t0:.1f}s: {w}x{h} @ {fps:.1f} fps (reported by stream)")

    frames = 0
    start = time.time()
    last = None
    while time.time() - start < args.seconds:
        ok, frame = cap.read()
        if not ok:
            print("read() failed")
            break
        frames += 1
        last = frame
    elapsed = max(time.time() - start, 1e-6)
    print(f"Received {frames} frames in {elapsed:.1f}s = {frames / elapsed:.1f} fps")
    if last is not None:
        out = DATA_DIR / "snapshot_test.jpg"
        cv2.imwrite(str(out), last)
        print(f"Saved snapshot to {out}")
    cap.release()
    return 0 if frames else 2


if __name__ == "__main__":
    sys.exit(main())
