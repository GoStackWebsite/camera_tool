"""Train the staff-uniform classifier from labelled person crops.

Dataset layout (built by `--build` from a labels CSV, or prepared by hand / the MCP server):
    datasets/staff/train/{staff,customer}/*.jpg
    datasets/staff/val/{staff,customer}/*.jpg
Each image is the upper body of one person (head to hips), any size.

    .venv\\Scripts\\python tools\\staff_train.py --build labels.csv --frames <dir>   # crops -> dataset
    .venv\\Scripts\\python tools\\staff_train.py --train [--epochs 40] [--imgsz 128]  # -> models/staff_cls.pt
    .venv\\Scripts\\python tools\\staff_train.py --eval                                # accuracy on val/

labels.csv columns: file, x1, y1, x2, y2 (pixel box of the person in <frames>/<file>), label (staff|customer),
and optionally split (train|val). `file` may contain sub-folders (2026-10-12/cam31-....jpg). Without a split
column every fifth frame goes to validation; with one, the caller decides (e.g. the last part of each day, so
validation people are not the ones trained on). Label only what is clearly the uniform or clearly not:
ambiguous dark tops are better left out than guessed.
The model is a YOLO11n classification network fine-tuned on these crops; the engine loads models/staff_cls.pt
automatically (Settings › Staff detection shows "learned model"). Copy that one file to another store of the
same chain and no training is needed there; a chain with a different uniform re-runs --build/--train on its
own captures (the MCP server tool `train_staff_model` does this end to end).
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "datasets" / "staff"
MODEL = ROOT / "models" / "staff_cls.pt"
CROP_TOP, CROP_BOTTOM = 0.0, 0.62          # upper body: head to hips (the uniform tee, print on front or back)


def upper_body(im, box):
    """Exactly the crop the engine classifies (app/staffcls.py), head band pixelated."""
    from app.staffcls import mask_head, upper_body as engine_crop
    crop = engine_crop(im, box)
    return mask_head(crop) if crop is not None else im[0:0, 0:0]


def build(labels_csv: Path, frames_dir: Path, val_every: int = 5, seed: int = 7) -> None:
    import cv2
    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8")))
    if DATA.exists():
        shutil.rmtree(DATA)
    files = sorted({r["file"] for r in rows})
    random.Random(seed).shuffle(files)
    val_files = set(files[::val_every])                 # split by frame so the same person/pose is not in both
    n = {"train": 0, "val": 0}
    for i, r in enumerate(rows):
        label = r["label"].strip().lower()
        if label not in ("staff", "customer"):
            continue
        im = cv2.imread(str(frames_dir / r["file"]))
        if im is None:
            continue
        crop = upper_body(im, tuple(int(float(r[k])) for k in ("x1", "y1", "x2", "y2")))
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 12:
            continue
        split = (r.get("split") or "").strip().lower()
        if split not in ("train", "val"):
            split = "val" if r["file"] in val_files else "train"
        out = DATA / split / label
        out.mkdir(parents=True, exist_ok=True)
        day = Path(r["file"]).parent.name
        cv2.imwrite(str(out / f"{day + '_' if day else ''}{Path(r['file']).stem}_{i}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n[split] += 1
    print(f"dataset built: {n['train']} train, {n['val']} val crops in {DATA}")


def train(epochs: int, imgsz: int) -> None:
    from ultralytics import YOLO
    model = YOLO("yolo11n-cls.pt")
    # workers=2: the default starts one loader process per CPU core (about 640 MB each, 10 GB on this PC), which
    # exhausted the machine's memory and killed the running camera engine three times on 2026-10-12
    res = model.train(data=str(DATA), epochs=epochs, imgsz=imgsz, batch=32, workers=2, fliplr=0.5, degrees=5, scale=0.3,
                      hsv_h=0.01, hsv_s=0.4, hsv_v=0.4, erasing=0.2, project=str(ROOT / "runs"), name="staff_cls", exist_ok=True,
                      verbose=False, plots=False)
    best = Path(res.save_dir) / "weights" / "best.pt"
    MODEL.parent.mkdir(exist_ok=True)
    shutil.copy(best, MODEL)
    print(f"model saved: {MODEL}")
    evaluate()


def evaluate() -> dict:
    from ultralytics import YOLO
    model = YOLO(str(MODEL))
    names = model.names
    stats = {}
    for label in ("staff", "customer"):
        imgs = sorted((DATA / "val" / label).glob("*.jpg"))
        if not imgs:
            continue
        ok = 0
        for p in imgs:
            r = model.predict(str(p), imgsz=128, verbose=False)[0]
            pred = names[int(r.probs.top1)]
            ok += pred == label
        stats[label] = {"n": len(imgs), "correct": ok}
    s, c = stats.get("staff", {"n": 0, "correct": 0}), stats.get("customer", {"n": 0, "correct": 0})
    recall = s["correct"] / s["n"] if s["n"] else 0.0
    fp = c["n"] - c["correct"]
    precision = s["correct"] / (s["correct"] + fp) if s["correct"] + fp else 0.0
    print(f"val: staff recall {recall:.0%} ({s['correct']}/{s['n']}), precision {precision:.0%}, customer specificity {c['correct']}/{c['n']}")
    return {"recall": recall, "precision": precision, **stats}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", metavar="LABELS_CSV")
    ap.add_argument("--frames", metavar="DIR")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=128)
    a = ap.parse_args()
    if a.build:
        build(Path(a.build), Path(a.frames or "."))
    if a.train:
        train(a.epochs, a.imgsz)
    if a.eval and not a.train:
        evaluate()
    if not (a.build or a.train or a.eval):
        ap.print_help()
        sys.exit(1)
