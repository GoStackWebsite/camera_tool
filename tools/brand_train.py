"""Train the three-way uniform classifier: customer / GoFix staff / GoGizmo staff (any brand names work).

The model learns the whole upper body (print text on the back, logo on the chest, cut of the tee), which is far
more robust than counting logo-coloured pixels. The engine (app/staffcls.py) uses it for both staff detection
(1 - P(customer)) and the brand (best staff class) when it is installed as models/staff_cls.pt.

    .venv\\Scripts\\python tools\\brand_train.py --build labels.csv --frames <dir> [--labels2 other.csv --frames2 <dir>]
    .venv\\Scripts\\python tools\\brand_train.py --train [--epochs 60]
    .venv\\Scripts\\python tools\\brand_train.py --eval

labels.csv: file,x1,y1,x2,y2,label  with label in customer | <brand name> (e.g. gofix, gogizmo); optional split
column (train|val). Crops are made exactly like the engine's (upper body, head band pixelated)."""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "datasets" / "brand"
MODEL = ROOT / "models" / "brand_cls.pt"


def build(sets: list[tuple[Path, Path]], val_every: int = 5, seed: int = 7) -> None:
    import cv2
    from app.staffcls import mask_head, upper_body
    if DATA.exists():
        shutil.rmtree(DATA)
    n: dict[str, int] = {}
    for labels_csv, frames_dir in sets:
        rows = list(csv.DictReader(open(labels_csv, encoding="utf-8")))
        files = sorted({r["file"] for r in rows})
        random.Random(seed).shuffle(files)
        val_files = set(files[::val_every])
        for i, r in enumerate(rows):
            label = r["label"].strip().lower()
            if not label:
                continue
            im = cv2.imread(str(frames_dir / r["file"]))
            if im is None:
                continue
            crop = upper_body(im, tuple(int(float(r[k])) for k in ("x1", "y1", "x2", "y2")))
            if crop is None:
                continue
            split = (r.get("split") or "").strip().lower()
            if split not in ("train", "val"):
                split = "val" if r["file"] in val_files else "train"
            out = DATA / split / label
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{Path(r['file']).stem}_{i}.jpg"), mask_head(crop), [cv2.IMWRITE_JPEG_QUALITY, 95])
            n[f"{split}/{label}"] = n.get(f"{split}/{label}", 0) + 1
    print("dataset:", dict(sorted(n.items())))


def train(epochs: int, imgsz: int = 128, out: Path = MODEL) -> None:
    from ultralytics import YOLO
    model = YOLO("yolo11n-cls.pt")
    # workers=2: ultralytics otherwise starts one data-loader process per CPU core (~640 MB each); two keep a retrain
    # on the store PC from competing with the running camera engine for memory
    res = model.train(data=str(DATA), epochs=epochs, imgsz=imgsz, batch=32, workers=2, fliplr=0.5, degrees=5, scale=0.3, hsv_h=0.0,
                      hsv_s=0.3, hsv_v=0.4, erasing=0.2, project=str(ROOT / "runs"), name="brand_cls", exist_ok=True, verbose=False, plots=False)
    out.parent.mkdir(exist_ok=True)
    shutil.copy(Path(res.save_dir) / "weights" / "best.pt", out)
    print("model saved:", out)
    evaluate(out)


def evaluate(path: Path = MODEL) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(path))
    names = {int(k): str(v) for k, v in model.names.items()}
    labels = sorted(p.name for p in (DATA / "val").iterdir() if p.is_dir())
    conf = {a: {b: 0 for b in labels} for a in labels}
    for label in labels:
        for p in sorted((DATA / "val" / label).glob("*.jpg")):
            r = model.predict(str(p), imgsz=128, verbose=False)[0]
            conf[label][names[int(r.probs.top1)]] += 1
    print("confusion (rows = truth, columns = predicted):", labels)
    for a in labels:
        print(f"  {a:10s}", [conf[a][b] for b in labels])
    staff = [l for l in labels if l != "customer"]
    tp = sum(conf[a][b] for a in staff for b in staff)
    fn = sum(conf[a]["customer"] for a in staff)
    fp = sum(conf["customer"][b] for b in staff) if "customer" in conf else 0
    brand_ok = sum(conf[a][a] for a in staff)
    out = {"staff_recall": tp / (tp + fn) if tp + fn else None, "staff_precision": tp / (tp + fp) if tp + fp else None,
           "brand_accuracy": brand_ok / tp if tp else None}
    print({k: (round(v, 3) if v is not None else None) for k, v in out.items()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", metavar="LABELS_CSV")
    ap.add_argument("--frames", metavar="DIR")
    ap.add_argument("--labels2", metavar="LABELS_CSV")
    ap.add_argument("--frames2", metavar="DIR")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", metavar="FILE", help="train: write the model here instead of models/brand_cls.pt")
    ap.add_argument("--model", metavar="FILE", help="eval: the model to evaluate (default models/brand_cls.pt)")
    a = ap.parse_args()
    if a.build:
        sets = [(Path(a.build), Path(a.frames or "."))]
        if a.labels2:
            sets.append((Path(a.labels2), Path(a.frames2 or a.frames or ".")))
        build(sets)
    if a.train:
        train(a.epochs, out=Path(a.out) if a.out else MODEL)
    if a.eval and not a.train:
        evaluate(Path(a.model) if a.model else MODEL)
