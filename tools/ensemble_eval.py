"""How do the staff model, the brand model and their average score the validation crops, per capture day?
python tools/ensemble_eval.py [staff.pt brand.pt]   (uses datasets/staff/val built by tools/staff_train.py --build)"""
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "datasets" / "staff" / "val"
staff = YOLO(sys.argv[1] if len(sys.argv) > 2 else str(ROOT / "models" / "staff_cls.pt"))
brand = YOLO(sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "models" / "brand_cls.pt"))
si = {v: k for k, v in staff.names.items()}["staff"]
ci = {v: k for k, v in brand.names.items()}["customer"]
res = defaultdict(list)
for label in ("staff", "customer"):
    for p in sorted((VAL / label).glob("*.jpg")):
        ps = float(staff.predict(str(p), imgsz=128, verbose=False)[0].probs.data[si])
        pb = 1.0 - float(brand.predict(str(p), imgsz=128, verbose=False)[0].probs.data[ci])
        res[label].append((p.name[:10], ps, pb, (ps + pb) / 2))
for label, rows in res.items():
    n = len(rows)
    print(f"{label} crops: {n}")
    for i, name in ((1, "staff model"), (2, "brand model"), (3, "average of both")):
        hi = sum(r[i] >= 0.8 for r in rows)
        mid = sum(0.5 <= r[i] < 0.8 for r in rows)
        lo = sum(r[i] < 0.5 for r in rows)
        print(f"   {name:16s} >=0.8: {hi:3d} ({hi / n:4.0%})   0.5-0.8: {mid:3d}   <0.5: {lo:3d} ({lo / n:4.0%})")
    for day in sorted({r[0] for r in rows}):
        d = [r for r in rows if r[0] == day]
        print(f"      {day}: average of both >=0.8: {sum(r[3] >= 0.8 for r in d)}/{len(d)}, mean {sum(r[3] for r in d) / len(d):.2f}")
