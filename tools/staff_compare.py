"""Compare staff models on the validation crops, per capture day: python tools/staff_compare.py a.pt b.pt
Prints staff recall and customer false positives per day, and lists each model's wrong crops."""
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "datasets" / "staff" / "val"


def run(path: str) -> dict:
    model = YOLO(path)
    names = model.names
    out = defaultdict(lambda: {"staff": [0, 0], "customer": [0, 0], "wrong": []})
    for label in ("staff", "customer"):
        for p in sorted((VAL / label).glob("*.jpg")):
            r = model.predict(str(p), imgsz=int(model.overrides.get("imgsz") or 128), verbose=False)[0]
            probs = {names[i]: float(v) for i, v in enumerate(r.probs.data.tolist())}
            day = p.name[:10]
            ok = names[int(r.probs.top1)] == label
            out[day][label][0] += ok
            out[day][label][1] += 1
            if not ok:
                out[day]["wrong"].append(f"{label}:{p.name} P(staff)={probs.get('staff', 0):.2f}")
    return out


for path in sys.argv[1:]:
    res = run(path)
    print(f"\n{path}")
    for day in sorted(res):
        s, c = res[day]["staff"], res[day]["customer"]
        print(f"  {day}: staff found {s[0]}/{s[1]}, customers kept as customers {c[0]}/{c[1]}")
        for w in res[day]["wrong"]:
            print("      wrong ->", w)
