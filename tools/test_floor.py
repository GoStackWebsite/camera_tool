"""Floor plan calibration and linking people by where they stand: python tools/test_floor.py

No camera needed. Two simulated ceiling cameras (a real perspective projection of the floor) look at one shop
floor from different corners; their pictures of floor-tile corners are the calibration marks. Checked: the fitted
mapping puts picture points back on the floor to within centimetres, bad calibrations are refused or flagged, and
store fusion links a person seen by both cameras by their floor position alone, without any shared region."""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import analytics as A                                   # noqa: E402
from app.config import CameraConfig, StaffConfig, Thresholds     # noqa: E402
from app.detector import PERSON_CLASS, Detection                 # noqa: E402
from app.floor import FloorError, FloorMap, clean_points, fit    # noqa: E402
from app.fusion import StoreFusion                               # noqa: E402

W, H = 1280, 720
TILE = 0.6


class Cam:
    """A pinhole camera at `pos` (metres, z up) looking at floor point `look`; maps floor (m) -> picture fractions."""

    def __init__(self, pos, look, f=700.0):
        c, t = np.array(pos, float), np.array([look[0], look[1], 0.0])
        z = (t - c) / np.linalg.norm(t - c)                      # viewing direction
        x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)     # picture right
        y = np.cross(z, x)                                       # picture down
        R = np.stack([x, y, z])
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        self.Hf = K @ np.column_stack([R[:, 0], R[:, 1], -R @ c])   # floor plane (z = 0) -> pixels

    def img(self, X, Y):
        p = self.Hf @ [X, Y, 1.0]
        return [p[0] / p[2] / W, p[1] / p[2] / H]


CAM_A = Cam((0.0, 0.0, 2.8), (3.0, 2.5))       # overview camera in one corner
CAM_B = Cam((6.0, 0.5, 2.6), (3.5, 2.5))       # billing camera in the other


def marks(cam: Cam, tiles, noise_px: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for tx, ty in tiles:
        x, y = cam.img(tx * TILE, ty * TILE)
        out.append({"img": [x + rng.normal(0, noise_px) / W, y + rng.normal(0, noise_px) / H], "floor": [tx, ty]})
    return out


TILES_A = [(2, 2), (6, 2), (2, 6), (6, 6), (4, 4), (3, 7)]
TILES_B = [(3, 2), (8, 2), (3, 6), (8, 6), (5, 4), (7, 7)]

results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


print("== fitting one camera")
for cam, tiles, label in ((CAM_A, TILES_A, "A"), (CAM_B, TILES_B, "B")):
    for tx, ty in tiles:
        x, y = cam.img(tx * TILE, ty * TILE)
        assert 0 < x < 1 and 0 < y < 1, f"camera {label} does not see tile {tx},{ty}: test set-up"
f = fit(clean_points(marks(CAM_A, TILES_A)), TILE)
probe = [(1.7, 1.9), (3.1, 2.6), (2.4, 3.9), (3.6, 4.1)]
worst = 0.0
fm = FloorMap(Path(tempfile.mkdtemp()) / "floor.json")
fm.unit_m, fm.fits = TILE, {"A": f}
for X, Y in probe:
    got = fm.to_floor("A", *CAM_A.img(X, Y))
    worst = max(worst, math.dist(got, (X, Y)))
check("exact marks: any floor point comes back within 1 cm", worst < 0.01, f"worst {worst * 100:.2f} cm")
check("six marks: the calibration is checked, error about 0", f["checked"] and f["error_m"] < 0.005, f"error {f['error_m']:.4f} m")

f = fit(clean_points(marks(CAM_A, TILES_A, noise_px=2.0, seed=3)), TILE)
check("clicks 2 px off: still within 10 cm on average", f["error_m"] < 0.10, f"error {f['error_m'] * 100:.1f} cm")

f = fit(clean_points(marks(CAM_A, TILES_A[:4])), TILE)
check("four marks: usable, but not checked yet", not f["checked"], str(f["checked"]))

bad = marks(CAM_A, TILES_A)
bad[4]["floor"], bad[5]["floor"] = bad[5]["floor"], bad[4]["floor"]       # two floor positions typed the wrong way round
try:
    f = fit(clean_points(bad), TILE)
    check("two marks swapped: flagged by a large error at one of them", f["error_m"] > 0.25 and f["worst"] in (4, 5),
          f"error {f['error_m']:.2f} m, worst mark {f['worst'] + 1}")
except FloorError as e:
    check("two marks swapped: refused", True, str(e))

line = [{"img": [0.2 + 0.1 * i, 0.5 + 0.05 * i], "floor": [i, i + 1]} for i in range(5)]
try:
    fit(clean_points(line), TILE)
    check("marks on one line in the picture: refused", False)
except FloorError as e:
    check("marks on one line in the picture: refused", "line" in str(e), str(e))

try:
    clean_points([{"img": [0.2, 0.2], "floor": [1, 1]}, {"img": [0.3, 0.4], "floor": [1, 1]}])
    check("two marks with one floor position: refused", False)
except FloorError as e:
    check("two marks with one floor position: refused", True, str(e))

fm.fits = {"A": fit(clean_points(marks(CAM_A, TILES_A)), TILE)}
Hf = fm.fits["A"]["H"]
yh = -(Hf[2, 0] * 0.5 + Hf[2, 2]) / Hf[2, 1]                  # where the floor's horizon crosses the middle of the picture
below, above = fm.to_floor("A", 0.5, yh + 0.05), fm.to_floor("A", 0.5, yh - 0.05)
check("a point beyond the floor's horizon is not on the floor", above is None and below is not None, f"horizon y={yh:.2f} above={above}")

print("== saving and reading back")
d = Path(tempfile.mkdtemp())
fm = FloorMap(d / "floor.json")
st = fm.save_camera("A", marks(CAM_A, TILES_A[:3]), 0.6)
check("three marks are saved but not usable yet", not st["cameras"]["A"]["ok"] and "4 marks" in st["cameras"]["A"]["message"], st["cameras"]["A"]["message"])
st = fm.save_camera("A", marks(CAM_A, TILES_A), 0.6)
c = st["cameras"]["A"]
Hinv = np.array(c["to_image"])
p = Hinv @ [1.2, 3.0, 1.0]
check("the page gets floor -> picture to draw the tile grid", math.dist([p[0] / p[2], p[1] / p[2]], CAM_A.img(1.2, 3.0)) < 1e-3)
fm2 = FloorMap(d / "floor.json")
check("a new reader sees the same calibration", fm2.calibrated("A") and abs(fm2.unit_m - 0.6) < 1e-9)
st = fm.save_camera("A", [], None)
check("no marks: the camera is no longer calibrated", "A" not in st["cameras"] and not fm.calibrated("A"))
try:
    fm.save_camera("A", marks(CAM_A, TILES_A), 0.0)
    check("tile size 0 refused", False)
except FloorError as e:
    check("tile size 0 refused", True, str(e))


# ---------------------------------------------------------------- store fusion on the floor plan
class FakeStore:
    def __getattr__(self, name):
        return lambda *a, **k: None


class Shop:
    def __init__(self, calibrate=("A", "B")):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "fusion.json").write_text(json.dumps({"primary": "A", "regions": []}), encoding="utf-8")
        fm = FloorMap(self.dir / "floor.json")
        for cam, c, tiles in (("A", CAM_A, TILES_A), ("B", CAM_B, TILES_B)):
            if cam in calibrate:
                fm.save_camera(cam, marks(c, tiles), TILE)
        self.fusion = StoreFusion(self.dir / "fusion.json")
        self.cams = {"A": CAM_A, "B": CAM_B}
        self.an = {c: A.Analytics(CameraConfig(id=c, name=c, url="", zones=[]), Thresholds(), FakeStore(), None, StaffConfig())
                   for c in ("A", "B")}
        for an in self.an.values():
            an.fusion = self.fusion
        self.t = 1000.0

    def run(self, seconds, people, step=0.1, cut=()):
        """people: camera -> [(track id, floor X, floor Y)]; `cut`: (camera, track id) whose box the frame bottom cuts."""
        frame = np.full((H, W, 3), 90, np.uint8)
        end = self.t + seconds
        while self.t < end:
            for cam, an in self.an.items():
                dets = []
                for tid, X, Y in people.get(cam, []):
                    x, y = self.cams[cam].img(X, Y)
                    cx, y2 = int(x * W), int(y * H)
                    if (cam, tid) in cut:
                        y2 = H
                    dets.append(Detection(tid, PERSON_CLASS, cx - 30, y2 - 200, cx + 30, y2, 0.9))
                an.update(dets, W, H, self.t, None, frame)
            self.fusion.tick(self.an, self.t)
            self.t += step


print("== linking people by where they stand")
s = Shop()
check("fusion is on with a floor plan and no shared regions", s.fusion.active)
s.run(2.0, {"A": [(10, 3.0, 3.0)], "B": [(70, 3.05, 2.95)]})
b = s.an["B"].tracks[70]
check("one person, both cameras: the second camera shows the overview id", b.store_id == 10, f"store_id={b.store_id}")
links = s.fusion.status()["links"]
check("the link says it was made on the floor plan", links and links[0]["region"] == "floor plan" and links[0]["floor_m"] < 0.2, str(links))

s = Shop()
s.run(2.0, {"A": [(11, 2.5, 3.0), (12, 3.7, 3.0)], "B": [(71, 3.7, 3.0), (72, 2.5, 3.0)]})
b1, b2 = s.an["B"].tracks[71], s.an["B"].tracks[72]
check("two people 1.2 m apart: each linked to the right one", b1.store_id == 12 and b2.store_id == 11, f"71->{b1.store_id} 72->{b2.store_id}")

s = Shop()
s.run(2.0, {"A": [(13, 2.0, 2.0)], "B": [(73, 4.0, 3.5)]})
check("two different people 2.5 m apart: not linked", s.an["B"].tracks[73].store_id is None)

s = Shop()
s.run(2.0, {"A": [(14, 3.0, 3.0), (15, 3.4, 3.0)], "B": [(74, 3.2, 3.0)]})
check("one person between two others 0.4 m apart: not guessed", s.an["B"].tracks[74].store_id is None,
      f"store_id={s.an['B'].tracks[74].store_id}")

s = Shop()
s.run(2.0, {"A": [(16, 3.0, 3.0)], "B": [(76, 3.0, 3.0)]}, cut={("B", 76)})
check("box cut by the frame bottom: its foot is not on the floor, no floor link", s.an["B"].tracks[76].store_id is None)

s = Shop(calibrate=("A",))
s.run(2.0, {"A": [(17, 3.0, 3.0)], "B": [(77, 3.0, 3.0)]})
check("only one camera calibrated: no floor links, fusion stays off", s.an["B"].tracks[77].store_id is None and not s.fusion.active)

s = Shop()
s.run(2.0, {"A": [(18, 3.0, 3.0)], "B": [(78, 3.0, 3.0)]})
s.run(4.0, {"A": [(18, 3.0, 3.0)], "B": [(78, 5.0, 3.0)]})
check("they walk 2 m apart: the link drops", s.an["B"].tracks[78].store_id is None, f"store_id={s.an['B'].tracks[78].store_id}")

pos = s.fusion.positions
check("live floor positions for the page", "A" in pos and "B" in pos and abs(pos["B"][0]["floor"][0] - 5.0) < 0.05, json.dumps(pos))

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
