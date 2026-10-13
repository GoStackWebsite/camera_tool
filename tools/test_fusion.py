"""Tests for store fusion (two cameras of one store seeing the same people): python tools/test_fusion.py

No camera and no trained model needed: two Analytics instances are fed synthetic detections, a fake uniform
model answers what each scenario dictates, and app/fusion.py links the people standing in the shared regions."""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import analytics as A                                              # noqa: E402
from app.config import CameraConfig, StaffConfig, Thresholds, ZoneConfig    # noqa: E402
from app.congruence import CongruenceConnector                              # noqa: E402
from app.detector import PERSON_CLASS, Detection                            # noqa: E402
from app.fusion import StoreFusion, colour_cost                             # noqa: E402

W, H = 1280, 720
SEAT_A = [[0.0, 0.6], [0.24, 0.6], [0.24, 1.0], [0.0, 1.0]]          # overview camera: the cashier's seat
FRONT_A = [[0.24, 0.6], [0.6, 0.6], [0.6, 1.0], [0.24, 1.0]]         # ... and the floor in front of the billing desk
DESK_B = [[0.0, 0.1], [0.52, 0.1], [0.52, 1.0], [0.0, 1.0]]          # billing camera: behind the desk
FRONT_B = [[0.53, 0.25], [1.0, 0.25], [1.0, 1.0], [0.53, 1.0]]       # ... and in front of it
A_FRONT, A_SEAT, A_FLOOR = (0.42, 0.90), (0.10, 0.95), (0.75, 0.55)   # foot points on the overview camera
B_FRONT, B_DESK = (0.75, 0.95), (0.30, 0.90)                          # ... and on the billing camera
RED, WHITE = (40, 40, 200), (235, 235, 235)                           # BGR clothing colours


class FakeModel:
    def __init__(self):
        self.by_x: dict[int, float | None] = {}

    def staff_prob(self, frame, bbox):
        return self.by_x.get(bbox[0])

    def info(self):
        return {"loaded": True}


class FakeStore:
    def __init__(self):
        self.visits: list[dict] = []

    def add_visit(self, v):
        self.visits.append(v)

    def add_event(self, e): pass
    def add_crossing(self, c): pass
    def add_zone_visit(self, z): pass
    def add_minute(self, *a, **k): pass
    def add_queue_minute(self, *a, **k): pass


class Shop:
    """Two cameras of one store and the fusion between them."""

    def __init__(self):
        self.dir = tempfile.TemporaryDirectory()
        cfg = Path(self.dir.name) / "fusion.json"
        cfg.write_text(json.dumps({"primary": "A", "regions": [
            {"name": "Billing seat", "views": {"A": SEAT_A, "B": DESK_B}},
            {"name": "Billing front", "views": {"A": FRONT_A, "B": FRONT_B}}]}), encoding="utf-8")
        self.fusion = StoreFusion(cfg)
        self.models = {"A": FakeModel(), "B": FakeModel()}
        self.stores = {"A": FakeStore(), "B": FakeStore()}
        cams = {"A": CameraConfig(id="A", name="Shop floor", url="", zones=[ZoneConfig("Billing desk (staff)", SEAT_A, "staff")]),
                "B": CameraConfig(id="B", name="Billing desk", url="", zones=[ZoneConfig("Staff desk", DESK_B, "staff"),
                                                                               ZoneConfig("Billing counter", FRONT_B, "queue")])}
        self.an = {c: A.Analytics(cams[c], Thresholds(), self.stores[c], None, StaffConfig(), self.models[c], None) for c in cams}
        for an in self.an.values():
            an.fusion = self.fusion
        self.t = 1000.0

    def run(self, seconds: float, people: dict, step: float = 0.1) -> None:
        """people: camera -> list of (track id, foot, P(staff) or None, clothing colour or None, box height)."""
        end = self.t + seconds
        while self.t < end:
            for cam, an in self.an.items():
                frame = np.full((H, W, 3), 90, np.uint8)
                dets = []
                for tid, foot, prob, colour, height in people.get(cam, []):
                    cx, y2 = int(foot[0] * W), int(foot[1] * H)
                    d = Detection(tid, PERSON_CLASS, cx - 40, y2 - height, cx + 40, y2, 0.9)
                    if colour is not None:
                        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), colour, -1)
                    self.models[cam].by_x[d.x1] = prob
                    dets.append(d)
                an.update(dets, W, H, self.t, None, frame)
            self.fusion.tick(self.an, self.t)
            self.t += step

    def track(self, cam: str, tid: int):
        return self.an[cam].tracks[tid]


CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok)))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def who(t) -> str:
    return f"staff={t.is_staff} reason={t.staff_reason!r} store_id={t.store_id} peer_n={t.peer_n}"


def main() -> int:
    # 1. one customer in front of the billing desk, seen by both cameras
    s = Shop()
    s.run(4.0, {"A": [(10, A_FRONT, 0.03, None, 300)], "B": [(77, B_FRONT, 0.05, None, 300)]})
    b, a = s.track("B", 77), s.track("A", 10)
    check("same person on both cameras: the billing camera shows the overview camera's id", b.store_id == 10 and b.store_cam == "A", who(b))
    snaps = {c: an.snapshot(s.t) for c, an in s.an.items()}
    total = sum(x["occupancy"] - x["dup_customers"] for x in snaps.values())
    check("... and the store counts one customer, not two", snaps["A"]["occupancy"] == 1 and snaps["B"]["occupancy"] == 1 and total == 1,
          f"A={snaps['A']['occupancy']} B={snaps['B']['occupancy']} store={total}")
    check("... the overview track is not relabelled", a.store_id is None and not a.duplicate and a.peer_n > 0, who(a))

    # 2. customer in a polo that looks like the uniform from behind: one camera says staff, the other sees the logo
    s = Shop()
    s.run(10.0, {"A": [(11, A_FRONT, 0.97, None, 300)], "B": [(78, B_FRONT, 0.04, None, 300)]})
    a, b = s.track("A", 11), s.track("B", 78)
    check("views disagree (0.97 against 0.04): customer on both cameras", not a.is_staff and not b.is_staff, f"A: {who(a)} | B: {who(b)}")
    s.run(20.0, {"A": [(11, A_FRONT, 0.97, None, 300)], "B": [(78, B_FRONT, 0.04, None, 300)]})
    check("... and it stays that way", not s.track("A", 11).is_staff and not s.track("B", 78).is_staff, who(s.track("A", 11)))

    # 3. the cashier: the overview camera sees only the top of a head, the billing camera sees the uniform
    s = Shop()
    s.run(6.0, {"A": [(12, (0.10, 0.997), None, None, 120)], "B": [(79, B_DESK, 0.98, None, 300)]})
    a, b = s.track("A", 12), s.track("B", 79)
    check("cashier: staff on both cameras under one id", a.is_staff and b.is_staff and b.store_id == 12, f"A: {who(a)} | B: {who(b)}")
    snaps = {c: an.snapshot(s.t) for c, an in s.an.items()}
    check("... counted as one member of staff", sum(x["staff"] - x["dup_staff"] for x in snaps.values()) == 1,
          f"A={snaps['A']['staff']} B={snaps['B']['staff']}")

    # 4. both views agree on the uniform: staff, fast
    s = Shop()
    s.run(1.5, {"A": [(13, A_FRONT, 0.97, None, 300)], "B": [(80, B_FRONT, 0.98, None, 300)]})
    check("both views see the uniform: staff within 1.5 s on both", s.track("A", 13).is_staff and s.track("B", 80).is_staff, who(s.track("A", 13)))

    # 5. two customers at the desk: the pairs are made by clothing colour, whatever the track numbers
    s = Shop()
    s.run(3.0, {"A": [(20, (0.34, 0.90), 0.03, RED, 300), (21, (0.50, 0.90), 0.03, WHITE, 300)],
                "B": [(90, (0.90, 0.95), 0.03, RED, 300), (91, (0.65, 0.95), 0.03, WHITE, 300)]})
    check("two people in one region: matched by clothing colour", s.track("B", 90).store_id == 20 and s.track("B", 91).store_id == 21,
          f"red -> {s.track('B', 90).store_id}, white -> {s.track('B', 91).store_id}")
    check("colour cost: red against white is far, red against red is near", colour_cost((0, 200, 200), (0, 5, 235)) > 0.45 > colour_cost((0, 200, 200), (2, 190, 190)))

    # 6. someone the billing camera does not see is left alone; a link ends when the person walks away
    s = Shop()
    s.run(3.0, {"A": [(30, A_FLOOR, 0.03, None, 300), (31, A_FRONT, 0.03, None, 300)], "B": [(95, B_FRONT, 0.03, None, 300)]})
    check("a person outside the shared regions is not linked", s.track("B", 95).store_id == 31 and s.track("A", 30).peer_n == 0, who(s.track("B", 95)))
    s.run(4.0, {"A": [(30, A_FLOOR, 0.03, None, 300), (31, A_FLOOR, 0.03, None, 300)], "B": []})
    check("the link ends when the person leaves the second camera's view", s.track("A", 31).peer_n == 0 and not s.fusion.links, f"links={s.fusion.links}")

    # 7. a manual tag on one camera counts on the other
    s = Shop()
    s.run(3.0, {"A": [(40, A_FRONT, 0.03, None, 300)], "B": [(96, B_FRONT, 0.03, None, 300)]})
    s.an["A"].set_staff(40, True)
    s.run(1.0, {"A": [(40, A_FRONT, 0.03, None, 300)], "B": [(96, B_FRONT, 0.03, None, 300)]})
    check("tagged staff by hand on one camera: staff on the other too", s.track("B", 96).is_staff and s.track("B", 96).manual_staff is True, who(s.track("B", 96)))

    # 8. one visit per person: the second view adds its zones to the journey instead of a visit of its own
    s = Shop()
    s.run(5.0, {"A": [(50, A_FRONT, 0.03, None, 300)], "B": [(97, B_FRONT, 0.03, None, 300)]})
    s.run(4.0, {"A": [(50, A_FLOOR, 0.03, None, 300)], "B": []})
    s.run(4.0, {"A": [], "B": []})
    va, vb = s.stores["A"].visits, s.stores["B"].visits
    check("one visit is recorded, by the overview camera", len(va) == 1 and len(vb) == 0, f"A visits={len(va)} B visits={len(vb)}")
    check("... and it contains the zone the billing camera saw", bool(va) and "Billing counter" in va[0]["zones"], f"zones={va[0]['zones'] if va else None}")

    # 9. Congruence Eye: one visitor id per person; a zone is left under the id it was entered with
    sent = []
    stub = types.SimpleNamespace(configured=lambda: True, cam_id=lambda c: {"A": 31, "B": 32}[c], zone_id=lambda c, z: 74, session_id="s1",
                                 spool=types.SimpleNamespace(put=lambda kind, ev: sent.append(ev)), _wake_soon=lambda: None)
    s = Shop()
    s.run(3.0, {"A": [(60, A_FRONT, 0.03, None, 300)], "B": [(98, B_FRONT, 0.03, None, 300)]})
    b = s.track("B", 98)
    CongruenceConnector.visitor_event(stub, "B", "ZONE_ENTER", b, s.t, "Billing counter")
    b.store_id, b.store_cam = None, ""                   # the link is lost while the person is still in the zone
    CongruenceConnector.visitor_event(stub, "B", "ZONE_EXIT", b, s.t + 5, "Billing counter", 5.0)
    check("events of the second camera carry the overview camera's visitor id", sent[0]["visitorId"] == "A-s1-60" and sent[0]["trackId"] == "B-s1-98",
          f"visitorId={sent[0]['visitorId']} trackId={sent[0]['trackId']}")
    check("... ZONE_EXIT uses the id of its ZONE_ENTER", sent[1]["visitorId"] == "A-s1-60", f"exit visitorId={sent[1]['visitorId']}")

    # 10. no configuration: nothing changes
    with tempfile.TemporaryDirectory() as d:
        f = StoreFusion(Path(d) / "fusion.json")
        check("no fusion.json: fusion is off", not f.active and not f.status()["regions"])

    # 11. a foot point that strays into a shared region for a moment does not marry two different people
    s = Shop()
    s.run(3.0, {"A": [(70, A_SEAT, None, None, 120)], "B": [(99, B_FRONT, 0.03, None, 300)]})         # cashier seated; customer only on B
    s.run(0.5, {"A": [(70, (0.30, 0.95), None, None, 120)], "B": [(99, B_FRONT, 0.03, None, 300)]})   # the cashier leans over for half a second
    s.run(3.0, {"A": [(70, A_SEAT, None, None, 120)], "B": [(99, B_FRONT, 0.03, None, 300)]})
    check("half a second in the region is not a link", s.track("B", 99).store_id is None and not s.fusion.links, f"links={s.fusion.links}")
    s = Shop()
    s.run(4.0, {"A": [(71, A_FRONT, 0.03, WHITE, 300)], "B": [(100, B_FRONT, 0.03, (20, 20, 20), 300)]})
    check("one person each, but white against black clothing: not linked", s.track("B", 100).store_id is None, f"store_id={s.track('B', 100).store_id}")

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
