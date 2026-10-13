"""Rule tests for staff detection (no camera, no trained model needed): python tools/test_staff_rules.py

Fake models return the probabilities the scenario dictates, so the tests exercise the decision rules of
app/analytics.py alone: staff zones as sustained, lapsing evidence, the uniform model's flip-flop damping, the
two-model verdict on clothing, and people who are lost and found again inside a staff zone."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import analytics as A                       # noqa: E402
from app.config import CameraConfig, StaffConfig, Thresholds, ZoneConfig   # noqa: E402
from app.detector import PERSON_CLASS, Detection    # noqa: E402

W, H = 1280, 720
FRAME = np.full((H, W, 3), 90, np.uint8)
STAFF_ZONE = [[0.70, 0.30], [1.0, 0.30], [1.0, 0.95], [0.70, 0.95]]
IN_ZONE, ON_FLOOR = (0.85, 0.80), (0.35, 0.80)       # foot points


class FakeModel:
    """Stands in for staffcls.StaffClassifier; `prob` maps a track's box centre to P(staff)."""

    def __init__(self):
        self.by_x: dict[int, float] = {}

    def staff_prob(self, frame, bbox):
        return self.by_x.get(bbox[0])

    def info(self):
        return {"loaded": True}


class FakeBrand:
    """Stands in for staffcls.BrandClassifier: one P(customer) for everybody."""

    def __init__(self, p_customer: float):
        self.p_customer = p_customer

    def read(self, frame, bbox):
        return self.p_customer, "gogizmo", 1.0 - self.p_customer

    def info(self):
        return {"loaded": True}


class FakeEmitter:
    """Stands in for the Congruence Eye connector: records what would be sent."""

    def __init__(self):
        self.sent: list[tuple] = []

    def visitor_event(self, cam, etype, t, ts, zone=None, dwell=None):
        self.sent.append((etype, bool(t.is_staff), round(ts, 2)))


DOOR_LINE = [[0.99, 0.90], [0.01, 0.90]]            # like the shop: people come in from the frame bottom, IN is above


def make(with_model: bool = True, brand: FakeBrand | None = None, zone: list | None = None, line: list | None = None):
    cam = CameraConfig(id="t", name="test", url="", line=line, zones=[ZoneConfig("Counter staff side", zone or STAFF_ZONE, "staff")])
    model = FakeModel() if with_model else None
    an = A.Analytics(cam, Thresholds(), None, None, StaffConfig(), model, brand)
    an.emitter = FakeEmitter()
    return an, model


def walk_in(an, model, track_id: int, prob: float | None, t0: float, x: float = 0.45, step: float = 0.07) -> tuple[float, float]:
    """Someone comes in under the camera: a short clipped box at the bottom first (no clothing visible), then the
    whole person walking up the shop. Returns (end time, time the count line was crossed)."""
    t, crossed = t0, None
    for y, height in ((1.0, 120), (1.0, 150), (0.999, 200), (0.97, 300), (0.93, 300), (0.89, 300), (0.86, 300), (0.82, 300), (0.78, 300)):
        d = det(track_id, (x, y), height)
        model.by_x[d.x1] = prob
        an.update([d], W, H, t, None, FRAME)
        if crossed is None and y < 0.89:
            crossed = t
        t += step * 3
    end = t + 8.0
    while t < end:                                       # then stands in the shop
        d = det(track_id, (x, 0.78))
        model.by_x[d.x1] = prob
        an.update([d], W, H, t, None, FRAME)
        t += step
    return t, crossed


def det(track_id: int, foot: tuple[float, float], height: int = 300) -> Detection:
    cx, y2 = int(foot[0] * W), int(foot[1] * H)
    return Detection(track_id, PERSON_CLASS, cx - 40, y2 - height, cx + 40, y2, 0.9)


def run(an, model, track_id: int, foot, seconds: float, prob: float | None, t0: float, step: float = 0.25, height: int = 300) -> float:
    """Keep one person at `foot` for `seconds`; the fake model answers `prob` for them. Returns the end time."""
    t = t0
    end = t0 + seconds
    while t < end:
        d = det(track_id, foot, height)
        if model is not None:
            model.by_x[d.x1] = prob
        an.update([d], W, H, t, None, FRAME)
        t += step
    return t


def lose(an, t: float, seconds: float = 4.0) -> float:
    """Nobody in view for a while: every track is lost and closed."""
    an.update([], W, H, t + seconds, None, FRAME)
    return t + seconds + 0.25


CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def state(an, track_id: int) -> str:
    t = an.tracks[track_id]
    return f"staff={t.is_staff} reason={t.staff_reason!r} bank={t.staff_zone_time:.0f}s"


def main() -> int:
    zs = Thresholds().staff_zone_seconds

    # 1. the customer in the screenshots: stands in the staff zone, then walks to the shop floor
    an, m = make()
    t = run(an, m, 1, IN_ZONE, zs + 5, 0.55, 1000.0)
    check("someone who stays in a staff zone is staff while there", an.tracks[1].is_staff and an.tracks[1].staff_reason == "staff zone", state(an, 1))
    t = run(an, m, 1, ON_FLOOR, A.STAFF_ZONE_AWAY_SECONDS - 10, 0.55, t)
    check("still staff shortly after stepping out (fetching something)", an.tracks[1].is_staff, state(an, 1))
    t = run(an, m, 1, ON_FLOOR, 20, 0.55, t)
    check("a customer who stood in the staff zone is a customer again once back on the floor", not an.tracks[1].is_staff, state(an, 1))

    # 2. real staff in uniform: zone first, clothing confirms, stays staff on the floor
    an, m = make()
    t = run(an, m, 2, IN_ZONE, zs + 5, 0.97, 1000.0)
    check("uniformed staff in the zone are confirmed by clothing", an.tracks[2].is_staff and an.tracks[2].staff_reason == "uniform", state(an, 2))
    t = run(an, m, 2, ON_FLOOR, 300, 0.97, t)
    check("uniformed staff stay staff after five minutes on the shop floor", an.tracks[2].is_staff, state(an, 2))

    # 3. zone edge: a seated customer whose box drifts into the zone a few seconds per minute never adds up
    an, m = make()
    t = 1000.0
    for _ in range(30):                                  # 30 minutes
        t = run(an, m, 3, IN_ZONE, 4, 0.60, t)
        t = run(an, m, 3, ON_FLOOR, 56, 0.60, t)
    check("brushing the zone edge for 30 minutes does not make a customer staff", not an.tracks[3].is_staff, state(an, 3))

    # 4. no clothing evidence at all (box too small / hidden): the zone decides, and it is not taken back blindly
    an, m = make()
    t = run(an, m, 4, IN_ZONE, zs + 5, None, 1000.0)
    check("with no clothing evidence a staff zone still identifies staff", an.tracks[4].is_staff, state(an, 4))
    t = run(an, m, 4, ON_FLOOR, 120, None, t)
    check("... and is not revoked while the model has not seen the person", an.tracks[4].is_staff, state(an, 4))

    # 5. out of uniform (owner in a shirt): the zone needs MODEL_ZONE_PATIENCE x longer, then holds while they are there
    an, m = make()
    t = run(an, m, 5, IN_ZONE, zs * 3, 0.05, 1000.0)
    check("someone clearly not in uniform is not staff after a short stay in the zone", not an.tracks[5].is_staff, state(an, 5))
    t = run(an, m, 5, IN_ZONE, zs * A.MODEL_ZONE_PATIENCE, 0.05, t)
    check("... but is after a long, sustained stay", an.tracks[5].is_staff, state(an, 5))
    t = run(an, m, 5, ON_FLOOR, 60, 0.05, t)
    t = run(an, m, 5, IN_ZONE, 5, 0.05, t)
    check("... and is staff again at once when back behind the counter", an.tracks[5].is_staff, state(an, 5))

    # 6. without a learned model the old behaviour is kept: zone staff stay staff on the floor
    an, m = make(with_model=False)
    t = run(an, None, 6, IN_ZONE, zs + 5, None, 1000.0)
    t = run(an, None, 6, ON_FLOOR, 300, None, t)
    check("no learned model installed: zone staff are kept (nothing else could re-identify them)", an.tracks[6].is_staff, state(an, 6))

    # 7. dark-clothed customer on the floor: the model wavers; after it took "staff" back, it needs far more
    an, m = make()
    t = run(an, m, 7, ON_FLOOR, 12, 0.95, 1000.0)
    first = an.tracks[7].is_staff
    t = run(an, m, 7, ON_FLOOR, 8, 0.10, t)
    demoted = not an.tracks[7].is_staff and an.tracks[7].model_demotions == 1
    t = run(an, m, 7, ON_FLOOR, 6, 0.86, t)              # the run that used to re-promote
    check("model flip-flop: a short good-looking run no longer re-promotes after a demotion", first and demoted and not an.tracks[7].is_staff,
          f"first={first} demoted={demoted} " + state(an, 7))
    t = run(an, m, 7, ON_FLOOR, 15, 0.97, t)
    check("... a full, clear window still does (real staff are not locked out)", an.tracks[7].is_staff, state(an, 7))
    t = run(an, m, 7, ON_FLOOR, 8, 0.10, t)
    t = run(an, m, 7, ON_FLOOR, 30, 0.99, t)
    check("... and after two demotions clothing alone no longer makes them staff", not an.tracks[7].is_staff and an.tracks[7].model_demotions == 2, state(an, 7))

    # 8. clothing verdict = both models: the staff model alone is sure, the brand model says "customer"
    an, m = make(brand=FakeBrand(0.65))
    t = run(an, m, 8, ON_FLOOR, 40, 0.98, 1000.0)
    check("black T-shirt customer: staff model says staff, brand model says customer -> stays a customer", not an.tracks[8].is_staff, state(an, 8))
    an, m = make(brand=FakeBrand(0.02))
    t = run(an, m, 8, ON_FLOOR, 12, 0.98, 1000.0)
    check("... both models agree on the uniform -> staff", an.tracks[8].is_staff and an.tracks[8].staff_reason == "uniform", state(an, 8))

    # 9. seated cashier of whom only the head shows: the tracker loses and re-finds her every few seconds
    an, m = make()
    t = run(an, m, 20, IN_ZONE, zs + 5, None, 1000.0)
    t = lose(an, t)
    t = run(an, m, 21, IN_ZONE, 0.5, None, t)
    check("lost and found again at the same spot in a staff zone: still staff, not a new visitor", an.tracks[21].is_staff, state(an, 21))
    t = run(an, m, 22, (IN_ZONE[0] + 0.12, IN_ZONE[1]), 0.5, None, t)
    check("... someone appearing elsewhere in the zone starts from zero", not an.tracks[22].is_staff, state(an, 22))
    an, m = make()
    t = 1000.0
    for i in range(4):                                   # four 8-second fragments of the same seated person
        t = run(an, m, 30 + i, IN_ZONE, 8, None, t)
        last = an.tracks[30 + i]
        t = lose(an, t)
    check("short fragments at one spot add up to the staff-zone time", last.is_staff, f"staff={last.is_staff} bank={last.staff_zone_time:.0f}s")

    # 10. a zone drawn to the frame edge contains the people whose box ends on that edge
    an, m = make(zone=[[0.70, 0.60], [1.0, 0.60], [1.0, 1.0], [0.70, 1.0]])
    t = run(an, m, 40, (0.85, 1.0), zs + 5, None, 1000.0)
    check("zone drawn to the bottom edge counts a person whose box ends at the edge", an.tracks[40].is_staff, state(an, 40))

    # 11. a short box cut off by the frame bottom shows hair, not clothing: no vote, so the zone is not held back
    an, m = make(zone=[[0.70, 0.60], [1.0, 0.60], [1.0, 1.0], [0.70, 1.0]])
    t = run(an, m, 41, (0.85, 0.993), zs + 5, 0.0, 1000.0, height=150)
    check("head-only crop at the frame bottom gives no 'customer' votes and the staff zone decides", an.tracks[41].is_staff and not an.tracks[41].model_votes,
          state(an, 41) + f" votes={len(an.tracks[41].model_votes)}")

    # 12. how fast: a clear view of the uniform is staff in about a second, not five
    an, m = make()
    t = run(an, m, 50, ON_FLOOR, 1.2, 0.97, 1000.0, step=0.07)
    check("clear view of the uniform: staff within 1.2 s of coming into view", an.tracks[50].is_staff, state(an, 50))
    an, m = make()
    t = run(an, m, 51, ON_FLOOR, 1.2, 0.05, 1000.0, step=0.07)
    check("clear customer: judged (no longer 'identifying') within 1.2 s", an.tracks[51].judged and not an.tracks[51].is_staff, state(an, 51))
    an, m = make()
    t = run(an, m, 52, ON_FLOOR, 2.5, 0.85, 1000.0, step=0.07)
    early = an.tracks[52].is_staff
    t = run(an, m, 52, ON_FLOOR, 2.0, 0.85, t, step=0.07)
    check("less clear view (0.85): not on a burst of samples, but after 3 s of them", not early and an.tracks[52].is_staff, f"early={early} " + state(an, 52))
    an, m = make()
    t = run(an, m, 53, ON_FLOOR, 0.5, None, 1000.0, step=0.07)
    snap = an.snapshot(t)
    check("someone not judged yet is 'identifying', not a customer", snap["occupancy"] == 0 and snap["identifying"] == 1, f"occupancy={snap['occupancy']} identifying={snap['identifying']}")
    t = run(an, m, 53, ON_FLOOR, A.JUDGE_MAX_SECONDS, None, t, step=0.07)
    snap = an.snapshot(t)
    check("... and counts as a visitor once nobody could tell for 6 s", snap["occupancy"] == 1 and snap["identifying"] == 0, f"occupancy={snap['occupancy']}")

    # 13. staff walking back in through the door are not walk-ins
    an, m = make(line=DOOR_LINE)
    t, crossed = walk_in(an, m, 60, 0.97, 1000.0)
    entry = [e for e in an.emitter.sent if e[0] == "ENTRY"]
    check("staff coming in: no customer walk-in is counted", an.entries == 0 and an.staff_entries == 1, f"entries={an.entries} staff_entries={an.staff_entries}")
    check("... the entry is sent once, flagged staff, with the time of the crossing (not of the verdict)", len(entry) == 1 and entry[0][1] and 0 <= entry[0][2] - crossed <= 0.5, f"{entry} crossed={crossed:.2f}")
    an, m = make(line=DOOR_LINE)
    t, crossed = walk_in(an, m, 61, 0.03, 1000.0)
    entry = [e for e in an.emitter.sent if e[0] == "ENTRY"]
    check("customer coming in: one walk-in, time of the crossing", an.entries == 1 and an.staff_entries == 0 and len(entry) == 1 and not entry[0][1]
          and 0 <= entry[0][2] - crossed <= 0.5, f"entries={an.entries} {entry} crossed={crossed:.2f}")
    an, m = make(line=DOOR_LINE)
    t, crossed = walk_in(an, m, 62, None, 1000.0)
    check("nobody could tell who came in: booked as a walk-in after the hold", an.entries == 1 and len([e for e in an.emitter.sent if e[0] == "ENTRY"]) == 1, f"entries={an.entries}")
    an, m = make(line=DOOR_LINE)
    for i, y in enumerate((1.0, 0.97, 0.93, 0.88, 0.85, 0.82)):        # comes in and is lost at once
        d = det(63, (0.45, y))
        m.by_x[d.x1] = None
        an.update([d], W, H, 1000.0 + 0.2 * i, None, FRAME)
    before = an.entries
    lose(an, 1001.2)
    check("lost during the hold: the entry is booked when the track closes, not dropped", before == 0 and an.entries == 1, f"before={before} after={an.entries}")

    # 14. the tracker gives someone a new number (id switch): they are not judged from scratch
    an, m = make()
    t = run(an, m, 70, ON_FLOOR, 3.0, 0.97, 1000.0, step=0.07)
    d = det(71, ON_FLOOR)
    m.by_x[d.x1] = 0.97
    an.update([d], W, H, t, None, FRAME)
    check("staff under a new track number at the same spot: staff from the first frame", an.tracks[71].is_staff and an.tracks[71].judged, state(an, 71))
    an, m = make()
    t = run(an, m, 72, ON_FLOOR, 3.0, 0.03, 1000.0, step=0.07)
    d = det(73, ON_FLOOR)
    m.by_x[d.x1] = 0.03
    an.update([d], W, H, t, None, FRAME)
    snap = an.snapshot(t)
    check("customer under a new track number: still a counted customer, no 'identifying' gap", an.tracks[73].judged and snap["identifying"] == 0, f"judged={an.tracks[73].judged} identifying={snap['identifying']}")
    an, m = make()
    t = run(an, m, 74, ON_FLOOR, 3.0, 0.97, 1000.0, step=0.07)
    an.set_staff(74, False)                              # the operator says: not staff
    d = det(75, ON_FLOOR)
    m.by_x[d.x1] = 0.97
    an.update([d], W, H, t, None, FRAME)
    t = run(an, m, 75, ON_FLOOR, 5.0, 0.97, t + 0.07, step=0.07)
    check("a manual 'not staff' tag survives the id switch", not an.tracks[75].is_staff and an.tracks[75].manual_staff is False, state(an, 75))
    an, m = make(line=DOOR_LINE)
    t = run(an, m, 76, (0.45, 0.999), 3.0, 0.97, 1000.0, step=0.07)       # staff leaving, last seen at the frame bottom
    t = lose(an, t, 2.0)
    d = det(77, (0.45, 0.999))
    m.by_x[d.x1] = None
    an.update([d], W, H, t, None, FRAME)
    check("at the door nothing carries over: the next person in is not the one who just left", not an.tracks[77].is_staff and not an.tracks[77].judged, state(an, 77))

    # 15. someone waiting for a staff zone's verdict is "identifying" all the way, never a visitor in between
    an, m = make()
    t = run(an, m, 80, IN_ZONE, 10.0, None, 1000.0, step=0.1, height=150)
    snap = an.snapshot(t)
    mid = (snap["occupancy"], snap["identifying"], an.tracks[80].is_staff)
    t = run(an, m, 80, IN_ZONE, zs, None, t, step=0.1, height=150)
    check("in a staff zone, no clothing visible: 'identifying' until the zone decides, then staff", mid == (0, 1, False) and an.tracks[80].is_staff,
          f"after 10 s (customers, identifying, staff)={mid}, then " + state(an, 80))

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
