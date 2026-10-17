"""One walk-in per person (Thresholds.reentry_minutes), no camera or model needed: python tools/test_reentry.py

Someone who crosses in, out and in again without leaving the view is one walk-in and one exit. Someone who went out
of the door and comes back as a new track is recognised by appearance against the store's list of recent leavers
(app/reid.py RevisitGate) and is not counted again; a different person, a return after the window, or two look-alike
leavers are counted as new walk-ins. Staff entries are not touched. A fake embedder returns the vector the scenario
dictates, so only the counting rules are tested here, not the appearance model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import analytics as A                       # noqa: E402
from app.config import CameraConfig, StaffConfig, Thresholds   # noqa: E402
from app.detector import PERSON_CLASS, Detection    # noqa: E402
from app.reid import REVISIT_SIMILARITY, RevisitGate, colour_signature  # noqa: E402

W, H = 1280, 720
FRAME = np.full((H, W, 3), 90, np.uint8)
LINE = [[0.99, 0.50], [0.01, 0.50]]                  # IN is above the line (people come in from the frame bottom)
OUTSIDE, INSIDE = 0.80, 0.30                          # foot y on either side
DIM = 64


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM)
    return v / np.linalg.norm(v)


def near(v: np.ndarray, cos: float, seed: int) -> np.ndarray:
    """A unit vector at cosine `cos` from v."""
    r = unit(seed)
    r = r - np.dot(r, v) * v
    r /= np.linalg.norm(r)
    return cos * v + np.sqrt(1 - cos * cos) * r


def outfit(top_bgr, bottom_bgr) -> np.ndarray:
    """Colour histograms of a person in these clothes (the real colour_signature on a painted box)."""
    img = np.zeros((160, 80, 3), np.uint8)
    img[:80], img[80:] = top_bgr, bottom_bgr
    return colour_signature(img, (0, 0, 80, 160))


RED_BLUE = outfit((40, 40, 200), (150, 60, 20))
GREEN_GREY = outfit((40, 160, 40), (120, 120, 120))


class FakeEmbedder:
    """Stands in for reid.Embedder: the appearance of a box is looked up by its left edge. Everybody wears the same
    colours unless a scenario says otherwise (`colour_by_x`), so the embedding rules are what is tested."""

    def __init__(self):
        self.by_x: dict[int, np.ndarray] = {}
        self.colour_by_x: dict[int, np.ndarray] = {}

    def embed(self, frame, bbox):
        return self.by_x.get(bbox[0])

    def colour(self, frame, bbox):
        return self.colour_by_x.get(bbox[0], RED_BLUE)


class FakeEmitter:
    """Stands in for the Congruence Eye connector, including how it gives every person a visitor id."""

    def __init__(self):
        self.sent: list[tuple] = []

    def visitor_event(self, cam, etype, t, ts, zone=None, dwell=None):
        if not t.visitor_key:
            t.visitor_key = f"{cam}-s-{t.id}"
        self.sent.append((etype, t.id, t.visitor_key, bool(t.is_staff)))

    def count(self, etype):
        return sum(1 for e in self.sent if e[0] == etype)


def make(minutes: float = 30.0, gate: RevisitGate | None = None):
    cam = CameraConfig(id="door", name="door", url="", line=LINE, zones=[])
    an = A.Analytics(cam, Thresholds(reentry_minutes=minutes), None, None, StaffConfig())
    an.emitter = FakeEmitter()
    an.embedder = FakeEmbedder()
    an.revisits = gate or RevisitGate()
    return an


def det(tid: int, x: float, y: float) -> Detection:
    cx, y2 = int(x * W), int(y * H)
    return Detection(tid, PERSON_CLASS, cx - 40, y2 - 160, cx + 40, y2, 0.9)


def walk(an, tid: int, x: float, y0: float, y1: float, t: float, seconds: float = 3.0, step: float = 0.1,
         look: np.ndarray | None = None) -> float:
    n = max(2, int(seconds / step))
    for i in range(n + 1):
        d = det(tid, x, y0 + (y1 - y0) * i / n)
        if look is not None:
            an.embedder.by_x[d.x1] = look
        an.update([d], W, H, t, None, FRAME)
        t += step
    return t


def stay(an, tid: int, x: float, y: float, t: float, seconds: float, look: np.ndarray | None = None) -> float:
    return walk(an, tid, x, y, y, t, seconds, look=look)


def gone(an, t: float, seconds: float = 2.5) -> float:
    """Nobody in view for a while: lost tracks close."""
    end = t + seconds
    while t < end:
        an.update([], W, H, t, None, FRAME)
        t += 0.1
    return t


results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


A_LOOK, B_LOOK = unit(1), unit(2)

print("== same track: in, out, in, out")
an = make()
t = walk(an, 1, 0.40, OUTSIDE, INSIDE, 0.0, look=A_LOOK)
t = stay(an, 1, 0.40, INSIDE, t, 3.0, look=A_LOOK)
first = an.entries
t = walk(an, 1, 0.40, INSIDE, OUTSIDE, t, look=A_LOOK)
t = stay(an, 1, 0.40, OUTSIDE, t, 2.0, look=A_LOOK)
t = walk(an, 1, 0.40, OUTSIDE, INSIDE, t, look=A_LOOK)
t = stay(an, 1, 0.40, INSIDE, t, 3.0, look=A_LOOK)
t = walk(an, 1, 0.40, INSIDE, OUTSIDE, t, look=A_LOOK)
t = stay(an, 1, 0.40, OUTSIDE, t, 2.0, look=A_LOOK)
check("first crossing is a walk-in", first == 1, f"entries={first}")
check("one walk-in and one exit after in-out-in-out", an.entries == 1 and an.exits == 1, f"entries={an.entries} exits={an.exits}")
check("the return is logged as 'back in' and 'out again'", an.reentries == 1 and an.reexits == 1, f"reentries={an.reentries} reexits={an.reexits}")
check("Congruence Eye gets one ENTRY and one EXIT", an.emitter.count("ENTRY") == 1 and an.emitter.count("EXIT") == 1, str(an.emitter.sent))
check("events say why", any(e["type"] == "reentry" and "crossed in again" in e["message"] for e in an.events))

print("== came back as a new track, same appearance")
an = make()
t = walk(an, 10, 0.40, OUTSIDE, INSIDE, 0.0, look=A_LOOK)
t = stay(an, 10, 0.40, INSIDE, t, 4.0, look=A_LOOK)
t = walk(an, 10, 0.40, INSIDE, OUTSIDE, t, look=A_LOOK)
t = gone(an, t)
check("the leaver is on the store's list", an.revisits.stats()["waiting"] == 1, str(an.revisits.stats()))
t += 120.0                                                 # two minutes later
t = walk(an, 11, 0.55, OUTSIDE, INSIDE, t, look=near(A_LOOK, 0.95, 7))
t = stay(an, 11, 0.55, INSIDE, t, 3.0, look=near(A_LOOK, 0.95, 7))
check("not a second walk-in", an.entries == 1 and an.reentries == 1, f"entries={an.entries} reentries={an.reentries}")
check("no second ENTRY to Congruence Eye", an.emitter.count("ENTRY") == 1, str(an.emitter.sent))
check("keeps the first visit's visitor id", an.tracks[11].visitor_key == "door-s-10", str(an.tracks[11].visitor_key))
check("the event names who it was", any(e["type"] == "reentry" and "#10" in e["message"] for e in an.events))
t = walk(an, 11, 0.55, INSIDE, OUTSIDE, t, look=near(A_LOOK, 0.95, 7))
t = stay(an, 11, 0.55, OUTSIDE, t, 2.0)
check("their way out again is not a second exit", an.exits == 1 and an.emitter.count("EXIT") == 1, f"exits={an.exits}")

print("== a different person comes in")
t = gone(an, t)
t = walk(an, 12, 0.30, OUTSIDE, INSIDE, t + 30.0, look=B_LOOK)
t = stay(an, 12, 0.30, INSIDE, t, 3.0, look=B_LOOK)
check("counted as a new walk-in", an.entries == 2 and an.emitter.count("ENTRY") == 2, f"entries={an.entries}")

print("== back after the window")
an = make(minutes=30.0)
t = walk(an, 20, 0.40, OUTSIDE, INSIDE, 0.0, look=A_LOOK)
t = stay(an, 20, 0.40, INSIDE, t, 4.0, look=A_LOOK)
t = walk(an, 20, 0.40, INSIDE, OUTSIDE, t, look=A_LOOK)
t = gone(an, t)
t += 40 * 60.0
t = walk(an, 21, 0.40, OUTSIDE, INSIDE, t, look=A_LOOK)
t = stay(an, 21, 0.40, INSIDE, t, 3.0, look=A_LOOK)
check("40 min later with a 30 min window: a new walk-in", an.entries == 2 and an.reentries == 0, f"entries={an.entries}")

print("== two look-alikes left, one comes back")
an = make()
lookalike = near(A_LOOK, 0.97, 3)
an.revisits.remember(A_LOOK, RED_BLUE, 10.0, "door", 30, True, "door-s-30")       # two customers in nearly the same
an.revisits.remember(lookalike, RED_BLUE, 12.0, "door", 31, True, "door-s-31")    # clothes went out a moment apart
t = 70.0
between = (A_LOOK + lookalike) / np.linalg.norm(A_LOOK + lookalike)   # as close to one as to the other
t = walk(an, 32, 0.45, OUTSIDE, INSIDE, t, look=between)
t = stay(an, 32, 0.45, INSIDE, t, 3.0, look=between)
check("cannot tell which one: counted as a new walk-in", an.entries == 1 and an.reentries == 0 and an.revisits.stats()["ambiguous"] == 1,
      f"entries={an.entries} ambiguous={an.revisits.stats()['ambiguous']}")

print("== similarity just below the bar")
an = make()
t = walk(an, 40, 0.40, OUTSIDE, INSIDE, 0.0, look=A_LOOK)
t = stay(an, 40, 0.40, INSIDE, t, 4.0, look=A_LOOK)
t = walk(an, 40, 0.40, INSIDE, OUTSIDE, t, look=A_LOOK)
t = gone(an, t)
t = walk(an, 41, 0.40, OUTSIDE, INSIDE, t + 60.0, look=near(A_LOOK, REVISIT_SIMILARITY - 0.03, 5))
t = stay(an, 41, 0.40, INSIDE, t, 3.0, look=near(A_LOOK, REVISIT_SIMILARITY - 0.03, 5))
check(f"cosine {REVISIT_SIMILARITY - 0.03:.2f}: a new walk-in", an.entries == 2, f"entries={an.entries}")

print("== same build, different clothes")
an = make()
t = walk(an, 45, 0.40, OUTSIDE, INSIDE, 0.0, look=A_LOOK)
t = stay(an, 45, 0.40, INSIDE, t, 4.0, look=A_LOOK)
t = walk(an, 45, 0.40, INSIDE, OUTSIDE, t, look=A_LOOK)
t = gone(an, t)
an.embedder.colour_by_x[det(46, 0.70, OUTSIDE).x1] = GREEN_GREY
t = walk(an, 46, 0.70, OUTSIDE, INSIDE, t + 60.0, look=A_LOOK)
t = stay(an, 46, 0.70, INSIDE, t, 3.0, look=A_LOOK)
check("embedding alike but the colours differ: a new walk-in", an.entries == 2 and an.reentries == 0, f"entries={an.entries}")

print("== timing of the booking")
an = make()
t = walk(an, 50, 0.40, OUTSIDE, INSIDE, 0.0, look=A_LOOK)
check("nobody recently left: booked straight away", an.entries == 1, f"entries={an.entries}")
t = stay(an, 50, 0.40, INSIDE, t, 3.0, look=A_LOOK)
t = walk(an, 50, 0.40, INSIDE, OUTSIDE, t, look=A_LOOK)
t = gone(an, t)
t = walk(an, 51, 0.65, OUTSIDE, INSIDE, t + 60.0, look=None)       # no clean view of the newcomer at all
waiting = an.entries
t = stay(an, 51, 0.65, INSIDE, t, A.REVISIT_HOLD_SECONDS + 1.0)
check("someone just left and no view of the newcomer: waits, then counts after the hold", waiting == 1 and an.entries == 2,
      f"while waiting={waiting} after={an.entries}")

print("== staff are not touched")
an = make()
t = walk(an, 60, 0.40, OUTSIDE, OUTSIDE - 0.05, 0.0)
an.tracks[60].manual_staff, an.tracks[60].is_staff, an.tracks[60].judged = True, True, True
t = walk(an, 60, 0.40, OUTSIDE - 0.05, INSIDE, t)
t = walk(an, 60, 0.40, INSIDE, OUTSIDE, t)
t = stay(an, 60, 0.40, OUTSIDE, t, 2.0)
t = walk(an, 60, 0.40, OUTSIDE, INSIDE, t)
t = stay(an, 60, 0.40, INSIDE, t, 2.0)
check("staff in-out-in: two staff entries, no walk-ins", an.staff_entries == 2 and an.entries == 0 and an.reentries == 0,
      f"staff_entries={an.staff_entries} entries={an.entries}")

print("== switched off (reentry_minutes = 0)")
an = make(minutes=0.0)
t = walk(an, 70, 0.40, OUTSIDE, INSIDE, 0.0, look=A_LOOK)
t = stay(an, 70, 0.40, INSIDE, t, 2.0, look=A_LOOK)
t = walk(an, 70, 0.40, INSIDE, OUTSIDE, t, look=A_LOOK)
t = stay(an, 70, 0.40, OUTSIDE, t, 2.0, look=A_LOOK)
t = walk(an, 70, 0.40, OUTSIDE, INSIDE, t, look=A_LOOK)
t = stay(an, 70, 0.40, INSIDE, t, 2.0, look=A_LOOK)
check("every crossing counts, as before", an.entries == 2 and an.exits == 1 and an.reentries == 0, f"entries={an.entries} exits={an.exits}")

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
