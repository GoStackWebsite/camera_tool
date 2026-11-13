"""IN / OUT for people still being identified, no camera or model needed: python tools/test_identifying_cross.py

A person who crosses the count line while still "identifying" (staff or customer not decided yet) must be captured the
moment they cross: the raw door count goes up, their box says "Crossed IN/OUT", and an IN shows as pending ("+1")
until it is booked as a walk-in or as staff. A fake clothing model that never makes up its mind keeps them
identifying for the full JUDGE_MAX_SECONDS."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import analytics as A                       # noqa: E402
from app.config import CameraConfig, StaffConfig, Thresholds   # noqa: E402
from app.detector import PERSON_CLASS, Detection    # noqa: E402

W, H = 1280, 720
FRAME = np.full((H, W, 3), 90, np.uint8)
LINE = [[0.99, 0.50], [0.01, 0.50]]                  # IN is above the line
OUTSIDE, INSIDE = 0.80, 0.30


class UnsureModel:
    """Clothing model that cannot tell: 0.7 is neither clearly staff nor clearly a customer."""

    def __init__(self, prob: float = 0.7):
        self.prob = prob

    def staff_prob(self, frame, bbox):
        return self.prob


def make(prob: float = 0.7, minutes: float = 0.0):
    cam = CameraConfig(id="door", name="door", url="", line=LINE, zones=[])
    an = A.Analytics(cam, Thresholds(reentry_minutes=minutes), None, None, StaffConfig())
    an.staff_model = UnsureModel(prob)
    return an


def det(tid: int, x: float, y: float) -> Detection:
    cx, y2 = int(x * W), int(y * H)
    return Detection(tid, PERSON_CLASS, cx - 40, y2 - 160, cx + 40, y2, 0.9)


def walk(an, tid, x, y0, y1, t, seconds=1.5, step=0.1):
    n = max(2, int(seconds / step))
    st = None
    for i in range(n + 1):
        st = an.update([det(tid, x, y0 + (y1 - y0) * i / n)], W, H, t, None, FRAME)
        t += step
    return t, st


results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


print("== crosses IN while identifying")
an = make()
t, st = walk(an, 1, 0.4, OUTSIDE, INSIDE, 100.0)
tr = an.tracks[1]
check("still identifying", not tr.judged)
check("door count goes up at once", an.door_in == 1, f"door_in={an.door_in}")
check("box says Crossed IN", tr.cross_mark is not None and tr.cross_mark[0] == "IN")
st = an.snapshot(t - 0.1)
check("shown as pending walk-in", st is not None and st.get("pending_in") == 1 and st.get("entries") == 0,
      f"pending_in={st.get('pending_in') if st else None} entries={st.get('entries') if st else None}")
t, st = walk(an, 1, 0.4, INSIDE, INSIDE, t, seconds=A.ENTRY_HOLD_SECONDS + 1.0)
check("booked as a walk-in once nobody could tell", an.entries == 1 and an.door_in == 1, f"entries={an.entries}")
check("box then says what it counted as", tr.cross_mark[0] == "IN: walk-in", str(tr.cross_mark))

print("== crosses OUT while identifying")
an = make()
t, _ = walk(an, 2, 0.4, INSIDE, OUTSIDE, 100.0)
tr = an.tracks[2]
check("still identifying", not tr.judged)
check("exit and door out counted at once", an.exits == 1 and an.door_out == 1, f"exits={an.exits} door_out={an.door_out}")
check("box says Crossed OUT: exit", tr.cross_mark is not None and tr.cross_mark[0] == "OUT: exit", str(tr.cross_mark))

print("== staff crossing: in the door count, not a walk-in")
an = make(prob=0.99)
t, _ = walk(an, 3, 0.4, OUTSIDE, INSIDE, 100.0, seconds=3.0)
t, _ = walk(an, 3, 0.4, INSIDE, INSIDE, t, seconds=2.0)
check("staff entry, no walk-in, door counted", an.staff_entries == 1 and an.entries == 0 and an.door_in == 1,
      f"staff_entries={an.staff_entries} entries={an.entries} door_in={an.door_in}")

print("== same person back in within the re-entry window")
an = make(prob=0.1, minutes=30.0)
t, _ = walk(an, 4, 0.4, OUTSIDE, INSIDE, 100.0, seconds=3.0)
t, _ = walk(an, 4, 0.4, INSIDE, INSIDE, t, seconds=A.ENTRY_HOLD_SECONDS + 1.0)
t, _ = walk(an, 4, 0.4, INSIDE, OUTSIDE, t, seconds=3.0)
t, _ = walk(an, 4, 0.4, OUTSIDE, OUTSIDE, t, seconds=2.0)
t, _ = walk(an, 4, 0.4, OUTSIDE, INSIDE, t, seconds=3.0)
t, _ = walk(an, 4, 0.4, INSIDE, INSIDE, t, seconds=A.ENTRY_HOLD_SECONDS + 1.0)
tr = an.tracks[4]
check("door counts both, In counts one", an.door_in == 2 and an.entries == 1 and an.reentries == 1,
      f"door_in={an.door_in} entries={an.entries} reentries={an.reentries}")
check("box says why In did not go up", tr.cross_mark[0] == "IN: came back, not a new walk-in", str(tr.cross_mark))
st = an.snapshot(t - 0.1)
check("came back is in the stats", st.get("reentries") == 1)

failed = results.count(False)
print(f"\n{len(results) - failed}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
