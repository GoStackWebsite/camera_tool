"""The phone-detection switch (Thresholds.phone_detection), no camera or model needed: python tools/test_phone_switch.py

Off (the default): a phone in someone's hand leaves no trace: no "using phone" state, no event, no alert.
On: the same frames raise the phone-usage alert. Switching off while someone holds a phone clears the state."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import analytics as A                       # noqa: E402
from app.config import CameraConfig, StaffConfig, Thresholds   # noqa: E402
from app.detector import PERSON_CLASS, PHONE_CLASS, Detection    # noqa: E402

W, H = 1280, 720
FRAME = np.full((H, W, 3), 90, np.uint8)
PERSON = Detection(1, PERSON_CLASS, 500, 200, 620, 600, 0.9)
PHONE = Detection(2, PHONE_CLASS, 540, 330, 570, 380, 0.6)      # inside the person's box


class FakeAlerts:
    def __init__(self):
        self.fired: list[str] = []

    def fire(self, key, atype, *a, **k):
        self.fired.append(atype)

    def resolve(self, *a, **k):
        pass

    def __getattr__(self, name):          # anything else the analytics may call on the alert manager
        return lambda *a, **k: None


def run(thr: Thresholds, seconds: float = 12.0):
    cam = CameraConfig(id="t1", name="Test", url="")
    alerts = FakeAlerts()
    an = A.Analytics(cam, thr, None, alerts, StaffConfig())
    t0, now = 1_000_000.0, 1_000_000.0
    while now - t0 <= seconds:
        an.update([PERSON, PHONE], W, H, now, None, FRAME)
        now += 0.2
    return an, alerts, now


results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


print("== phone detection off (default)")
check("default is off", Thresholds().phone_detection is False)
an, alerts, now = run(Thresholds(phone_everyone=True))
t = an.tracks[1]
check("no phone state on the track", t.phone_since is None and t.phone_box is None, f"since={t.phone_since} box={t.phone_box}")
check("no phone alert", "phone_usage" not in alerts.fired, str(alerts.fired))
check("no phone event", not any(e.get("type") in ("phone", "phone_demo") for e in an.snapshot(now).get("events", [])))
check("phones tile stays 0", an.snapshot(now).get("phones") == 0, str(an.snapshot(now).get("phones")))
check("the phone is not counted as a person", an.snapshot(now).get("occupancy") == 1, str(an.snapshot(now).get("occupancy")))

print("== phone detection on")
an, alerts, now = run(Thresholds(phone_detection=True, phone_everyone=True))
t = an.tracks[1]
check("phone state on the track", t.phone_since is not None and t.phone_box is not None)
check("phone alert fired once", alerts.fired.count("phone_usage") == 1, str(alerts.fired))
check("phones tile shows 1", an.snapshot(now).get("phones") == 1, str(an.snapshot(now).get("phones")))

print("== switched off while someone holds a phone")
an.apply_thresholds(Thresholds(phone_detection=False, phone_everyone=True))
check("state cleared at once", t.phone_since is None and t.phone_box is None)
an.update([PERSON, PHONE], W, H, now, None, FRAME)
check("stays clear on the next frame", t.phone_since is None and t.phone_box is None and an.snapshot(now).get("phones") == 0)

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
