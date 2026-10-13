"""List the engine's staff decisions since a time today: python tools/staff_events.py [HH:MM]"""
import datetime
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
hm = sys.argv[1] if len(sys.argv) > 1 else "00:00"
t0 = datetime.datetime.combine(datetime.date.today(), datetime.time(*map(int, hm.split(":")))).timestamp()
db = sqlite3.connect(f"file:{(ROOT / 'data' / 'retail.db').as_posix()}?mode=ro", uri=True)
rows = db.execute("select ts, camera_id, track_id, message, zone from events where ts >= ? and type = 'staff' order by ts", (t0,)).fetchall()
c = Counter()
for ts, cam, tid, msg, zone in rows:
    kind = msg.split("(")[-1].rstrip(")") if "classified as staff" in msg else "back to customer"
    c[(cam, kind)] += 1
    print(datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S"), cam, f"#{tid}", msg.split(" ", 2)[2], f"[{zone}]" if zone else "")
print(dict(sorted(c.items())))
