"""Watch the running engine for a while and report what the staff logic and the store fusion did:
python tools/watch_live.py [seconds]   (default 180)"""
import json
import sys
import time
import urllib.request

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
end = time.time() + secs
seen: dict = {}          # (camera, id) -> {"first": ts, "staff_at": ts or None, "judged_at": ts or None}
links: dict = {}
last = {}
while time.time() < end:
    try:
        s = json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/stats", timeout=5))
    except Exception:
        time.sleep(2)
        continue
    now = time.time()
    for cid, cam in s["cameras"].items():
        for t in cam.get("tracks", []):
            k = (cam["name"], t["id"])
            r = seen.setdefault(k, {"first": now - t["dwell"], "staff_at": None, "judged_at": None, "reason": "", "store_id": None})
            if t["staff"] and r["staff_at"] is None:
                r["staff_at"], r["reason"] = now, t["staff_reason"]
            if t.get("judged") and r["judged_at"] is None:
                r["judged_at"] = now
            r["staff"] = t["staff"]
            r["store_id"] = t.get("store_id") or r["store_id"]
        last[cam["name"]] = {k: cam.get(k) for k in ("entries", "staff_entries", "exits", "staff_exits")}
    for l in (s.get("fusion") or {}).get("links", []):
        k = (l["camera"], l["track"], l["store_id"])
        if k not in links:
            links[k] = (time.strftime("%H:%M:%S"), l["region"])
    store = s.get("store") or {}
    time.sleep(1.0)
print("people seen:", len(seen))
quick = [r["staff_at"] - r["first"] for r in seen.values() if r["staff_at"]]
print("became staff:", len(quick), "| seconds from first sight to 'staff':", sorted(round(q, 1) for q in quick))
jq = [r["judged_at"] - r["first"] for r in seen.values() if r["judged_at"] and not r["staff_at"]]
print("judged customers:", len(jq), "| seconds to verdict:", sorted(round(q, 1) for q in jq))
print("links made (camera, its track -> store id):")
for k, v in links.items():
    print("   ", v[0], k[0], f"#{k[1]} -> {k[2]}", v[1])
print("counters:", last)
print("store now:", {k: store.get(k) for k in ("occupancy", "staff")})
