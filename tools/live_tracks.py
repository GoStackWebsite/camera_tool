"""Print the engine's live tracks with the evidence behind each staff decision: python tools/live_tracks.py"""
import json
import urllib.request

s = json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/stats", timeout=8))
cams = s.get("cameras", s)
for cam in (cams if isinstance(cams, list) else cams.values()):
    print(cam.get("name"), {k: cam.get(k) for k in ("occupancy", "staff", "entries", "exits")})
    for t in cam.get("tracks", []):
        print("   ", {k: t.get(k) for k in ("id", "staff", "staff_reason", "model_pct", "zone_bank", "demotions", "dwell", "zones", "foot")})
for k in ("staff_model", "brand_model"):
    m = s.get(k) or {}
    print(k, {x: m.get(x) for x in ("loaded", "trained", "samples")})
