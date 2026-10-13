"""Print the engine's live tracks with the evidence behind each staff decision, and the cross-camera links:
python tools/live_tracks.py"""
import json
import urllib.request

s = json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/stats", timeout=8))
cams = s.get("cameras", s)
st = s.get("store") or {}
print("STORE", {k: st.get(k) for k in ("occupancy", "staff", "entries", "exits")})
for cam in (cams if isinstance(cams, list) else cams.values()):
    print(cam.get("name"), {k: cam.get(k) for k in ("occupancy", "staff", "identifying", "entries", "staff_entries", "exits", "staff_exits")},
          "AI fps", (cam.get("camera") or {}).get("inference_fps", cam.get("inference_fps")))
    for t in cam.get("tracks", []):
        print("   ", {k: t.get(k) for k in ("id", "store_id", "staff", "staff_reason", "judged", "model_pct", "peer_pct", "zone_bank", "demotions", "dwell", "zones", "foot")})
f = s.get("fusion") or {}
print("fusion", {"active": f.get("active"), "links": f.get("links"), "error": f.get("error")})
for k in ("staff_model", "brand_model"):
    m = s.get(k) or {}
    print(k, {x: m.get(x) for x in ("loaded", "trained", "samples")})
