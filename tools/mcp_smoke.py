"""Smoke test for the congruence-eye MCP server: calls the plain functions directly.

Run:  D:\\cam\\.venv\\Scripts\\python.exe D:\\cam\\tools\\mcp_smoke.py
Needs Congruence Eye (:3000) and the engine (:8000) running.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp_server import server as S  # noqa: E402

DAY = "2026-10-09"
FROM, TO = "2026-10-03", "2026-10-09"
failures = 0


def section(title: str) -> None:
    print("\n" + "=" * 8, title, "=" * 8)


def run(name: str, fn, *args):
    global failures
    t0 = time.time()
    try:
        res = fn(*args)
    except Exception as exc:  # the functions should never raise
        failures += 1
        print(f"{name}: RAISED {type(exc).__name__}: {exc}")
        return {}
    dt = round(time.time() - t0, 1)
    if "error" in res:
        failures += 1
        print(f"{name} ({dt}s): ERROR {res['error']}")
    else:
        print(f"{name} ({dt}s): ok, keys={list(res.keys())}")
    for n in res.get("notes", []):
        print("   note:", n)
    return res


def j(obj, n: int = 400) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + "..."


section("help")
h = run("help", S.help)
print("  tools:", len(h.get("tools", {})))

section("list_stores")
r = run("list_stores", S.list_stores)
for s in r.get("stores", []):
    print(f"  #{s['id']} {s['name']!r} active={s['isActive']} cams={s['cameras']} zones={s['zone_types']} "
          f"edge={s['edge_version']} seen {s['edge_last_seen_min_ago']} min ago health={s['health']} online={s['cameras_online']}")

section("engine_status")
r = run("engine_status", S.engine_status)
print("  store:", r.get("store"), "model:", r.get("model"), f"online {r.get('cameras_online')}/{r.get('cameras_total')}")
for c in r.get("cameras", []):
    print(f"  cam {c['id']} enabled={c['enabled']} line={c['has_count_line']} zones={c['zones_by_role']} frame={c['frame']} status={c['status']}")
print("  staff:", j({k: v for k, v in r.get("staff", {}).items() if k != "uniforms"}), "uniforms:", [u.get("name") for u in r.get("staff", {}).get("uniforms", [])])
print("  connector:", j({k: r.get("connector", {}).get(k) for k in ("enabled", "online", "managed", "backlog", "snapshot_seconds", "sent")}))

section("feature_audit")
r = run("feature_audit", S.feature_audit)
for f in r.get("features", []):
    print(f"  [{f['area']}] {f['feature']:<48} {f['status']:<11} {j(f['detail'], 110)}")
print("  disabled/missing:", r.get("disabled_or_missing"))

section(f"engine_day_summary({DAY})")
r = run("engine_day_summary", S.engine_day_summary, DAY)
e = r.get("engine", {})
print("  visits:", e.get("visits"), "staff:", e.get("staff_visits"), "share%:", e.get("staff_share_pct"), "gender:", e.get("gender"), "age:", e.get("age_bands"))
print("  crossings:", j({k: v for k, v in e.get("crossings", {}).items() if k != "hourly"}))
print("  hourly:", j(e.get("crossings", {}).get("hourly"), 300))
print("  events:", e.get("events_by_type"))
print("  zones:", j(e.get("zone_visits"), 300))
print("  health transitions:", len(e.get("camera_health_transitions", [])), "open/close:", e.get("open_close"), "occupancy:", e.get("occupancy"))
print("  CE:", j(r.get("congruence")))

section(f"staff_recognition_report({DAY})")
r = run("staff_recognition_report", S.staff_recognition_report, DAY)
print("  staff events:", r.get("staff_events", {}).get("total"), "reasons:", j(r.get("staff_events", {}).get("reasons"), 200))
print("  by camera/hour:", j(r.get("staff_events", {}).get("by_camera_hour"), 300))
print("  visits:", j(r.get("visits"), 400))
print("  staff zone visits:", j(r.get("staff_zone_visits"), 300))
print("  live tracks:", len(r.get("live_tracks", [])), "model:", r.get("learned_model", {}).get("exists"), "config:", j(r.get("config"), 300))

section(f"compare_stores({FROM},{TO})")
r = run("compare_stores", S.compare_stores, FROM, TO)
for s in r.get("stores", []):
    print(f"  #{s['store_id']} {s['store']!r}: footfall={s['footfall']} uniq={s['uniqueVisitors']} conv={s['conversionPct']} eng={s['engagedPct']} "
          f"bounce={s['bounceRatePct']} dwell={s['avgDwellSec']} M/F={s['male']}/{s['female']} staffExcl={s['staffExcluded']} "
          f"late={s['lateOpenings']} sop={s['sopBreaches']} uptime={s['cameraUptimePct']} ai={s['aiBreaches']} flags={s['flags']}")
print("  rankings by_footfall:", j(r.get("rankings", {}).get("by_footfall"), 300))

section(f"analyse_all_stores({FROM},{TO})")
r = run("analyse_all_stores", S.analyse_all_stores, FROM, TO)
print("  headline:", r.get("headline"))
print("  problem counts:", r.get("problem_counts"))
for p in r.get("problems", []):
    print(f"   - [{p['severity']}] #{p['store_id']} {p['kind']}: {p['detail'][:140]}")
for rec in r.get("recommendations", []):
    print("   *", rec[:160])
print("  storeops totals:", r.get("storeops_totals"), "ai totals:", r.get("ai_totals"))

section(f"list_captures(7, {DAY})")
r = run("list_captures", S.list_captures, 7, DAY)
print("  folder:", r.get("folder"), "total:", r.get("total_frames"))
for c in r.get("cameras", []):
    print("  ", j(c))

print(f"\nDONE: {failures} failure(s)")
sys.exit(1 if failures else 0)
