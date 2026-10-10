"""Congruence Eye MCP server (stdio).

Lets Claude analyse every store of the Congruence Eye retail-analytics platform
(NestJS REST API) together with the local camera engine (FastAPI + SQLite), and
manage the engine's learned staff-uniform model.

Run:  python D:\\cam\\mcp_server\\server.py
Config: D:\\cam\\data\\mcp.json (override with env MCP_CONFIG).

All logic lives in plain module-level functions; the @mcp.tool() wrappers at the
bottom are thin so a smoke test can call the functions directly.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# --------------------------------------------------------------------------- config

ROOT = Path(__file__).resolve().parent.parent          # D:\cam
DEFAULT_CONFIG = ROOT / "data" / "mcp.json"
DEFAULT_SNAPSHOTS = r"D:\New folder\congruence-eye-smarterasp\App_Data\uploads\snapshots"

def _local_tz():
    """Asia/Kolkata (fixed +05:30, no DST); works without the tzdata package."""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Asia/Kolkata")
    except Exception:
        return timezone(timedelta(hours=5, minutes=30), "IST")


TZ = _local_tz()
HTTP_TIMEOUT = 20
SECRET_KEYS = ("password", "api_key", "token", "secret")


def load_config() -> Dict[str, Any]:
    """Read mcp.json (path from env MCP_CONFIG). Missing keys get defaults."""
    path = Path(os.environ.get("MCP_CONFIG") or DEFAULT_CONFIG)
    cfg: Dict[str, Any] = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            print(f"[mcp] bad config {path}: {exc}", file=sys.stderr)
    cfg.setdefault("ce_url", "http://localhost:3000")
    cfg.setdefault("ce_email", "")
    cfg.setdefault("ce_password", "")
    cfg.setdefault("engine_url", "http://localhost:8000")
    cfg.setdefault("engine_db", str(ROOT / "data" / "retail.db"))
    cfg.setdefault("engine_root", str(ROOT))
    cfg.setdefault("snapshots_dir", DEFAULT_SNAPSHOTS)
    cfg["_path"] = str(path)
    return cfg


CFG = load_config()


class CEError(Exception):
    """HTTP / API failure talking to Congruence Eye."""


# --------------------------------------------------------------------------- helpers

def today() -> date:
    return datetime.now(TZ).date()


def date_range(from_date: Optional[str], to_date: Optional[str], days: int = 7) -> Tuple[str, str]:
    """Default window: last `days` days ending today (YYYY-MM-DD strings)."""
    t = today()
    to_d = date.fromisoformat(to_date) if to_date else t
    from_d = date.fromisoformat(from_date) if from_date else to_d - timedelta(days=days - 1)
    if from_d > to_d:
        from_d, to_d = to_d, from_d
    return from_d.isoformat(), to_d.isoformat()


def day_bounds(day: str) -> Tuple[float, float]:
    """Unix bounds [start, end) of a local (Asia/Kolkata) calendar day."""
    d = date.fromisoformat(day)
    start = datetime(d.year, d.month, d.day, tzinfo=TZ)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def local_time(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), TZ).strftime("%Y-%m-%d %H:%M:%S")


def local_hour(ts: float) -> int:
    return datetime.fromtimestamp(float(ts), TZ).hour


def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def age_minutes(s: Optional[str]) -> Optional[float]:
    dt = parse_iso(s)
    if not dt:
        return None
    return round((datetime.now(timezone.utc) - dt).total_seconds() / 60, 1)


def pct(part: float, whole: float) -> Optional[float]:
    return round(100.0 * part / whole, 1) if whole else None


def val(x: Any) -> Any:
    """Unwrap CE kpi objects {value: ...}."""
    if isinstance(x, dict) and "value" in x:
        return x["value"]
    return x


def mask_secrets(obj: Any) -> Any:
    """Deep-copy with password/token/api_key values replaced by '***'."""
    if isinstance(obj, dict):
        return {k: ("***" if any(s in k.lower() for s in SECRET_KEYS) and v else mask_secrets(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask_secrets(v) for v in obj]
    return obj


def err(exc: Exception) -> Dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- CE client

class CEClient:
    """Thin REST client with cached bearer token and re-login on 401."""

    def __init__(self, cfg: Dict[str, Any]):
        self.base = cfg["ce_url"].rstrip("/")
        self.email = cfg["ce_email"]
        self.password = cfg["ce_password"]
        self.token: Optional[str] = None
        self.role: Optional[str] = None
        self.session = requests.Session()

    def login(self) -> None:
        try:
            r = self.session.post(f"{self.base}/api/auth/login",
                                  json={"email": self.email, "password": self.password}, timeout=HTTP_TIMEOUT)
        except requests.RequestException as exc:
            raise CEError(f"cannot reach Congruence Eye at {self.base}: {exc}") from exc
        if r.status_code != 200:
            raise CEError(f"login failed ({r.status_code}) for {self.email}: {r.text[:200]}")
        body = r.json()
        self.token = body.get("accessToken") or body.get("token") or body.get("access_token")
        self.role = (body.get("user") or {}).get("role")
        if not self.token:
            raise CEError("login response had no accessToken")

    def get(self, path: str, params: Optional[Dict[str, Any]] = None, _retry: bool = True) -> Any:
        if not self.token:
            self.login()
        params = {k: str(v) for k, v in (params or {}).items() if v is not None}
        try:
            r = self.session.get(f"{self.base}{path}", params=params,
                                 headers={"Authorization": f"Bearer {self.token}"}, timeout=HTTP_TIMEOUT)
        except requests.RequestException as exc:
            raise CEError(f"GET {path}: {exc}") from exc
        if r.status_code == 401 and _retry:
            self.token = None
            self.login()
            return self.get(path, params, _retry=False)
        if r.status_code >= 400:
            raise CEError(f"GET {path} -> {r.status_code}: {r.text[:200]}")
        return r.json()


CE = CEClient(CFG)


def ce_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    return CE.get(path, params)


def ce_try(path: str, params: Optional[Dict[str, Any]] = None, notes: Optional[List[str]] = None, default: Any = None) -> Any:
    """GET that records the failure in `notes` instead of raising."""
    try:
        return CE.get(path, params)
    except CEError as exc:
        if notes is not None:
            notes.append(f"{path} failed: {exc}")
        return default


# --------------------------------------------------------------------------- engine client

def engine_get(path: str) -> Any:
    url = CFG["engine_url"].rstrip("/") + path
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
    except requests.RequestException as exc:
        raise CEError(f"cannot reach engine at {url}: {exc}") from exc
    if r.status_code >= 400:
        raise CEError(f"GET {url} -> {r.status_code}: {r.text[:200]}")
    return r.json()


def engine_try(path: str, notes: List[str], default: Any = None) -> Any:
    try:
        return engine_get(path)
    except CEError as exc:
        notes.append(f"engine {path} failed: {exc}")
        return default


def open_db() -> sqlite3.Connection:
    p = Path(CFG["engine_db"])
    if not p.exists():
        raise CEError(f"engine DB not found: {p}")
    con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def staff_model_info() -> Dict[str, Any]:
    p = Path(CFG["engine_root"]) / "models" / "staff_cls.pt"
    if p.exists():
        st = p.stat()
        return {"path": str(p), "exists": True, "size_kb": round(st.st_size / 1024, 1),
                "mtime": datetime.fromtimestamp(st.st_mtime, TZ).strftime("%Y-%m-%d %H:%M:%S")}
    return {"path": str(p), "exists": False, "size_kb": 0, "mtime": None}


# --------------------------------------------------------------------------- CE helpers

def get_stores(include_inactive: bool = False) -> List[Dict[str, Any]]:
    stores = ce_get("/api/stores") or []
    return [s for s in stores if include_inactive or s.get("isActive")]


def store_by_id(store_id: int) -> Optional[Dict[str, Any]]:
    for s in ce_get("/api/stores") or []:
        if int(s.get("id", -1)) == int(store_id):
            return s
    return None


def infra_map(store_ids: List[int], notes: List[str]) -> Dict[int, Dict[str, Any]]:
    if not store_ids:
        return {}
    data = ce_try("/api/infra/health", {"storeIds": ",".join(str(i) for i in store_ids)}, notes, {}) or {}
    return {int(s["storeId"]): s for s in data.get("stores", []) if "storeId" in s}


def engine_store(notes: List[str]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(CE store mapped to the engine, engine config). Match by name, then by mapped camera ids."""
    cfg = engine_try("/api/config", notes, {}) or {}
    name = (cfg.get("store") or {}).get("name")
    stores = ce_try("/api/stores", None, notes, []) or []
    for s in stores:
        if name and s.get("name") == name:
            return s, cfg
    remote_ids = {int(v) for v in (cfg.get("congruence") or {}).get("cameras", {}).values() if str(v).isdigit()}
    if remote_ids:
        for s in stores:
            cams = ce_try("/api/cameras", {"storeId": s["id"]}, notes, []) or []
            if remote_ids & {int(c["id"]) for c in cams}:
                notes.append(f"engine store name '{name}' not found in CE; mapped via camera ids to store {s['id']} '{s.get('name')}'")
                return s, cfg
    notes.append(f"engine store '{name}' could not be mapped to a CE store")
    return None, cfg


# =========================================================================== tools

def help() -> Dict[str, Any]:
    """Describe every tool of this server."""
    return {
        "server": "congruence-eye",
        "config_file": CFG["_path"],
        "ce_url": CFG["ce_url"], "engine_url": CFG["engine_url"], "engine_db": CFG["engine_db"],
        "tools": {
            "help": "This description.",
            "counting_audits(limit=20)": "Manual-count accuracy audits from the engine (counting accuracy programme).",
            "pos_status()": "Live POS bridge state: bills today, forwarding, CSV inbox.",
            "capture_rate(hours=24)": "Walk-ins vs passers-by (capture rate) and conversion from the engine.",
            "list_stores(include_inactive=False)": "All CE stores with camera/zone counts, edge device last-seen/version and infra health.",
            "store_kpis(store_id, from_date=None, to_date=None)": "Merged overview+traffic+conversion+demographics+zones+queue+storeops+AI summary for one store (default last 7 days).",
            "compare_stores(from_date=None, to_date=None, include_inactive=False)": "One row per store (footfall, conversion, bounce, dwell, gender, staffExcluded, late openings, SOP breaches, uptime, AI breaches), rankings and flags.",
            "live_status(store_id=None)": "Live occupancy/footfall/cameras for one store or every active store.",
            "detections(store_id, from_date=None, to_date=None, type=None, limit=50)": "AI-manager detections (optionally filtered by type).",
            "engine_status()": "Local engine health, Congruence connector status, staff/uniform config, learned model presence, demographics, per-camera setup.",
            "engine_day_summary(day, camera_id=None)": "Engine SQLite day summary (visits, staff, gender, age, hourly crossings, events, zone visits, camera health, open/close) with CE footfall delta.",
            "staff_recognition_report(day)": "Staff detection quality for a day: staff events per hour/camera, is_staff share, live tracks with reasons, uniform config, model presence, over/under-detection notes.",
            "feature_audit()": "Table of engine and CE features and whether each is enabled/configured.",
            "analyse_all_stores(from_date=None, to_date=None)": "Cross-store analysis: headline, per-store rows, problems and recommendations.",
            "train_staff_model(labels_csv, frames_dir, epochs=40)": "Build the staff dataset from a labels CSV + frames dir and train models/staff_cls.pt (subprocess, up to 30 min).",
            "evaluate_staff_model()": "Evaluate the existing models/staff_cls.pt on the validation split.",
            "list_captures(store_id, day)": "Count/time-span of captured snapshot frames per camera for a store-day (training material).",
        },
        "conventions": "Dates are YYYY-MM-DD (Asia/Kolkata). Every tool returns a JSON dict; failures return {'error': ...}; 'notes' lists anomalies (inactive store, offline cameras, zero data).",
    }


def list_stores(include_inactive: bool = False) -> Dict[str, Any]:
    """CE stores with camera/zone counts, edge device status and infra health."""
    notes: List[str] = []
    try:
        stores = get_stores(include_inactive)
    except CEError as exc:
        return err(exc)
    infra = infra_map([s["id"] for s in stores], notes)
    rows = []
    for s in stores:
        sid = s["id"]
        cams = ce_try("/api/cameras", {"storeId": sid}, notes, []) or []
        zones = ce_try("/api/zones", {"storeId": sid}, notes, []) or []
        edges = ce_try("/api/edge-devices", {"storeId": sid}, notes, []) or []
        edge = max(edges, key=lambda e: e.get("lastSeenAt") or "") if edges else None
        h = infra.get(sid, {})
        row = {
            "id": sid, "code": s.get("code"), "name": s.get("name"), "region": s.get("regionName"),
            "city": s.get("city"), "format": s.get("format"), "isActive": s.get("isActive"),
            "hours": f"{s.get('openTime')}-{s.get('closeTime')}",
            "cameras": len(cams), "zones": len(zones),
            "zone_types": dict(Counter(z.get("type") for z in zones)),
            "edge_devices": len(edges),
            "edge_last_seen": edge.get("lastSeenAt") if edge else None,
            "edge_last_seen_min_ago": age_minutes(edge.get("lastSeenAt")) if edge else None,
            "edge_version": edge.get("version") if edge else None,
            "health": h.get("health"), "cameras_online": h.get("camerasOnline"), "uptime_pct_7d": h.get("uptimePct7d"),
            "issues": h.get("issues", []),
        }
        if not s.get("isActive"):
            notes.append(f"store {sid} '{s.get('name')}' is inactive")
        if not cams:
            notes.append(f"store {sid} '{s.get('name')}' has no cameras")
        if not edges:
            notes.append(f"store {sid} '{s.get('name')}' has no edge device")
        rows.append(row)
    return {"count": len(rows), "stores": rows, "notes": notes}


def _store_metrics(store: Dict[str, Any], from_d: str, to_d: str, notes: List[str]) -> Dict[str, Any]:
    """Per-store numbers used by compare_stores / analyse_all_stores."""
    sid = store["id"]
    p = {"storeId": sid, "from": from_d, "to": to_d}
    ov = ce_try("/api/analytics/overview", p, notes, {}) or {}
    tr = ce_try("/api/analytics/traffic", p, notes, {}) or {}
    dm = ce_try("/api/analytics/demographics", p, notes, {}) or {}
    so = ce_try("/api/analytics/storeops", p, notes, {}) or {}
    ai = ce_try("/api/ai-manager/summary", p, notes, {}) or {}
    k = ov.get("kpis", {})
    t = tr.get("totals", {})
    g = dm.get("gender", {})
    st = so.get("totals", {})
    at = ai.get("totals", {})
    footfall = val(k.get("footfall")) or t.get("footfallIn") or 0
    daily = ov.get("daily", []) or []
    return {
        "store": store.get("name"), "store_id": sid, "code": store.get("code"), "isActive": store.get("isActive"),
        "footfall": footfall,
        "uniqueVisitors": val(k.get("uniqueVisitors")) or t.get("uniqueVisitors") or 0,
        "transactions": val(k.get("transactions")) or 0,
        "revenue": val(k.get("revenue")) or 0,
        "conversionPct": val(k.get("conversionPct")),
        "engagedPct": val(k.get("engagedPct")),
        "bounceRatePct": val(k.get("bounceRatePct")),
        "avgDwellSec": val(k.get("avgDwellSec")),
        "male": g.get("male", 0), "female": g.get("female", 0), "genderUnknown": g.get("unknown", 0),
        "staffExcluded": t.get("staffExcluded", 0),
        "lateOpenings": st.get("lateOpenings", 0), "earlyClosings": st.get("earlyClosings", 0),
        "sopBreaches": st.get("sopBreaches", 0),
        "cameraUptimePct": st.get("cameraUptimePct"), "offlineCameras": st.get("offlineCameras", 0),
        "aiBreaches": at.get("breached", 0), "aiCompliancePct": at.get("compliancePct"),
        "openAlerts": val(k.get("openAlerts")),
        "_daily": daily,
        "_openClose": so.get("openClose", []) or [],
    }


def _flags(row: Dict[str, Any]) -> List[str]:
    f = []
    if not row["footfall"]:
        f.append("zero footfall")
    if row.get("cameraUptimePct") is not None and row["cameraUptimePct"] < 95:
        f.append(f"camera uptime {row['cameraUptimePct']}% < 95%")
    if (row.get("bounceRatePct") or 0) > 80:
        f.append(f"bounce {row['bounceRatePct']}% > 80%")
    if row["footfall"] and not row.get("staffExcluded"):
        f.append("staff never detected (staffExcluded = 0 with footfall > 0)")
    if row.get("lateOpenings"):
        f.append(f"{row['lateOpenings']} late opening(s)")
    return f


def compare_stores(from_date: Optional[str] = None, to_date: Optional[str] = None,
                   include_inactive: bool = False) -> Dict[str, Any]:
    """Side-by-side KPIs for all (active) stores with rankings and flags."""
    notes: List[str] = []
    try:
        from_d, to_d = date_range(from_date, to_date)
        stores = get_stores(include_inactive)
    except (CEError, ValueError) as exc:
        return err(exc)
    rows = []
    for s in stores:
        r = _store_metrics(s, from_d, to_d, notes)
        r["flags"] = _flags(r)
        if not s.get("isActive"):
            notes.append(f"store {s['id']} '{s.get('name')}' is inactive")
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
    rows.sort(key=lambda r: r["footfall"] or 0, reverse=True)

    def rank(key: str, reverse: bool = True, only_with_data: bool = True) -> List[Dict[str, Any]]:
        cand = [r for r in rows if r.get(key) is not None and (not only_with_data or r["footfall"])]
        cand.sort(key=lambda r: r[key], reverse=reverse)
        return [{"store": r["store"], "store_id": r["store_id"], key: r[key]} for r in cand]

    rankings = {
        "by_footfall": rank("footfall", only_with_data=False),
        "by_conversionPct": rank("conversionPct"),
        "by_engagedPct": rank("engagedPct"),
        "lowest_bounceRatePct": rank("bounceRatePct", reverse=False),
        "by_avgDwellSec": rank("avgDwellSec"),
        "most_aiBreaches": rank("aiBreaches", only_with_data=False),
    }
    if not any(r["footfall"] for r in rows):
        notes.append("no store has footfall in this period")
    return {"period": {"from": from_d, "to": to_d}, "count": len(rows), "stores": rows,
            "rankings": rankings, "flagged": [{"store": r["store"], "store_id": r["store_id"], "flags": r["flags"]} for r in rows if r["flags"]],
            "notes": notes}


def store_kpis(store_id: int, from_date: Optional[str] = None, to_date: Optional[str] = None) -> Dict[str, Any]:
    """Merged analytics for one store for the period (default last 7 days)."""
    notes: List[str] = []
    try:
        from_d, to_d = date_range(from_date, to_date)
        store = store_by_id(store_id)
    except (CEError, ValueError) as exc:
        return err(exc)
    if not store:
        return {"error": f"store {store_id} not found"}
    if not store.get("isActive"):
        notes.append(f"store {store_id} '{store.get('name')}' is inactive")
    p = {"storeId": store_id, "from": from_d, "to": to_d}
    ov = ce_try("/api/analytics/overview", p, notes, {}) or {}
    tr = ce_try("/api/analytics/traffic", p, notes, {}) or {}
    cv = ce_try("/api/analytics/conversion", p, notes, {}) or {}
    dm = ce_try("/api/analytics/demographics", p, notes, {}) or {}
    zn = ce_try("/api/analytics/zones", p, notes, []) or []
    qu = ce_try("/api/analytics/queue", p, notes, {}) or {}
    so = ce_try("/api/analytics/storeops", p, notes, {}) or {}
    ai = ce_try("/api/ai-manager/summary", p, notes, {}) or {}
    kpis = {k: val(v) for k, v in (ov.get("kpis") or {}).items()}
    daily = ov.get("daily", []) or []
    if not kpis.get("footfall") and not (tr.get("totals") or {}).get("footfallIn"):
        notes.append("zero footfall in this period")
    zero_days = [d["date"] for d in daily if not d.get("footfall")]
    if zero_days:
        notes.append(f"{len(zero_days)} day(s) with zero footfall: {', '.join(zero_days)}")
    hourly = [h for h in (tr.get("hourly") or []) if h.get("footfall")]
    return {
        "store": {k: store.get(k) for k in ("id", "code", "name", "regionName", "city", "format", "isActive", "openTime", "closeTime")},
        "period": {"from": from_d, "to": to_d},
        "kpis": kpis,
        "daily": daily,
        "traffic": tr.get("totals", {}),
        "traffic_hourly_nonzero": hourly,
        "conversion": {k: cv.get(k) for k in ("funnel", "segments", "totals") if k in cv},
        "conversion_hourly_nonzero": [h for h in (cv.get("hourly") or []) if h.get("billableEntities")],
        "demographics": dm,
        "zones": zn,
        "queue": qu.get("totals", {}),
        "queue_by_counter": qu.get("byCounter", []),
        "storeops": {"totals": so.get("totals", {}), "openClose": so.get("openClose", []),
                     "events_count": len(so.get("events", []) or []),
                     "event_types": dict(Counter(e.get("type") for e in so.get("events", []) or []))},
        "ai": {"totals": ai.get("totals", {}), "byType": ai.get("byType", [])},
        "notes": notes,
    }


def live_status(store_id: Optional[int] = None) -> Dict[str, Any]:
    """Live view for one store or all active stores."""
    notes: List[str] = []
    try:
        stores = [store_by_id(store_id)] if store_id is not None else get_stores(False)
    except CEError as exc:
        return err(exc)
    if store_id is not None and not stores[0]:
        return {"error": f"store {store_id} not found"}
    out = []
    for s in stores:
        live = ce_try("/api/analytics/live", {"storeId": s["id"]}, notes, {}) or {}
        cams = live.get("cameras", []) or []
        offline = [c["name"] for c in cams if (c.get("status") or "").upper() != "ONLINE"]
        if offline:
            notes.append(f"store {s['id']} '{s.get('name')}': offline cameras: {', '.join(offline)}")
        if not s.get("isActive"):
            notes.append(f"store {s['id']} '{s.get('name')}' is inactive")
        out.append({
            "store_id": s["id"], "store": s.get("name"), "asOf": live.get("asOf"),
            "todayFootfall": live.get("todayFootfall"), "currentOccupancy": live.get("currentOccupancy"),
            "lastHourFootfall": live.get("lastHourFootfall"), "todayTransactions": live.get("todayTransactions"),
            "queues": live.get("queues", []),
            "cameras": [{"id": c.get("id"), "name": c.get("name"), "status": c.get("status"),
                         "lastHeartbeatAt": c.get("lastHeartbeatAt"), "heartbeat_min_ago": age_minutes(c.get("lastHeartbeatAt"))} for c in cams],
            "cameras_online": sum(1 for c in cams if (c.get("status") or "").upper() == "ONLINE"),
            "recentEvents": (live.get("recentEvents") or [])[:10],
        })
    return {"count": len(out), "stores": out, "notes": notes}


def detections(store_id: int, from_date: Optional[str] = None, to_date: Optional[str] = None,
               type: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """AI-manager detections for a store, newest first, optionally filtered by type."""
    notes: List[str] = []
    try:
        from_d, to_d = date_range(from_date, to_date)
        data = ce_get("/api/ai-manager/detections", {"storeId": store_id, "from": from_d, "to": to_d})
    except (CEError, ValueError) as exc:
        return err(exc)
    items = data.get("items", []) if isinstance(data, dict) else (data or [])
    if type:
        items = [i for i in items if (i.get("type") or "").upper() == type.upper()]
    by_type = dict(Counter(i.get("type") for i in items))
    by_status = dict(Counter(i.get("status") for i in items))
    if not items:
        notes.append("no detections for this filter")
    keep = ("id", "ruleName", "type", "cameraName", "zoneName", "ts", "status", "details", "snapshotUrl", "reviewStatus")
    return {"store_id": store_id, "period": {"from": from_d, "to": to_d}, "type": type,
            "total_reported": data.get("total") if isinstance(data, dict) else len(items),
            "matched": len(items), "by_type": by_type, "by_status": by_status,
            "items": [{k: i.get(k) for k in keep} for i in items[:limit]], "notes": notes}


def _camera_rows(cfg: Dict[str, Any], health: Dict[str, Any]) -> List[Dict[str, Any]]:
    hmap = {c.get("camera_id"): c for c in (health.get("cameras") or [])}
    rows = []
    for c in cfg.get("cameras", []) or []:
        roles: Dict[str, List[str]] = defaultdict(list)
        for z in c.get("zones", []) or []:
            roles[z.get("role") or "unspecified"].append(z.get("name"))
        h = hmap.get(c.get("id"), {})
        rows.append({
            "id": c.get("id"), "name": c.get("name"), "enabled": c.get("enabled"), "managed": c.get("managed"),
            "remote_id": c.get("remote_id"), "has_count_line": bool(c.get("line")),
            "zones_by_role": dict(roles), "zone_count": len(c.get("zones", []) or []),
            "frame": f"{c.get('frame_w')}x{c.get('frame_h')}" if c.get("frame_w") else None,
            "blur_faces": c.get("blur_faces"),
            "status": h.get("status"), "connected": h.get("connected"), "error": h.get("error"),
            "fps": h.get("fps"), "inference_fps": h.get("inference_fps"), "uptime_pct": h.get("uptime_pct"), "outages": h.get("outages"),
        })
    return rows


def engine_status() -> Dict[str, Any]:
    """Local engine health, connector status and staff/demographics configuration."""
    notes: List[str] = []
    health = engine_try("/api/health", notes, {}) or {}
    cfg = engine_try("/api/config", notes, {}) or {}
    conn = engine_try("/api/congruence/status", notes, {}) or {}
    if not health and not cfg:
        return {"error": "engine unreachable", "notes": notes}
    cams = _camera_rows(cfg, health)
    for c in cams:
        if c["enabled"] and not c["connected"]:
            notes.append(f"camera {c['id']} '{c['name']}' offline: {c.get('error') or c.get('status')}")
    staff = cfg.get("staff") or {}
    uniforms = staff.get("uniforms") or []
    if not uniforms:
        notes.append("no staff uniforms configured")
    if not any(c["zones_by_role"].get("staff") for c in cams):
        notes.append("no staff zones configured on any camera")
    model = staff_model_info()
    if not model["exists"]:
        notes.append("learned staff model models/staff_cls.pt not present (engine falls back to uniform colour/logo rules)")
    cg = cfg.get("congruence") or {}
    if conn and conn.get("enabled") and not conn.get("online"):
        notes.append(f"Congruence connector offline: {conn.get('last_error')}")
    if conn.get("backlog"):
        notes.append(f"connector backlog {conn.get('backlog')} (dead {conn.get('dead')})")
    return {
        "engine_url": CFG["engine_url"],
        "store": cfg.get("store"), "model": health.get("model"),
        "notifier": health.get("notifier"),
        "cameras": cams,
        "cameras_online": sum(1 for c in cams if c["connected"]), "cameras_total": len(cams),
        "staff": {"after_minutes": staff.get("after_minutes"), "uniforms": uniforms, "uniform_count": len(uniforms),
                  "learned_model": model, "staff_zone_seconds": (cfg.get("thresholds") or {}).get("staff_zone_seconds")},
        "demographics": cfg.get("demographics"),
        "thresholds": cfg.get("thresholds"),
        "connector": mask_secrets({
            "enabled": conn.get("enabled", cg.get("enabled")), "online": conn.get("online"), "last_error": conn.get("last_error"),
            "last_ok": local_time(conn.get("last_ok")), "backlog": conn.get("backlog"), "dead": conn.get("dead"),
            "sent": conn.get("sent"), "managed": conn.get("managed", cg.get("managed")), "base_url": cg.get("base_url"),
            "snapshot_seconds": cg.get("snapshot_seconds"), "heartbeat_seconds": cg.get("heartbeat_seconds"),
            "camera_map": cg.get("cameras"), "zone_map": cg.get("zones"), "sync": conn.get("sync"),
        }),
        "notes": notes,
    }


def _db_day(con: sqlite3.Connection, day: str, camera_id: Optional[str]) -> Dict[str, Any]:
    """Raw per-day aggregates from retail.db."""
    t0, t1 = day_bounds(day)
    cam_sql = " AND camera_id = ?" if camera_id else ""
    cam_arg: Tuple[Any, ...] = (camera_id,) if camera_id else ()

    def q(sql: str, *args: Any) -> List[sqlite3.Row]:
        return con.execute(sql, args).fetchall()

    visits = q(f"SELECT is_staff, gender, age_band, engaged, dwell, crossed_in, camera_id FROM visits "
               f"WHERE start_ts >= ? AND start_ts < ?{cam_sql}", t0, t1, *cam_arg)
    n_visits = len(visits)
    staff_visits = sum(1 for v in visits if v["is_staff"])
    gender = Counter((v["gender"] or "unknown") for v in visits if not v["is_staff"])
    age = Counter((v["age_band"] or "unknown") for v in visits if not v["is_staff"])
    cust = [v for v in visits if not v["is_staff"]]
    per_cam_visits = Counter(v["camera_id"] for v in visits)

    cross = q(f"SELECT ts, direction, track_id, is_staff, camera_id FROM crossings WHERE ts >= ? AND ts < ?{cam_sql}", t0, t1, *cam_arg)
    hourly: Dict[int, Dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0, "staff_in": 0})
    in_tracks, in_tracks_nonstaff, out_tracks = set(), set(), set()
    for c in cross:
        h = hourly[local_hour(c["ts"])]
        key = (c["camera_id"], c["track_id"])
        if c["direction"] == "in":
            h["in"] += 1
            in_tracks.add(key)
            if c["is_staff"]:
                h["staff_in"] += 1
            else:
                in_tracks_nonstaff.add(key)
        else:
            h["out"] += 1
            out_tracks.add(key)

    events = dict(q(f"SELECT type, COUNT(*) FROM events WHERE ts >= ? AND ts < ?{cam_sql} GROUP BY type", t0, t1, *cam_arg))
    zone_rows = q(f"SELECT zone, COUNT(*) AS visits, SUM(is_staff) AS staff, AVG(dwell) AS avg_dwell, SUM(dwell) AS total_dwell, "
                  f"COUNT(DISTINCT track_id) AS tracks FROM zone_visits WHERE enter_ts >= ? AND enter_ts < ?{cam_sql} "
                  f"GROUP BY zone ORDER BY visits DESC", t0, t1, *cam_arg)
    health_rows = q(f"SELECT camera_id, ts, status, detail FROM camera_health WHERE ts >= ? AND ts < ?{cam_sql} ORDER BY ts", t0, t1, *cam_arg)
    oc = con.execute("SELECT opened_at, closed_at FROM daily_store WHERE date = ?", (day,)).fetchone()
    minute = con.execute(f"SELECT MAX(max_occ), AVG(avg_occ), AVG(staff_avg), COUNT(*) FROM minute_stats "
                         f"WHERE ts >= ? AND ts < ?{cam_sql}", (t0, t1, *cam_arg)).fetchone()
    return {
        "visits": n_visits, "staff_visits": staff_visits, "customer_visits": n_visits - staff_visits,
        "staff_share_pct": pct(staff_visits, n_visits),
        "visits_per_camera": dict(per_cam_visits),
        "engaged_customers": sum(1 for v in cust if v["engaged"]),
        "avg_customer_dwell_sec": round(sum(v["dwell"] for v in cust) / len(cust), 1) if cust else None,
        "gender": {"M": gender.get("M", 0), "F": gender.get("F", 0), "unknown": gender.get("unknown", 0)},
        "age_bands": dict(sorted(age.items())),
        "crossings": {"in": sum(1 for c in cross if c["direction"] == "in"), "out": sum(1 for c in cross if c["direction"] == "out"),
                      "distinct_in_tracks": len(in_tracks), "distinct_in_tracks_nonstaff": len(in_tracks_nonstaff),
                      "distinct_out_tracks": len(out_tracks),
                      "hourly": [{"hour": h, **hourly[h]} for h in sorted(hourly)]},
        "events_by_type": events,
        "zone_visits": [{"zone": r["zone"], "visits": r["visits"], "staff": r["staff"], "distinct_tracks": r["tracks"],
                         "avg_dwell_sec": round(r["avg_dwell"] or 0, 1), "total_dwell_min": round((r["total_dwell"] or 0) / 60, 1)} for r in zone_rows],
        "camera_health_transitions": [{"camera_id": r["camera_id"], "time": local_time(r["ts"]), "status": r["status"], "detail": r["detail"]} for r in health_rows],
        "open_close": {"opened_at": local_time(oc["opened_at"]) if oc else None, "closed_at": local_time(oc["closed_at"]) if oc else None},
        "occupancy": {"max": minute[0], "avg": round(minute[1], 2) if minute[1] is not None else None,
                      "staff_avg": round(minute[2], 2) if minute[2] is not None else None, "minutes_recorded": minute[3]},
    }


def engine_day_summary(day: Optional[str] = None, camera_id: Optional[str] = None) -> Dict[str, Any]:
    """Engine SQLite day summary plus the mapped CE store's footfall for the same day."""
    notes: List[str] = []
    day = day or today().isoformat()
    try:
        date.fromisoformat(day)
        con = open_db()
    except (CEError, ValueError) as exc:
        return err(exc)
    try:
        data = _db_day(con, day, camera_id)
    finally:
        con.close()
    if not data["visits"] and not data["crossings"]["in"]:
        notes.append(f"engine has no visits or crossings on {day}" + (f" for camera {camera_id}" if camera_id else ""))
    if data["visits"] and not data["staff_visits"]:
        notes.append("no staff visits recorded that day")
    if data["visits"] and data["gender"]["unknown"] == data["customer_visits"]:
        notes.append("no gender classified on any visit (demographics off or faces too small)")

    ce_part: Dict[str, Any] = {}
    store, _cfg = engine_store(notes)
    if store:
        tr = ce_try("/api/analytics/traffic", {"storeId": store["id"], "from": day, "to": day}, notes, {}) or {}
        tot = tr.get("totals", {}) or {}
        ce_in = tot.get("footfallIn") or 0
        eng_in = data["crossings"]["distinct_in_tracks_nonstaff"]
        delta = eng_in - ce_in
        delta_pct = pct(abs(delta), max(ce_in, 1)) if (ce_in or eng_in) else None
        ce_part = {"store_id": store["id"], "store": store.get("name"), "footfallIn": ce_in, "footfallOut": tot.get("footfallOut"),
                   "uniqueVisitors": tot.get("uniqueVisitors"), "staffExcluded": tot.get("staffExcluded"),
                   "engine_distinct_nonstaff_in_tracks": eng_in, "delta": delta, "delta_pct": delta_pct}
        if camera_id:
            notes.append("CE footfall is store-wide while the engine numbers are filtered to one camera")
        elif delta_pct is not None and delta_pct > 20:
            notes.append(f"engine distinct non-staff 'in' tracks ({eng_in}) differ from CE footfallIn ({ce_in}) by {delta_pct}% (> 20%)")
    return {"day": day, "camera_id": camera_id, "engine": data, "congruence": ce_part, "notes": notes}


def staff_recognition_report(day: Optional[str] = None) -> Dict[str, Any]:
    """How well staff are being recognised on a given day."""
    notes: List[str] = []
    day = day or today().isoformat()
    try:
        t0, t1 = day_bounds(day)
        con = open_db()
    except (CEError, ValueError) as exc:
        return err(exc)
    try:
        ev = con.execute("SELECT ts, camera_id, message FROM events WHERE type = 'staff' AND ts >= ? AND ts < ?", (t0, t1)).fetchall()
        per_hour_cam: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        reasons: Counter = Counter()
        for r in ev:
            per_hour_cam[r["camera_id"]][local_hour(r["ts"])] += 1
            msg = r["message"] or ""
            m = re.search(r"\((.*?)\)", msg)
            reasons[m.group(1) if m else re.sub(r"#\d+", "#N", msg)[:60]] += 1
        vis = con.execute("SELECT camera_id, is_staff, COUNT(*) AS n, AVG(dwell) AS d FROM visits WHERE start_ts >= ? AND start_ts < ? "
                          "GROUP BY camera_id, is_staff", (t0, t1)).fetchall()
        zs = con.execute("SELECT zone, SUM(is_staff) AS staff, COUNT(*) AS n FROM zone_visits WHERE enter_ts >= ? AND enter_ts < ? GROUP BY zone",
                         (t0, t1)).fetchall()
    finally:
        con.close()
    per_cam: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"visits": 0, "staff_visits": 0, "avg_staff_dwell_sec": None, "avg_customer_dwell_sec": None})
    for r in vis:
        c = per_cam[r["camera_id"]]
        c["visits"] += r["n"]
        if r["is_staff"]:
            c["staff_visits"] += r["n"]
            c["avg_staff_dwell_sec"] = round(r["d"] or 0, 1)
        else:
            c["avg_customer_dwell_sec"] = round(r["d"] or 0, 1)
    for c in per_cam.values():
        c["staff_share_pct"] = pct(c["staff_visits"], c["visits"])
    total_visits = sum(c["visits"] for c in per_cam.values())
    total_staff = sum(c["staff_visits"] for c in per_cam.values())
    share = pct(total_staff, total_visits)

    cfg = engine_try("/api/config", notes, {}) or {}
    stats = engine_try("/api/stats", notes, {}) or {}
    live_tracks = []
    for cid, cam in (stats.get("cameras") or {}).items():
        for t in cam.get("tracks", []) or []:
            live_tracks.append({"camera_id": cid, "track": t.get("id"), "staff": t.get("staff"), "staff_reason": t.get("staff_reason"),
                                "votes": t.get("votes"), "logo_pct": t.get("logo_pct"), "gender": t.get("gender"), "age_band": t.get("age_band")})
    staff_cfg = cfg.get("staff") or {}
    uniforms = staff_cfg.get("uniforms") or []
    staff_zones = [f"{c.get('id')}/{z.get('name')}" for c in cfg.get("cameras", []) or [] for z in c.get("zones", []) or [] if z.get("role") == "staff"]
    demo = (cfg.get("demographics") or {}).get("enabled")
    model = staff_model_info()

    if not uniforms:
        notes.append("no staff uniforms configured: staff can only be inferred from staff zones / dwell time")
    if not staff_zones:
        notes.append("no staff zones configured")
    if demo is False:
        notes.append("demographics disabled")
    if not model["exists"]:
        notes.append("learned staff model not present")
    if total_visits == 0:
        notes.append(f"no visits recorded on {day}")
    elif share is not None and share > 40:
        notes.append(f"staff share {share}% of visits > 40%: likely over-detection (uniform colour too generic or staff zone covers customer area)")
    elif share is not None and share < 3:
        notes.append(f"staff share {share}% of visits < 3%: likely under-detection (check uniform HSV / logo settings, staff zones)")
    if not live_tracks:
        notes.append("no live tracks (cameras offline or store empty)")
    return {
        "day": day,
        "staff_events": {"total": len(ev), "by_camera_hour": {c: dict(sorted(h.items())) for c, h in per_hour_cam.items()}, "reasons": dict(reasons.most_common(10))},
        "visits": {"total": total_visits, "staff": total_staff, "staff_share_pct": share, "per_camera": dict(per_cam)},
        "staff_zone_visits": [{"zone": r["zone"], "visits": r["n"], "staff": r["staff"], "staff_pct": pct(r["staff"] or 0, r["n"])} for r in zs],
        "live_tracks": live_tracks,
        "config": {"after_minutes": staff_cfg.get("after_minutes"), "uniforms": uniforms, "staff_zones": staff_zones,
                   "staff_zone_seconds": (cfg.get("thresholds") or {}).get("staff_zone_seconds"), "demographics_enabled": demo},
        "learned_model": model,
        "notes": notes,
    }


def feature_audit() -> Dict[str, Any]:
    """Which engine and CE features are enabled / configured."""
    notes: List[str] = []
    cfg = engine_try("/api/config", notes, {}) or {}
    health = engine_try("/api/health", notes, {}) or {}
    conn = engine_try("/api/congruence/status", notes, {}) or {}
    rows: List[Dict[str, Any]] = []

    def add(area: str, feature: str, status: str, detail: Any = "") -> None:
        rows.append({"area": area, "feature": feature, "status": status, "detail": detail})

    cams = _camera_rows(cfg, health)
    th = cfg.get("thresholds") or {}
    for c in cams:
        add("engine", f"camera {c['id']} ({c['name']})", "enabled" if c["enabled"] else "disabled",
            f"{'online' if c['connected'] else 'OFFLINE'}, frame {c['frame']}, blur_faces={'on' if c['blur_faces'] else 'off'}")
        add("engine", f"  count line on {c['id']}", "configured" if c["has_count_line"] else "MISSING", "footfall in/out counting")
        add("engine", f"  zones on {c['id']}", "configured" if c["zone_count"] else "MISSING", c["zones_by_role"] or "no zones")
    if not cams:
        add("engine", "cameras", "MISSING", "no cameras configured")
    uniforms = (cfg.get("staff") or {}).get("uniforms") or []
    add("engine", "staff uniforms", "configured" if uniforms else "MISSING", [u.get("name") for u in uniforms])
    staff_zones = [z for c in cams for z in c["zones_by_role"].get("staff", [])]
    add("engine", "staff zones", "configured" if staff_zones else "MISSING", staff_zones)
    model = staff_model_info()
    add("engine", "learned staff model", "present" if model["exists"] else "MISSING", f"{model['path']} ({model['mtime']})" if model["exists"] else model["path"])
    add("engine", "staff by dwell time", "enabled" if (cfg.get("staff") or {}).get("after_minutes") else "disabled", f"after_minutes={(cfg.get('staff') or {}).get('after_minutes')}")
    demo = cfg.get("demographics") or {}
    add("engine", "demographics (gender/age)", "enabled" if demo.get("enabled") else "disabled", {k: v for k, v in demo.items() if k != "enabled"})
    for key, label in (("phone_seconds", "phone usage"), ("loiter_seconds", "loitering"), ("unattended_seconds", "unattended customer"),
                       ("crowd_threshold", "crowd"), ("queue_max", "queue breach"), ("engage_seconds", "engagement"), ("run_speed", "running")):
        add("engine", f"{label} threshold", "configured" if th.get(key) else "MISSING", {key: th.get(key)})
    nt = cfg.get("notifications") or {}
    channels = {"telegram": bool(nt.get("telegram_bot_token") and nt.get("telegram_chat_id")),
                "webhook": bool(nt.get("webhook_url")),
                "smtp": bool(nt.get("smtp_host") and nt.get("email_to"))}
    add("engine", "notifier channels", "configured" if any(channels.values()) else "MISSING",
        {"channels": channels, "types": nt.get("types"), "sent": (health.get("notifier") or {}).get("sent")})
    rep = cfg.get("report") or {}
    add("engine", "daily email report", "enabled" if rep.get("enabled") else "disabled", {"daily_time": rep.get("daily_time"), "email_to": rep.get("email_to") or "(none)"})
    cg = cfg.get("congruence") or {}
    add("engine", "Congruence connector", "enabled" if cg.get("enabled") else "disabled",
        {"managed": cg.get("managed"), "online": conn.get("online"), "snapshot_seconds": cg.get("snapshot_seconds"),
         "heartbeat_seconds": cg.get("heartbeat_seconds"), "mapped_cameras": conn.get("mapped_cameras"), "mapped_zones": conn.get("mapped_zones"),
         "backlog": conn.get("backlog"), "last_error": conn.get("last_error")})
    add("engine", "face blurring", "enabled" if cams and all(c["blur_faces"] for c in cams) else ("partial" if any(c["blur_faces"] for c in cams) else "disabled"),
        {c["id"]: c["blur_faces"] for c in cams})
    add("engine", "detector model", "configured" if health.get("model") else "MISSING", health.get("model"))

    # ---- Congruence Eye
    settings = ce_try("/api/settings", None, notes, {}) or {}
    stores = ce_try("/api/stores", None, notes, []) or []
    active = [s for s in stores if s.get("isActive")]
    rules = ce_try("/api/ai-manager/rules", None, notes, []) or []
    if settings:
        add("ce", "alert email", "enabled" if settings.get("alertEmailEnabled") else "disabled", "")
        add("ce", "bounceSec", "configured", settings.get("bounceSec"))
        add("ce", "engagedStoreDwellSec", "configured", settings.get("engagedStoreDwellSec"))
        add("ce", "repeatWindowDays", "configured", settings.get("repeatWindowDays"))
        add("ce", "aiServiceUrl", "configured" if settings.get("aiServiceUrl") else "MISSING", settings.get("aiServiceUrl"))
        add("ce", "company / currency", "configured", f"{settings.get('companyName')} / {settings.get('currency')}")
    add("ce", "stores", "configured" if active else "MISSING", f"{len(active)} active of {len(stores)}")
    cam_total = zone_total = edge_total = 0
    per_store = []
    for s in active:
        cams_ce = ce_try("/api/cameras", {"storeId": s["id"]}, notes, []) or []
        zones_ce = ce_try("/api/zones", {"storeId": s["id"]}, notes, []) or []
        edges = ce_try("/api/edge-devices", {"storeId": s["id"]}, notes, []) or []
        cam_total += len(cams_ce)
        zone_total += len(zones_ce)
        edge_total += len(edges)
        zt = dict(Counter(z.get("type") for z in zones_ce))
        per_store.append({"store_id": s["id"], "store": s.get("name"), "cameras": len(cams_ce), "zones": len(zones_ce), "zone_types": zt, "edge_devices": len(edges)})
        add("ce", f"store {s['id']} '{s.get('name')}' zone types", "configured" if zt.get("ENTRY_LINE") else ("partial" if zt else "MISSING"),
            zt if zt else "no zones (no ENTRY_LINE: no footfall)")
    add("ce", "cameras / zones / edge devices (active stores)", "configured" if cam_total else "MISSING",
        {"cameras": cam_total, "zones": zone_total, "edge_devices": edge_total})
    active_rules = [r for r in rules if r.get("isActive")]
    add("ce", "AI rules", "configured" if active_rules else "MISSING", {"active": len(active_rules), "total": len(rules), "types": dict(Counter(r.get("type") for r in active_rules))})
    disabled = [r["feature"] for r in rows if r["status"] in ("MISSING", "disabled")]
    return {"features": rows, "per_store": per_store, "disabled_or_missing": disabled, "notes": notes}


def analyse_all_stores(from_date: Optional[str] = None, to_date: Optional[str] = None) -> Dict[str, Any]:
    """Cross-store analysis with headline, problems and recommendations."""
    notes: List[str] = []
    try:
        from_d, to_d = date_range(from_date, to_date)
        stores = get_stores(False)
    except (CEError, ValueError) as exc:
        return err(exc)
    if not stores:
        return {"error": "no active stores", "notes": notes}
    ids = [s["id"] for s in stores]
    infra = infra_map(ids, notes)
    infra_tot = (ce_try("/api/infra/health", {"storeIds": ",".join(map(str, ids))}, notes, {}) or {}).get("totals", {})
    ai_all = ce_try("/api/ai-manager/summary", {"storeIds": ",".join(map(str, ids)), "from": from_d, "to": to_d}, notes, {}) or {}
    so_all = ce_try("/api/analytics/storeops", {"storeIds": ",".join(map(str, ids)), "from": from_d, "to": to_d}, notes, {}) or {}

    problems: List[Dict[str, Any]] = []
    recs: List[str] = []

    def prob(store: Dict[str, Any], kind: str, detail: str, severity: str = "medium") -> None:
        problems.append({"store_id": store["id"], "store": store.get("name"), "kind": kind, "severity": severity, "detail": detail})

    rows = []
    for s in stores:
        m = _store_metrics(s, from_d, to_d, notes)
        m["flags"] = _flags(m)
        h = infra.get(s["id"], {})
        cams = ce_try("/api/cameras", {"storeId": s["id"]}, notes, []) or []
        zones = ce_try("/api/zones", {"storeId": s["id"]}, notes, []) or []
        edges = ce_try("/api/edge-devices", {"storeId": s["id"]}, notes, []) or []
        zone_types = Counter(z.get("type") for z in zones if z.get("isActive", True))
        cams_with_zones = {z.get("cameraId") for z in zones}
        m.update({"health": h.get("health"), "camerasOnline": h.get("camerasOnline"), "camerasTotal": h.get("camerasTotal"),
                  "uptimePct7d": h.get("uptimePct7d"), "edgeLastSeenAt": h.get("edgeLastSeenAt"), "issues": h.get("issues", []),
                  "zone_types": dict(zone_types)})

        if h and (h.get("camerasOnline") or 0) < (h.get("camerasTotal") or 0):
            prob(s, "offline_cameras", f"{h.get('camerasTotal', 0) - (h.get('camerasOnline') or 0)} of {h.get('camerasTotal')} cameras offline", "high")
        late = [o for o in m["_openClose"] if (o.get("lateOpenMin") or 0) > 0]
        if late:
            prob(s, "late_openings", "; ".join(f"{o['date']} opened {o.get('actualOpen')} ({o['lateOpenMin']} min late)" for o in late))
        early = [o for o in m["_openClose"] if (o.get("earlyCloseMin") or 0) > 0]
        if early:
            prob(s, "early_closings", "; ".join(f"{o['date']} closed {o.get('actualClose')} ({o['earlyCloseMin']} min early)" for o in early))
        zero_days = [d["date"] for d in m["_daily"] if not d.get("footfall")]
        if zero_days:
            prob(s, "zero_footfall_days", f"{len(zero_days)} of {len(m['_daily'])} days with zero footfall: {', '.join(zero_days)}",
                 "high" if len(zero_days) == len(m["_daily"]) else "medium")
        if (m.get("bounceRatePct") or 0) > 80:
            prob(s, "high_bounce", f"bounce rate {m['bounceRatePct']}% (> 80%)")
        if m["footfall"] and not m.get("staffExcluded"):
            prob(s, "staff_never_detected", "footfall > 0 but staffExcluded = 0: staff counted as customers")
        if not zone_types.get("ENTRY_LINE"):
            prob(s, "no_entry_line", "store has no ENTRY_LINE zone: footfall cannot be counted", "high")
        for c in cams:
            if c.get("id") not in cams_with_zones:
                prob(s, "camera_without_zones", f"camera {c.get('id')} '{c.get('name')}' has no zones")
        if not edges:
            prob(s, "no_edge_device", "no edge device registered", "high")
        for e in edges:
            mins = age_minutes(e.get("lastSeenAt"))
            if mins is None or mins > 60:
                prob(s, "edge_not_seen", f"edge '{e.get('name')}' last seen {e.get('lastSeenAt')} ({mins} min ago)", "high")
        if m.get("cameraUptimePct") is not None and m["cameraUptimePct"] < 95:
            prob(s, "low_camera_uptime", f"camera uptime {m['cameraUptimePct']}% in period (< 95%)")
        rows.append({k: v for k, v in m.items() if not k.startswith("_")})
    rows.sort(key=lambda r: r["footfall"] or 0, reverse=True)

    kinds = Counter(p["kind"] for p in problems)
    if kinds.get("offline_cameras") or kinds.get("low_camera_uptime"):
        recs.append("Restore camera streams (check RTSP URL/credentials/network) and watch uptime; footfall and all downstream KPIs are unreliable while cameras are offline.")
    if kinds.get("no_entry_line"):
        recs.append("Draw an ENTRY_LINE zone on the entrance camera of every store so footfall and conversion can be computed.")
    if kinds.get("camera_without_zones"):
        recs.append("Add SECTION/QUEUE/STAFF_ONLY zones to cameras that have none; without zones they only contribute occupancy.")
    if kinds.get("staff_never_detected"):
        recs.append("Configure staff uniforms / STAFF_ONLY zones (or train the staff model) so staff are excluded from footfall and conversion.")
    if kinds.get("high_bounce"):
        recs.append("Bounce > 80%: verify the count line placement and bounceSec; if real, review entrance merchandising and greeter presence.")
    if kinds.get("late_openings") or kinds.get("early_closings"):
        recs.append("Review store open/close SOP: late openings / early closings detected (compare scheduled vs actual times in storeops).")
    if kinds.get("edge_not_seen") or kinds.get("no_edge_device"):
        recs.append("Edge devices not heartbeating: check the engine process / connector on the store PC.")
    if kinds.get("zero_footfall_days"):
        recs.append("Days with zero footfall usually mean the edge was down or the store was closed; confirm with camera health and open/close records.")
    total_ff = sum(r["footfall"] or 0 for r in rows)
    with_data = [r for r in rows if r["footfall"]]
    top = rows[0] if rows and rows[0]["footfall"] else None
    headline = (f"{len(stores)} active store(s), {from_d} to {to_d}: total footfall {total_ff} across {len(with_data)} store(s) with data; "
                f"infra {infra_tot.get('healthy', 0)} healthy / {infra_tot.get('degraded', 0)} degraded / {infra_tot.get('down', 0)} down, "
                f"{infra_tot.get('camerasOnline', 0)}/{infra_tot.get('camerasTotal', 0)} cameras online; "
                f"{len(problems)} problem(s) found.")
    if top:
        headline += f" Busiest: {top['store']} ({top['footfall']} footfall, bounce {top['bounceRatePct']}%, conversion {top['conversionPct']}%)."
    return {
        "period": {"from": from_d, "to": to_d},
        "headline": headline,
        "stores": rows,
        "infra_totals": infra_tot,
        "storeops_totals": so_all.get("totals", {}),
        "ai_totals": ai_all.get("totals", {}), "ai_by_store": ai_all.get("byStore", []), "ai_by_type": ai_all.get("byType", []),
        "problems": sorted(problems, key=lambda p: (p["severity"] != "high", p["store_id"], p["kind"])),
        "problem_counts": dict(kinds),
        "recommendations": recs,
        "notes": notes,
    }


# --------------------------------------------------------------------------- staff model tooling

METRICS_RE = re.compile(r"val:\s*staff recall\s*([\d.]+)%\s*\((\d+)/(\d+)\),\s*precision\s*([\d.]+)%,\s*customer specificity\s*(\d+)/(\d+)")


def parse_metrics(text: str) -> Optional[Dict[str, Any]]:
    """Parse 'val: staff recall X% (a/b), precision Y%, customer specificity c/d'."""
    m = None
    for m in METRICS_RE.finditer(text):
        pass
    if not m:
        return None
    return {"staff_recall_pct": float(m.group(1)), "staff_recall": f"{m.group(2)}/{m.group(3)}",
            "precision_pct": float(m.group(4)), "customer_specificity": f"{m.group(5)}/{m.group(6)}", "line": m.group(0)}


def run_staff_tool(args: List[str], timeout: int) -> Dict[str, Any]:
    root = Path(CFG["engine_root"])
    py = root / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)
    script = root / "tools" / "staff_train.py"
    if not script.exists():
        return {"error": f"{script} not found"}
    cmd = [str(py), str(script), *args]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s", "command": cmd}
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    tail = out.strip().splitlines()[-25:]
    return {"command": cmd, "returncode": p.returncode, "seconds": round(time.time() - t0, 1),
            "metrics": parse_metrics(out), "output_tail": tail}


def train_staff_model(labels_csv: str, frames_dir: str, epochs: int = 40) -> Dict[str, Any]:
    """Build datasets/staff from labels + frames, then train models/staff_cls.pt."""
    notes: List[str] = []
    if not Path(labels_csv).exists():
        return {"error": f"labels CSV not found: {labels_csv}"}
    if not Path(frames_dir).is_dir():
        return {"error": f"frames dir not found: {frames_dir}"}
    build = run_staff_tool(["--build", labels_csv, "--frames", frames_dir], timeout=30 * 60)
    if build.get("error") or build.get("returncode"):
        return {"stage": "build", "build": build, "error": build.get("error") or f"build exited {build.get('returncode')}"}
    train = run_staff_tool(["--train", "--epochs", str(int(epochs)), "--imgsz", "128"], timeout=30 * 60)
    model = staff_model_info()
    if train.get("error") or train.get("returncode"):
        return {"stage": "train", "build": build, "train": train, "model": model, "error": train.get("error") or f"train exited {train.get('returncode')}"}
    if not train.get("metrics"):
        notes.append("could not parse the 'val:' metrics line from the trainer output")
    if not model["exists"]:
        notes.append("training finished but models/staff_cls.pt is missing")
    return {"stage": "done", "build_tail": build["output_tail"], "metrics": train.get("metrics"), "train_seconds": train["seconds"],
            "model": model, "train_tail": train["output_tail"], "notes": notes}


def evaluate_staff_model() -> Dict[str, Any]:
    """Evaluate the existing learned staff model."""
    model = staff_model_info()
    if not model["exists"]:
        return {"error": "models/staff_cls.pt not present; train it first", "model": model}
    res = run_staff_tool(["--eval"], timeout=10 * 60)
    if res.get("error"):
        return res
    notes = [] if res.get("metrics") else ["could not parse the 'val:' metrics line"]
    return {"model": model, "metrics": res.get("metrics"), "returncode": res["returncode"], "output_tail": res["output_tail"], "notes": notes}


CAPTURE_RE = re.compile(r"^cam(\d+)-(\d{10,13})-([0-9a-f]+)\.(jpe?g|png)$", re.IGNORECASE)


def list_captures(store_id: int, day: Optional[str] = None) -> Dict[str, Any]:
    """Captured snapshot frames per camera for a store-day (training material)."""
    notes: List[str] = []
    day = day or today().isoformat()
    folder = Path(CFG["snapshots_dir"]) / str(store_id) / day
    if not folder.is_dir():
        parent = Path(CFG["snapshots_dir"]) / str(store_id)
        days = sorted(p.name for p in parent.iterdir() if p.is_dir()) if parent.is_dir() else []
        return {"store_id": store_id, "day": day, "folder": str(folder), "cameras": {}, "total_frames": 0,
                "notes": [f"no capture folder for store {store_id} on {day}", f"available days: {days[-10:]}" if days else "no captures at all for this store"]}
    per: Dict[str, Dict[str, Any]] = {}
    other = 0
    for p in folder.iterdir():
        m = CAPTURE_RE.match(p.name)
        if not m:
            other += 1
            continue
        cam, ms = m.group(1), int(m.group(2))
        ts = ms / 1000 if ms > 10**11 else float(ms)
        d = per.setdefault(cam, {"camera_id": int(cam), "frames": 0, "first_ts": ts, "last_ts": ts, "bytes": 0})
        d["frames"] += 1
        d["first_ts"] = min(d["first_ts"], ts)
        d["last_ts"] = max(d["last_ts"], ts)
        d["bytes"] += p.stat().st_size
    cams = []
    for d in per.values():
        span_min = (d["last_ts"] - d["first_ts"]) / 60
        cams.append({"camera_id": d["camera_id"], "frames": d["frames"], "first": local_time(d["first_ts"]), "last": local_time(d["last_ts"]),
                     "span_minutes": round(span_min, 1), "avg_interval_sec": round(span_min * 60 / max(d["frames"] - 1, 1), 1),
                     "size_mb": round(d["bytes"] / 1e6, 1)})
    cams.sort(key=lambda c: c["camera_id"])
    if other:
        notes.append(f"{other} file(s) did not match the cam<id>-<unixms>-<hash>.jpg pattern")
    if not cams:
        notes.append("no frames in folder")
    return {"store_id": store_id, "day": day, "folder": str(folder), "cameras": cams,
            "total_frames": sum(c["frames"] for c in cams), "notes": notes}


# =========================================================================== MCP registration

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:
    try:  # mcp 2.x renamed FastMCP to MCPServer
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:  # smoke tests may import this module without mcp installed
        _Server = None  # type: ignore

def counting_audits(limit: int = 20) -> Dict[str, Any]:
    """Manual-count accuracy audits recorded on the engine (Settings > Counting accuracy audit)."""
    notes: List[str] = []
    rows = engine_try("/api/audits?limit=%d" % limit, notes, []) or []
    acc = [r["accuracy_in"] for r in rows if r.get("accuracy_in") is not None]
    if not rows:
        notes.append("no audits yet: count by hand for 15-30 minutes in the engine's Settings to measure counting accuracy")
    elif acc and sum(acc) / len(acc) < 95:
        notes.append("average IN accuracy %.1f%% is below the 95%% leaders quote: check line placement, margin and camera angle" % (sum(acc) / len(acc)))
    return {"audits": rows, "avg_accuracy_in": round(sum(acc) / len(acc), 1) if acc else None, "notes": notes}


def pos_status() -> Dict[str, Any]:
    """Live POS bridge state on the engine: bills received today, forwarding, CSV inbox, last bills."""
    notes: List[str] = []
    st = engine_try("/api/pos/status", notes, {}) or {}
    bills = engine_try("/api/pos/transactions?hours=24&limit=20", notes, []) or []
    if st and not st.get("today", {}).get("bills"):
        notes.append("no bills today: connect the billing software to POST /api/pos/transactions or drop CSV exports into the inbox folder")
    if st.get("last_error"):
        notes.append("last error: %s" % st["last_error"])
    return {"status": st, "last_bills": bills, "notes": notes}


def capture_rate(hours: float = 24) -> Dict[str, Any]:
    """Walk-ins, passers-by and capture rate (needs a 'Pass-by' ENTRY_LINE zone on a street-facing camera), with conversion."""
    notes: List[str] = []
    t = engine_try("/api/traffic?hours=%s" % hours, notes, {}) or {}
    if t and not t.get("passby"):
        notes.append("no passers-by counted: draw an ENTRY_LINE named 'Pass-by' across the street / mall side in Congruence Eye > Admin > Zones and press Sync now")
    return {"hours": hours, "walk_ins": t.get("walk_ins"), "passby": t.get("passby"), "capture_rate_pct": t.get("capture_rate_pct"),
            "conversion_pct": t.get("conversion"), "sales": t.get("sales"), "notes": notes}


mcp = _Server("congruence-eye", instructions="Analyse Congruence Eye stores and the local camera engine; call `help` for the tool list.") if _Server else None

if mcp:
    @mcp.tool(name="help")
    def help_tool() -> dict:
        """Describe all tools of the congruence-eye server, config and conventions."""
        return help()

    @mcp.tool(name="counting_audits")
    def counting_audits_tool(limit: int = 20) -> dict:
        """Manual-count accuracy audits recorded on the engine, with the average IN accuracy."""
        return counting_audits(limit)

    @mcp.tool(name="pos_status")
    def pos_status_tool() -> dict:
        """Live POS bridge: bills received today, forwarding to Congruence Eye, CSV inbox state, last bills."""
        return pos_status()

    @mcp.tool(name="capture_rate")
    def capture_rate_tool(hours: float = 24) -> dict:
        """Walk-ins vs passers-by (capture rate) and conversion from the engine for the last N hours."""
        return capture_rate(hours)

    @mcp.tool(name="list_stores")
    def list_stores_tool(include_inactive: bool = False) -> dict:
        """List Congruence Eye stores with camera/zone counts, edge device last-seen/version and infra health."""
        return list_stores(include_inactive)

    @mcp.tool(name="store_kpis")
    def store_kpis_tool(store_id: int, from_date: Optional[str] = None, to_date: Optional[str] = None) -> dict:
        """Merged KPIs (overview, traffic, conversion, demographics, zones, queue, storeops, AI summary) for one store. Dates YYYY-MM-DD, default last 7 days."""
        return store_kpis(store_id, from_date, to_date)

    @mcp.tool(name="compare_stores")
    def compare_stores_tool(from_date: Optional[str] = None, to_date: Optional[str] = None, include_inactive: bool = False) -> dict:
        """Compare all stores side by side (footfall, conversion, bounce, dwell, gender, staff, SOP, uptime, AI breaches) with rankings and flags."""
        return compare_stores(from_date, to_date, include_inactive)

    @mcp.tool(name="live_status")
    def live_status_tool(store_id: Optional[int] = None) -> dict:
        """Live occupancy, today's footfall, queues and camera status for one store or all active stores."""
        return live_status(store_id)

    @mcp.tool(name="detections")
    def detections_tool(store_id: int, from_date: Optional[str] = None, to_date: Optional[str] = None, type: Optional[str] = None, limit: int = 50) -> dict:
        """AI-manager detections for a store (e.g. type=MOBILE_USAGE, UNATTENDED_CUSTOMER), newest first."""
        return detections(store_id, from_date, to_date, type, limit)

    @mcp.tool(name="engine_status")
    def engine_status_tool() -> dict:
        """Local camera engine: health, Congruence connector, staff uniforms + learned model, demographics, per-camera setup."""
        return engine_status()

    @mcp.tool(name="engine_day_summary")
    def engine_day_summary_tool(day: Optional[str] = None, camera_id: Optional[str] = None) -> dict:
        """Engine SQLite summary for a day (visits, staff, gender/age, hourly crossings, events, zones, camera health, open/close) plus CE footfall delta."""
        return engine_day_summary(day, camera_id)

    @mcp.tool(name="staff_recognition_report")
    def staff_recognition_report_tool(day: Optional[str] = None) -> dict:
        """Staff recognition quality for a day: staff events, is_staff share, live tracks with reasons, uniform config, model presence, over/under-detection notes."""
        return staff_recognition_report(day)

    @mcp.tool(name="feature_audit")
    def feature_audit_tool() -> dict:
        """Audit which engine and Congruence Eye features are enabled, configured, disabled or missing."""
        return feature_audit()

    @mcp.tool(name="analyse_all_stores")
    def analyse_all_stores_tool(from_date: Optional[str] = None, to_date: Optional[str] = None) -> dict:
        """Cross-store analysis: headline, per-store rows, infra/storeops/AI totals, problems and recommendations."""
        return analyse_all_stores(from_date, to_date)

    @mcp.tool(name="train_staff_model")
    def train_staff_model_tool(labels_csv: str, frames_dir: str, epochs: int = 40) -> dict:
        """Build the staff dataset from a labels CSV (file,x1,y1,x2,y2,label) + frames dir and train models/staff_cls.pt (may take up to 30 min)."""
        return train_staff_model(labels_csv, frames_dir, epochs)

    @mcp.tool(name="evaluate_staff_model")
    def evaluate_staff_model_tool() -> dict:
        """Evaluate the existing learned staff model (recall / precision / specificity)."""
        return evaluate_staff_model()

    @mcp.tool(name="list_captures")
    def list_captures_tool(store_id: int, day: Optional[str] = None) -> dict:
        """Count and time span of captured snapshot frames per camera for a store-day, to pick training material."""
        return list_captures(store_id, day)


if __name__ == "__main__":
    if not mcp:
        print("mcp package not installed", file=sys.stderr)
        sys.exit(1)
    mcp.run(transport="stdio")
