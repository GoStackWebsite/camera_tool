"""Daily Excel report (openpyxl)."""
from __future__ import annotations

import io
from datetime import date, datetime, timedelta

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import AppConfig
from .store import Store

HEAD_FILL = PatternFill("solid", fgColor="DDE7F5")


def _sheet(wb: Workbook, title: str, headers: list[str], rows: list[list], first: bool = False):
    ws = wb.active if first else wb.create_sheet()
    ws.title = title
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = HEAD_FILL
    for r in rows:
        ws.append(r)
    for i, h in enumerate(headers, 1):
        width = max([len(str(h))] + [len(str(r[i - 1])) for r in rows if i - 1 < len(r)]) + 2
        ws.column_dimensions[get_column_letter(i)].width = min(60, width)
    ws.freeze_panes = "A2"
    return ws


def _fmt(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""


def _dur(s: float) -> str:
    s = int(s)
    return f"{s // 60}m {s % 60:02d}s"


def build_daily_report(store: Store, config: AppConfig, day: date) -> tuple[str, bytes]:
    start = datetime.combine(day, datetime.min.time()).timestamp()
    end = start + 86400
    thr = config.thresholds
    roles = {(c.id, z.name): z.role for c in config.cameras for z in c.zones}
    cam_names = {c.id: c.name for c in config.cameras}

    traffic = store.traffic(start, end, thr.engage_seconds)
    zones = store.zones(start, end, roles, thr.zone_engage_seconds)
    alerts = store.alerts(start, limit=5000)
    alerts = [a for a in alerts if a["started"] < end]
    health = store.health(start, min(end, datetime.now().timestamp()), [c.id for c in config.cameras])
    daily = store.daily_store(day.strftime("%Y-%m-%d")) or {}
    demo = store.demographics(start, end)
    queues = store.queues(start, end, roles)["queues"]
    peak = traffic["peak_hour"]

    wb = Workbook()
    summary = [
        ["Store", config.store.name],
        ["Date", day.isoformat()],
        ["Walk-ins", traffic["walk_ins"]],
        ["Walk-in groups", traffic["groups"]],
        ["Exits", traffic["exits"]],
        ["Visits tracked", traffic["visits"]],
        ["Engaged visits", traffic["engaged"]],
        ["Bounced visits", traffic["bounced"]],
        ["Engaged %", traffic["engaged_pct"]],
        ["Average dwell", _dur(traffic["avg_dwell"])],
        ["Peak hour", datetime.fromtimestamp(peak).strftime("%H:%M") if peak else ""],
        ["Bills (sales entered)", traffic["sales"]["bills"]],
        ["Revenue", traffic["sales"]["revenue"]],
        ["Conversion %", traffic["conversion"] if traffic["conversion"] is not None else "n/a (enter sales)"],
        ["First person seen", _fmt(daily.get("opened_at"))],
        ["Last person seen", _fmt(daily.get("closed_at"))],
        ["Alerts raised", len(alerts)],
    ]
    _sheet(wb, "Summary", ["Metric", "Value"], summary, first=True)
    _sheet(wb, "Hourly", ["Hour", "Walk-ins", "Visits", "Engaged", "Exits"],
           [[datetime.fromtimestamp(h["ts"]).strftime("%H:%M"), h["walk_ins"], h["visits"], h["engaged"], h["exits"]]
            for h in traffic["hourly"]])
    _sheet(wb, "Zones", ["Camera", "Zone", "Role", "Footfall", "Unique", "Avg stay", "Engaged %", "Share %", "To checkout %", "Rank"],
           [[cam_names.get(z["camera_id"], z["camera_id"]), z["zone"], z["role"], z["footfall"], z["unique"], _dur(z["avg_dwell"]),
             z["engaged_pct"], z["share_pct"], z["to_checkout_pct"] if z["to_checkout_pct"] is not None else "", z.get("temp", "")]
            for z in zones["zones"]])
    _sheet(wb, "Demographics", ["Age band", "Visits", "Share %", "Men", "Women", "Engaged %", "Avg dwell"],
           [[b["band"], b["count"], b["share_pct"], b["male"], b["female"], b["engaged_pct"], _dur(b["avg_dwell"])] for b in demo["bands"]]
           + [[], ["Visits with a face seen", demo["with_demographics"], demo["coverage_pct"]],
              ["Men", demo["gender"]["male"], demo["gender"]["male_pct"]],
              ["Women", demo["gender"]["female"], demo["gender"]["female_pct"]], ["Average age", demo["avg_age"] or ""]])
    _sheet(wb, "Queues", ["Camera", "Queue", "Served", "Avg wait", "Max wait", "Avg length", "Max length", "Breaches", "Breach minutes"],
           [[cam_names.get(q["camera_id"], q["camera_id"]), q["zone"], q["served"], _dur(q["avg_wait"]), _dur(q["max_wait"]), q["avg_len"], q["max_len"],
             q["breaches"], q["breach_minutes"]] for q in queues])
    _sheet(wb, "Journeys", ["Camera", "From zone", "To zone", "Count"],
           [[cam_names.get(t["camera_id"], t["camera_id"]), t["from"], t["to"], t["count"]] for t in zones["transitions"]])
    _sheet(wb, "Alerts", ["Started", "Resolved", "Type", "Camera", "Zone", "Message", "Acknowledged"],
           [[_fmt(a["started"]), _fmt(a["resolved"]), a["type"], cam_names.get(a["camera_id"], a["camera_id"] or ""),
             a["zone"] or "", a["message"], "yes" if a["acked"] else ""] for a in alerts])
    _sheet(wb, "Cameras", ["Camera", "Uptime %", "Outages", "Status", "Last change"],
           [[cam_names.get(h["camera_id"], h["camera_id"]), h["uptime_pct"], h["outages"], h["status"], _fmt(h["last_change"])]
            for h in health])
    buf = io.BytesIO()
    wb.save(buf)
    return f"store-report-{day.isoformat()}.xlsx", buf.getvalue()
