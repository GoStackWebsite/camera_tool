"""SQLite persistence: events, per-minute stats, visits, zone visits, crossings, alerts, health, sales."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, camera_id TEXT NOT NULL,
    type TEXT NOT NULL, track_id INTEGER, zone TEXT, message TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS minute_stats (
    camera_id TEXT NOT NULL, ts INTEGER NOT NULL, avg_occ REAL NOT NULL, max_occ INTEGER NOT NULL,
    entries INTEGER NOT NULL, exits INTEGER NOT NULL, new_visitors INTEGER NOT NULL, staff_avg REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (camera_id, ts));
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, track_id INTEGER NOT NULL,
    start_ts REAL NOT NULL, end_ts REAL NOT NULL, dwell REAL NOT NULL, engaged INTEGER NOT NULL,
    is_staff INTEGER NOT NULL, zones TEXT NOT NULL, crossed_in INTEGER NOT NULL, group_id TEXT,
    gender TEXT, age INTEGER, age_band TEXT);
CREATE INDEX IF NOT EXISTS idx_visits_start ON visits(start_ts);
CREATE TABLE IF NOT EXISTS zone_visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, zone TEXT NOT NULL, track_id INTEGER NOT NULL,
    enter_ts REAL NOT NULL, exit_ts REAL NOT NULL, dwell REAL NOT NULL, is_staff INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_zone_visits_enter ON zone_visits(enter_ts);
CREATE TABLE IF NOT EXISTS crossings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, ts REAL NOT NULL, direction TEXT NOT NULL,
    track_id INTEGER NOT NULL, group_id TEXT NOT NULL, is_staff INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_crossings_ts ON crossings(ts);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, camera_id TEXT, type TEXT NOT NULL, zone TEXT,
    track_id INTEGER, message TEXT NOT NULL, started REAL NOT NULL, resolved REAL, acked INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS idx_alerts_started ON alerts(started);
CREATE TABLE IF NOT EXISTS camera_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, ts REAL NOT NULL, status TEXT NOT NULL, detail TEXT);
CREATE TABLE IF NOT EXISTS sales (date TEXT PRIMARY KEY, bills INTEGER NOT NULL, revenue REAL NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS queue_minutes (
    camera_id TEXT NOT NULL, zone TEXT NOT NULL, ts INTEGER NOT NULL, avg_len REAL NOT NULL, max_len INTEGER NOT NULL,
    avg_wait REAL NOT NULL, max_wait REAL NOT NULL, PRIMARY KEY (camera_id, zone, ts));
CREATE TABLE IF NOT EXISTS daily_store (date TEXT PRIMARY KEY, opened_at REAL, closed_at REAL);
CREATE TABLE IF NOT EXISTS audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, start_ts REAL NOT NULL, end_ts REAL NOT NULL,
    manual_in INTEGER NOT NULL, manual_out INTEGER, engine_in INTEGER NOT NULL, engine_out INTEGER NOT NULL,
    accuracy_in REAL NOT NULL, accuracy_out REAL, note TEXT, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS transactions (
    bill_no TEXT PRIMARY KEY, ts REAL NOT NULL, amount REAL NOT NULL, items INTEGER NOT NULL, staff_code TEXT,
    source TEXT, received REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_transactions_ts ON transactions(ts);
"""


def day_bounds(ts: float | None = None) -> tuple[float, float]:
    d = datetime.fromtimestamp(ts or time.time()).replace(hour=0, minute=0, second=0, microsecond=0)
    return d.timestamp(), (d + timedelta(days=1)).timestamp()


def _by_brand(visits: list[dict]) -> dict:
    """Visits attributed to the brand whose staff served the customer (None = not attended long enough)."""
    out: dict[str, int] = {}
    for v in visits:
        k = v.get("brand") or "unattended"
        out[k] = out.get(k, 0) + 1
    return out


class Store:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(visits)")}
            for col, typ in (("gender", "TEXT"), ("age", "INTEGER"), ("age_band", "TEXT")):
                if col not in cols:
                    self.conn.execute(f"ALTER TABLE visits ADD COLUMN {col} {typ}")
            if "clip" not in {r["name"] for r in self.conn.execute("PRAGMA table_info(alerts)")}:
                self.conn.execute("ALTER TABLE alerts ADD COLUMN clip TEXT")
            for col, typ in (("brand", "TEXT"), ("served_seconds", "REAL")):
                if col not in cols:
                    self.conn.execute(f"ALTER TABLE visits ADD COLUMN {col} {typ}")
            self.conn.commit()

    def _exec(self, sql: str, params: tuple = ()) -> int | None:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # ---- writes ---------------------------------------------------------
    def add_event(self, ev: dict) -> None:
        self._exec("INSERT INTO events(ts, camera_id, type, track_id, zone, message) VALUES (?,?,?,?,?,?)",
                   (ev["ts"], ev["camera_id"], ev["type"], ev.get("track_id"), ev.get("zone"), ev["message"]))

    def add_minute(self, camera_id: str, row: dict) -> None:
        self._exec("INSERT OR REPLACE INTO minute_stats(camera_id, ts, avg_occ, max_occ, entries, exits, new_visitors, staff_avg)"
                   " VALUES (?,?,?,?,?,?,?,?)",
                   (camera_id, row["ts"], row["avg_occ"], row["max_occ"], row["entries"], row["exits"],
                    row["new_visitors"], row.get("staff_avg", 0.0)))

    def add_visit(self, v: dict) -> None:
        self._exec("INSERT INTO visits(camera_id, track_id, start_ts, end_ts, dwell, engaged, is_staff, zones, crossed_in, group_id,"
                   " gender, age, age_band, brand, served_seconds) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (v["camera_id"], v["track_id"], v["start_ts"], v["end_ts"], v["dwell"], int(v["engaged"]),
                    int(v["is_staff"]), json.dumps(v["zones"]), int(v["crossed_in"]), v.get("group_id"),
                    v.get("gender"), v.get("age"), v.get("age_band"), v.get("brand"), v.get("served_seconds")))

    def add_zone_visit(self, z: dict) -> None:
        self._exec("INSERT INTO zone_visits(camera_id, zone, track_id, enter_ts, exit_ts, dwell, is_staff) VALUES (?,?,?,?,?,?,?)",
                   (z["camera_id"], z["zone"], z["track_id"], z["enter_ts"], z["exit_ts"], z["dwell"], int(z["is_staff"])))

    def add_crossing(self, c: dict) -> None:
        self._exec("INSERT INTO crossings(camera_id, ts, direction, track_id, group_id, is_staff) VALUES (?,?,?,?,?,?)",
                   (c["camera_id"], c["ts"], c["direction"], c["track_id"], c["group_id"], int(c["is_staff"])))

    def add_alert(self, a: dict) -> int:
        return self._exec("INSERT INTO alerts(key, camera_id, type, zone, track_id, message, started) VALUES (?,?,?,?,?,?,?)",
                          (a["key"], a.get("camera_id"), a["type"], a.get("zone"), a.get("track_id"), a["message"], a["started"])) or 0

    def set_alert_clip(self, alert_id: int, clip: str) -> None:
        self._exec("UPDATE alerts SET clip = ? WHERE id = ?", (clip, alert_id))

    # ---- counting accuracy audits ---------------------------------------------
    def crossing_counts(self, camera_id: str, since: float, until: float) -> dict:
        out = {"in": 0, "out": 0, "pass": 0, "staff_in": 0}
        for r in self._rows("SELECT direction, is_staff, COUNT(*) AS n FROM crossings WHERE camera_id = ? AND ts >= ? AND ts < ? GROUP BY 1, 2",
                            (camera_id, since, until)):
            if r["direction"] == "in" and r["is_staff"]:
                out["staff_in"] += r["n"]
            out[r["direction"]] = out.get(r["direction"], 0) + r["n"]
        return out

    def add_audit(self, a: dict) -> int:
        return self._exec("INSERT INTO audits(camera_id, start_ts, end_ts, manual_in, manual_out, engine_in, engine_out, accuracy_in, accuracy_out, note, created)"
                          " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                          (a["camera_id"], a["start_ts"], a["end_ts"], a["manual_in"], a.get("manual_out"), a["engine_in"], a["engine_out"],
                           a["accuracy_in"], a.get("accuracy_out"), a.get("note"), time.time())) or 0

    def audits(self, limit: int = 50) -> list[dict]:
        return self._rows("SELECT * FROM audits ORDER BY created DESC LIMIT ?", (limit,))

    # ---- POS bills ------------------------------------------------------------
    def add_transactions(self, txs: list[dict], source: str) -> None:
        now = time.time()
        days = set()
        with self.lock:
            for t in txs:
                self.conn.execute("INSERT OR REPLACE INTO transactions(bill_no, ts, amount, items, staff_code, source, received) VALUES (?,?,?,?,?,?,?)",
                                  (t["bill_no"], t["ts"], t["amount"], t["items"], t.get("staff_code"), source, now))
                days.add(datetime.fromtimestamp(t["ts"]).strftime("%Y-%m-%d"))
            for day in days:                       # the daily sales row (conversion) follows the live bills
                d0 = datetime.strptime(day, "%Y-%m-%d").timestamp()
                row = self.conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(amount), 0) AS rev FROM transactions WHERE ts >= ? AND ts < ?", (d0, d0 + 86400)).fetchone()
                self.conn.execute("INSERT OR REPLACE INTO sales(date, bills, revenue) VALUES (?,?,?)", (day, int(row["n"]), float(row["rev"])))
            self.conn.commit()

    def transactions(self, since: float, until: float, limit: int = 500) -> list[dict]:
        return self._rows("SELECT * FROM transactions WHERE ts >= ? AND ts < ? ORDER BY ts DESC LIMIT ?", (since, until, limit))

    def transactions_summary(self, now: float) -> dict:
        d0 = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        r = self._rows("SELECT COUNT(*) AS bills, COALESCE(SUM(amount), 0) AS revenue, COALESCE(SUM(items), 0) AS items, MAX(ts) AS last_ts FROM transactions WHERE ts >= ?", (d0,))[0]
        return {"bills": int(r["bills"]), "revenue": round(float(r["revenue"]), 2), "items": int(r["items"]), "last_ts": r["last_ts"]}

    def resolve_alert(self, alert_id: int, ts: float) -> None:
        self._exec("UPDATE alerts SET resolved=? WHERE id=? AND resolved IS NULL", (ts, alert_id))

    def ack_alert(self, alert_id: int) -> None:
        self._exec("UPDATE alerts SET acked=1 WHERE id=?", (alert_id,))

    def add_health(self, camera_id: str, status: str, detail: str = "") -> None:
        self._exec("INSERT INTO camera_health(camera_id, ts, status, detail) VALUES (?,?,?,?)", (camera_id, time.time(), status, detail))

    def rename_camera(self, old: str, new: str) -> None:
        """Move every row of camera `old` to id `new` (a hand-made camera that became a managed one)."""
        if old == new:
            return
        with self.lock:
            for table in ("events", "visits", "zone_visits", "crossings", "alerts", "camera_health"):
                self.conn.execute(f"UPDATE {table} SET camera_id=? WHERE camera_id=?", (new, old))
            for table in ("minute_stats", "queue_minutes"):     # camera_id is part of the primary key
                self.conn.execute(f"UPDATE OR IGNORE {table} SET camera_id=? WHERE camera_id=?", (new, old))
                self.conn.execute(f"DELETE FROM {table} WHERE camera_id=?", (old,))
            self.conn.commit()

    def rename_zone(self, camera_id: str, old: str, new: str) -> None:
        """A zone was renamed: its history (visits, queue minutes, events, alerts) follows the new name."""
        if old == new:
            return
        with self.lock:
            for table in ("events", "zone_visits", "alerts"):
                self.conn.execute(f"UPDATE {table} SET zone=? WHERE camera_id=? AND zone=?", (new, camera_id, old))
            self.conn.execute("UPDATE OR IGNORE queue_minutes SET zone=? WHERE camera_id=? AND zone=?", (new, camera_id, old))
            self.conn.execute("DELETE FROM queue_minutes WHERE camera_id=? AND zone=?", (camera_id, old))
            self.conn.commit()

    def set_sales(self, date: str, bills: int, revenue: float) -> None:
        self._exec("INSERT OR REPLACE INTO sales(date, bills, revenue) VALUES (?,?,?)", (date, bills, revenue))

    def set_daily_store(self, date: str, opened_at: float | None = None, closed_at: float | None = None) -> None:
        with self.lock:
            row = self.conn.execute("SELECT opened_at, closed_at FROM daily_store WHERE date=?", (date,)).fetchone()
            o = opened_at if opened_at is not None else (row["opened_at"] if row else None)
            c = closed_at if closed_at is not None else (row["closed_at"] if row else None)
            self.conn.execute("INSERT OR REPLACE INTO daily_store(date, opened_at, closed_at) VALUES (?,?,?)", (date, o, c))
            self.conn.commit()

    # ---- reads ----------------------------------------------------------
    def minutes_since(self, since_ts: float, camera_id: str | None = None) -> list[dict]:
        if camera_id:
            return self._rows("SELECT * FROM minute_stats WHERE ts >= ? AND camera_id = ? ORDER BY ts", (int(since_ts), camera_id))
        return self._rows("SELECT ts, SUM(avg_occ) avg_occ, SUM(max_occ) max_occ, SUM(entries) entries, SUM(exits) exits,"
                          " SUM(new_visitors) new_visitors, SUM(staff_avg) staff_avg FROM minute_stats WHERE ts >= ? GROUP BY ts ORDER BY ts",
                          (int(since_ts),))

    def events(self, since_ts: float = 0, limit: int = 200, types: list[str] | None = None, camera_id: str | None = None) -> list[dict]:
        sql = "SELECT ts, camera_id, type, track_id, zone, message FROM events WHERE ts >= ?"
        params: list = [since_ts]
        if types:
            sql += " AND type IN (%s)" % ",".join("?" * len(types))
            params += types
        if camera_id:
            sql += " AND camera_id = ?"
            params.append(camera_id)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return self._rows(sql, tuple(params))

    def alerts(self, since_ts: float = 0, open_only: bool = False, limit: int = 300) -> list[dict]:
        sql = "SELECT * FROM alerts WHERE started >= ?"
        if open_only:
            sql += " AND resolved IS NULL"
        sql += " ORDER BY started DESC LIMIT ?"
        return self._rows(sql, (since_ts, limit))

    def open_alerts(self) -> list[dict]:
        return self._rows("SELECT * FROM alerts WHERE resolved IS NULL ORDER BY started DESC")

    def visits(self, since: float, until: float, limit: int = 500) -> list[dict]:
        rows = self._rows("SELECT * FROM visits WHERE start_ts >= ? AND start_ts < ? ORDER BY start_ts DESC LIMIT ?", (since, until, limit))
        for r in rows:
            r["zones"] = json.loads(r["zones"])
        return rows

    def sales_between(self, since: float, until: float) -> dict:
        d0 = datetime.fromtimestamp(since).strftime("%Y-%m-%d")
        d1 = datetime.fromtimestamp(until - 1).strftime("%Y-%m-%d")
        rows = self._rows("SELECT COALESCE(SUM(bills),0) bills, COALESCE(SUM(revenue),0) revenue FROM sales WHERE date BETWEEN ? AND ?", (d0, d1))
        return rows[0] if rows else {"bills": 0, "revenue": 0}

    def sales_rows(self, limit: int = 60) -> list[dict]:
        return self._rows("SELECT * FROM sales ORDER BY date DESC LIMIT ?", (limit,))

    def daily_store(self, date: str) -> dict | None:
        rows = self._rows("SELECT * FROM daily_store WHERE date=?", (date,))
        return rows[0] if rows else None

    # ---- analytics ------------------------------------------------------
    def traffic(self, since: float, until: float, engage_seconds: float, camera_id: str | None = None) -> dict:
        """Footfall funnel, hourly series and the weekday x hour peak matrix for a period."""
        cam_sql = " AND camera_id = ?" if camera_id else ""
        cam_p: tuple = (camera_id,) if camera_id else ()
        visits = self._rows(f"SELECT start_ts, dwell, engaged, crossed_in, group_id, brand FROM visits"
                            f" WHERE start_ts >= ? AND start_ts < ? AND is_staff = 0{cam_sql}", (since, until) + cam_p)
        crossings = self._rows(f"SELECT ts, direction, group_id FROM crossings"
                               f" WHERE ts >= ? AND ts < ? AND is_staff = 0{cam_sql}", (since, until) + cam_p)
        ins = [c for c in crossings if c["direction"] == "in"]
        outs = [c for c in crossings if c["direction"] == "out"]
        passes = [c for c in crossings if c["direction"] == "pass"]
        returns = sum(1 for c in crossings if c["direction"] == "return_in")    # same person back in: not a walk-in
        has_line = bool(ins or outs)
        walk_ins = len(ins) if has_line else len(visits)
        groups = len({c["group_id"] for c in ins}) if has_line else len({v["group_id"] or f"v{i}" for i, v in enumerate(visits)})
        engaged = sum(1 for v in visits if v["engaged"])
        bounced = len(visits) - engaged
        avg_dwell = sum(v["dwell"] for v in visits) / len(visits) if visits else 0.0

        hourly: dict[int, dict] = defaultdict(lambda: {"walk_ins": 0, "visits": 0, "engaged": 0, "exits": 0})
        src = ins if has_line else visits
        for r in src:
            h = int((r["ts"] if has_line else r["start_ts"]) // 3600) * 3600
            hourly[h]["walk_ins"] += 1
        for v in visits:
            h = int(v["start_ts"] // 3600) * 3600
            hourly[h]["visits"] += 1
            hourly[h]["engaged"] += int(v["engaged"])
        for c in outs:
            hourly[int(c["ts"] // 3600) * 3600]["exits"] += 1

        # weekday x hour matrix: average walk-ins per hour slot over the days covered
        matrix = [[0.0] * 24 for _ in range(7)]
        days_seen: dict[tuple[int, int], set] = defaultdict(set)
        for r in src:
            ts = r["ts"] if has_line else r["start_ts"]
            dt = datetime.fromtimestamp(ts)
            matrix[dt.weekday()][dt.hour] += 1
            days_seen[(dt.weekday(), dt.hour)].add(dt.date())
        for wd in range(7):
            for h in range(24):
                n = len(days_seen.get((wd, h), ()))
                matrix[wd][h] = round(matrix[wd][h] / n, 1) if n else 0.0
        peak_hour = max(hourly.items(), key=lambda kv: kv[1]["walk_ins"])[0] if hourly else None
        sales = self.sales_between(since, until)
        conversion = round(sales["bills"] / walk_ins * 100, 1) if walk_ins and sales["bills"] else None
        return {
            "since": since, "until": until, "has_line": has_line,
            "walk_ins": walk_ins, "by_brand": _by_brand(visits), "passby": len(passes), "capture_rate_pct": round(100.0 * walk_ins / len(passes), 1) if passes else None, "groups": groups, "exits": len(outs), "returns": returns,
            "visits": len(visits), "engaged": engaged, "bounced": bounced,
            "engaged_pct": round(engaged / len(visits) * 100, 1) if visits else 0.0,
            "avg_dwell": round(avg_dwell, 1),
            "peak_hour": peak_hour,
            "hourly": [{"ts": h, **vals} for h, vals in sorted(hourly.items())],
            "matrix": matrix,
            "sales": sales, "conversion": conversion,
        }

    def zones(self, since: float, until: float, zone_roles: dict[tuple[str, str], str], zone_engage: float) -> dict:
        """Per-zone footfall, dwell, engagement, hot/cold ranking, transitions and zone-to-checkout rates."""
        zv = self._rows("SELECT camera_id, zone, track_id, enter_ts, exit_ts, dwell FROM zone_visits"
                        " WHERE enter_ts >= ? AND enter_ts < ? AND is_staff = 0 ORDER BY camera_id, track_id, enter_ts", (since, until))
        per: dict[tuple[str, str], dict] = {}
        for key, role in zone_roles.items():
            per[key] = {"camera_id": key[0], "zone": key[1], "role": role, "footfall": 0, "unique": set(), "dwell_sum": 0.0, "engaged": 0}
        for r in zv:
            key = (r["camera_id"], r["zone"])
            p = per.setdefault(key, {"camera_id": key[0], "zone": key[1], "role": zone_roles.get(key, "display"),
                                     "footfall": 0, "unique": set(), "dwell_sum": 0.0, "engaged": 0})
            p["footfall"] += 1
            p["unique"].add(r["track_id"])
            p["dwell_sum"] += r["dwell"]
            p["engaged"] += int(r["dwell"] >= zone_engage)

        # journeys: ordered zone sequence per track
        seq: dict[tuple[str, int], list[str]] = defaultdict(list)
        for r in zv:
            s = seq[(r["camera_id"], r["track_id"])]
            if not s or s[-1] != r["zone"]:
                s.append(r["zone"])
        transitions: dict[tuple[str, str, str], int] = defaultdict(int)
        to_checkout: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # zone -> [visited, reached checkout later]
        for (cam, _tid), zones in seq.items():
            for a, b in zip(zones, zones[1:]):
                transitions[(cam, a, b)] += 1
            for i, z in enumerate(zones):
                if zone_roles.get((cam, z)) == "checkout":
                    continue
                later_checkout = any(zone_roles.get((cam, zz)) == "checkout" for zz in zones[i + 1:])
                to_checkout[(cam, z)][0] += 1
                to_checkout[(cam, z)][1] += int(later_checkout)

        rows = []
        total = sum(p["footfall"] for p in per.values()) or 1
        for key, p in per.items():
            n = p["footfall"]
            tc = to_checkout.get(key)
            rows.append({
                "camera_id": p["camera_id"], "zone": p["zone"], "role": p["role"],
                "footfall": n, "unique": len(p["unique"]),
                "avg_dwell": round(p["dwell_sum"] / n, 1) if n else 0.0,
                "engaged_pct": round(p["engaged"] / n * 100, 1) if n else 0.0,
                "share_pct": round(n / total * 100, 1),
                "to_checkout_pct": round(tc[1] / tc[0] * 100, 1) if tc and tc[0] else None,
            })
        rows.sort(key=lambda r: r["footfall"], reverse=True)
        ranked = [r for r in rows if r["role"] not in ("staff",)]
        for i, r in enumerate(ranked):
            if r["footfall"] == 0:
                r["temp"] = "dead"
            elif i < max(1, len(ranked) // 3):
                r["temp"] = "hot"
            elif i >= len(ranked) - max(1, len(ranked) // 3):
                r["temp"] = "cold"
            else:
                r["temp"] = "warm"
        return {
            "zones": rows,
            "transitions": [{"camera_id": c, "from": a, "to": b, "count": n}
                            for (c, a, b), n in sorted(transitions.items(), key=lambda kv: -kv[1])[:50]],
        }

    def demographics(self, since: float, until: float, camera_id: str | None = None) -> dict:
        """Visits by age band and gender (customers only, visits where a face was seen)."""
        from .config import AGE_BAND_NAMES
        cam_sql = " AND camera_id = ?" if camera_id else ""
        rows = self._rows(f"SELECT age, age_band, gender, dwell, engaged FROM visits"
                          f" WHERE start_ts >= ? AND start_ts < ? AND is_staff = 0{cam_sql}",
                          (since, until) + ((camera_id,) if camera_id else ()))
        known = [r for r in rows if r["age_band"]]
        bands = {n: {"band": n, "count": 0, "engaged": 0, "dwell": 0.0, "male": 0, "female": 0} for n in AGE_BAND_NAMES}
        gender = {"M": 0, "F": 0}
        for r in known:
            b = bands.setdefault(r["age_band"], {"band": r["age_band"], "count": 0, "engaged": 0, "dwell": 0.0, "male": 0, "female": 0})
            b["count"] += 1
            b["engaged"] += int(r["engaged"])
            b["dwell"] += r["dwell"]
            if r["gender"] == "M":
                b["male"] += 1
                gender["M"] += 1
            else:
                b["female"] += 1
                gender["F"] += 1
        n = len(known)
        out_bands = []
        for b in bands.values():
            c = b["count"]
            out_bands.append({"band": b["band"], "count": c, "share_pct": round(c / n * 100, 1) if n else 0.0,
                              "engaged_pct": round(b["engaged"] / c * 100, 1) if c else 0.0,
                              "avg_dwell": round(b["dwell"] / c, 1) if c else 0.0, "male": b["male"], "female": b["female"]})
        ages = [r["age"] for r in known if r["age"] is not None]
        return {"total_visits": len(rows), "with_demographics": n,
                "coverage_pct": round(n / len(rows) * 100, 1) if rows else 0.0,
                "bands": out_bands,
                "gender": {"male": gender["M"], "female": gender["F"],
                           "male_pct": round(gender["M"] / n * 100, 1) if n else 0.0,
                           "female_pct": round(gender["F"] / n * 100, 1) if n else 0.0},
                "avg_age": round(sum(ages) / len(ages), 1) if ages else None}

    def add_queue_minute(self, camera_id: str, zone: str, row: dict) -> None:
        self._exec("INSERT OR REPLACE INTO queue_minutes(camera_id, zone, ts, avg_len, max_len, avg_wait, max_wait) VALUES (?,?,?,?,?,?,?)",
                   (camera_id, zone, row["ts"], row["avg_len"], row["max_len"], row["avg_wait"], row["max_wait"]))

    def queues(self, since: float, until: float, zone_roles: dict) -> dict:
        """Per queue zone: served, waits, lengths, breaches, plus a per-minute series."""
        queues = [(cam, zone) for (cam, zone), role in zone_roles.items() if role == "queue"]
        out = []
        series: dict[str, list] = {}
        for cam, zone in queues:
            zv = self._rows("SELECT dwell FROM zone_visits WHERE camera_id = ? AND zone = ? AND enter_ts >= ? AND enter_ts < ? AND is_staff = 0",
                            (cam, zone, since, until))
            waits = [r["dwell"] for r in zv]
            mins = self._rows("SELECT ts, avg_len, max_len, avg_wait, max_wait FROM queue_minutes"
                              " WHERE camera_id = ? AND zone = ? AND ts >= ? AND ts < ? ORDER BY ts", (cam, zone, int(since), int(until)))
            breaches = self._rows("SELECT COUNT(*) n, COALESCE(SUM(COALESCE(resolved, ?) - started), 0) secs FROM alerts"
                                  " WHERE type = 'queue_breach' AND camera_id = ? AND zone = ? AND started >= ? AND started < ?",
                                  (until, cam, zone, since, until))[0]
            out.append({
                "camera_id": cam, "zone": zone,
                "served": len(waits),
                "avg_wait": round(sum(waits) / len(waits), 1) if waits else 0.0,
                "max_wait": round(max(waits), 1) if waits else 0.0,
                "avg_len": round(sum(m["avg_len"] for m in mins) / len(mins), 2) if mins else 0.0,
                "max_len": max((m["max_len"] for m in mins), default=0),
                "breaches": breaches["n"], "breach_minutes": round(breaches["secs"] / 60, 1),
            })
            series[f"{cam}/{zone}"] = mins
        return {"queues": out, "series": series}

    def alert_counts(self, since: float, until: float) -> dict:
        rows = self._rows("SELECT type, COUNT(*) n, SUM(CASE WHEN resolved IS NULL THEN 1 ELSE 0 END) open_n"
                          " FROM alerts WHERE started >= ? AND started < ? GROUP BY type", (since, until))
        return {r["type"]: {"total": r["n"], "open": r["open_n"]} for r in rows}

    def health(self, since: float, until: float, camera_ids: list[str]) -> list[dict]:
        out = []
        for cid in camera_ids:
            rows = self._rows("SELECT ts, status FROM camera_health WHERE camera_id = ? AND ts < ? ORDER BY ts", (cid, until))
            if not rows:
                out.append({"camera_id": cid, "uptime_pct": 0.0, "outages": 0, "last_change": None, "status": "offline"})
                continue
            since = max(since, rows[0]["ts"])   # measure from when monitoring of this camera began
            # state before the window
            state = "offline"
            for r in rows:
                if r["ts"] <= since:
                    state = r["status"]
            online = 0.0
            t = since
            for r in rows:
                if r["ts"] < since:
                    continue
                if state == "online":
                    online += r["ts"] - t
                t = r["ts"]
                state = r["status"]
            if state == "online":
                online += until - t
            span = max(until - since, 1)
            outages = sum(1 for r in rows if r["ts"] >= since and r["status"] == "offline")
            out.append({"camera_id": cid, "uptime_pct": round(min(online / span, 1.0) * 100, 1), "outages": outages,
                        "last_change": rows[-1]["ts"] if rows else None, "status": state})
        return out

    def close(self) -> None:
        with self.lock:
            self.conn.close()
