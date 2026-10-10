"""Live POS integration: bills arrive here (webhook or CSV drop folder), are stored locally for conversion
and forwarded to Congruence Eye (POST /api/pos/transactions with the edge key).

Accepted shapes for POST /api/pos/transactions: {"transactions": [...]}, a list, or a single bill. Field
aliases (case-insensitive): billNo | bill_no | bill | invoice | invoice_no | receipt | id; ts | time |
timestamp | date | datetime; amount | total | net | grand_total | value; itemsCount | items | qty |
quantity | units; staffCode | staff | cashier | salesman | employee. Timestamps: ISO 8601, epoch seconds,
or "YYYY-MM-DD HH:MM[:SS]" in local time.

CSV drop folder data/pos_inbox/*.csv: same aliases as column headers; processed files move to done/."""
from __future__ import annotations

import csv
import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("pos")

ALIASES = {
    "bill_no": ("billno", "bill_no", "bill", "invoice", "invoice_no", "invoiceno", "receipt", "receipt_no", "id", "txn_id", "transaction_id"),
    "ts": ("ts", "time", "timestamp", "date", "datetime", "billed_at", "created_at"),
    "amount": ("amount", "total", "net", "net_amount", "grand_total", "grandtotal", "value", "bill_amount"),
    "items": ("itemscount", "items_count", "items", "qty", "quantity", "units", "item_count"),
    "staff_code": ("staffcode", "staff_code", "staff", "cashier", "salesman", "employee", "user"),
}


def _norm_key(k: str) -> str:
    return "".join(ch for ch in str(k).strip().lower() if ch.isalnum() or ch == "_")


def parse_ts(v) -> float:
    if v is None or v == "":
        return time.time()
    if isinstance(v, (int, float)):
        return float(v) / (1000.0 if float(v) > 1e11 else 1.0)
    s = str(v).strip()
    try:
        if s.replace(".", "", 1).isdigit():
            return parse_ts(float(s))
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.astimezone()           # naive = local time
        return d.timestamp()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp {s!r}")


def normalise(raw: dict) -> dict:
    """One bill in any supported shape -> {bill_no, ts, amount, items, staff_code}."""
    low = {_norm_key(k): v for k, v in raw.items()}
    out: dict = {}
    for field, names in ALIASES.items():
        for n in names:
            if n in low and low[n] not in (None, ""):
                out[field] = low[n]
                break
    if "bill_no" not in out:
        raise ValueError("bill number missing")
    out["bill_no"] = str(out["bill_no"]).strip()[:60]
    out["ts"] = parse_ts(out.get("ts"))
    out["amount"] = max(0.0, float(str(out.get("amount", 0)).replace(",", "") or 0))
    out["items"] = max(0, int(float(str(out.get("items", 0)).replace(",", "") or 0)))
    out["staff_code"] = (str(out["staff_code"]).strip()[:30] if out.get("staff_code") not in (None, "") else None)
    return out


def payload_items(body) -> list[dict]:
    if isinstance(body, dict) and isinstance(body.get("transactions"), list):
        return body["transactions"]
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        return [body]
    raise ValueError("expected a bill, a list of bills or {\"transactions\": [...]}")


def to_ce(tx: dict, store_id: int) -> dict:
    d = {"storeId": int(store_id), "billNo": tx["bill_no"], "ts": datetime.fromtimestamp(tx["ts"], timezone.utc).astimezone().isoformat(timespec="seconds"),
         "amount": round(float(tx["amount"]), 2), "itemsCount": int(tx["items"])}
    if tx.get("staff_code"):
        d["staffCode"] = tx["staff_code"]
    return d


class PosService:
    def __init__(self, store, connector, inbox: Path):
        self.store = store
        self.connector = connector
        self.inbox = Path(inbox)
        self.received = 0
        self.forwarded = 0
        self.last_error = ""
        self.last_file = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- ingest ------------------------------------------------------------
    def ingest(self, body, source: str = "webhook", forward: bool = True) -> dict:
        items = payload_items(body)
        accepted, errors = [], []
        for i, raw in enumerate(items):
            try:
                accepted.append(normalise(raw if isinstance(raw, dict) else {}))
            except (ValueError, TypeError) as e:
                errors.append({"index": i, "error": str(e)})
        if accepted:
            self.store.add_transactions(accepted, source)
            self.received += len(accepted)
            if forward and self.connector is not None and self.connector.configured():
                sent = self.connector.pos_transactions(accepted)
                self.forwarded += sent
        return {"accepted": len(accepted), "rejected": len(errors), "errors": errors[:10], "forwarded": bool(forward and accepted)}

    # ---- CSV drop folder -----------------------------------------------------
    def start(self) -> None:
        self.inbox.mkdir(parents=True, exist_ok=True)
        (self.inbox / "done").mkdir(exist_ok=True)
        self._thread = threading.Thread(target=self._watch, name="pos-inbox", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _watch(self) -> None:
        while not self._stop.wait(30):
            try:
                self.scan_inbox()
            except Exception as e:
                self.last_error = f"inbox: {e}"
                log.warning("POS inbox: %s", e)

    def scan_inbox(self) -> int:
        n = 0
        for f in sorted(self.inbox.glob("*.csv")):
            if time.time() - f.stat().st_mtime < 5:       # still being written
                continue
            try:
                with open(f, newline="", encoding="utf-8-sig") as fh:
                    rows = list(csv.DictReader(fh))
                r = self.ingest(rows, source=f"csv:{f.name}")
                n += r["accepted"]
                self.last_file = f"{f.name}: {r['accepted']} bills, {r['rejected']} rejected"
                shutil.move(str(f), str(self.inbox / "done" / f"{int(time.time())}_{f.name}"))
            except Exception as e:
                self.last_error = f"{f.name}: {e}"
                log.warning("POS csv %s: %s", f.name, e)
                shutil.move(str(f), str(self.inbox / "done" / f"FAILED_{int(time.time())}_{f.name}"))
        return n

    def status(self) -> dict:
        return {"received": self.received, "forwarded": self.forwarded, "last_error": self.last_error, "last_file": self.last_file,
                "inbox": str(self.inbox), "today": self.store.transactions_summary(time.time())}
