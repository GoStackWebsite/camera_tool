"""Alert lifecycle: open / resolve / one-shot, de-duplicated by key, persisted and notified."""
from __future__ import annotations

import logging
import threading
import time

from .notify import Notifier
from .store import Store

log = logging.getLogger("alerts")

COOLDOWN = 120.0   # seconds after resolving before the same key may open again

TITLES = {
    "queue_breach": "Queue breach",
    "unattended": "Unattended customer",
    "phone_usage": "Phone usage",
    "crowd": "Crowd",
    "loiter": "Lingering visitor",
    "camera_offline": "Camera offline",
    "store_open_late": "Store not opened",
    "store_not_closed": "Store still occupied",
}


class AlertManager:
    def __init__(self, store: Store, notifier: Notifier):
        self.store = store
        self.notifier = notifier
        self.lock = threading.Lock()
        self.open_alerts: dict[str, dict] = {}
        self.cooldown: dict[str, float] = {}
        self.listener = None     # Congruence Eye hook: (type, message, camera_id, zone, started, track_id)
        for a in store.open_alerts():       # survive restarts
            self.open_alerts[a["key"]] = a

    def open(self, key: str, atype: str, message: str, camera_id: str | None = None,
             zone: str | None = None, track_id: int | None = None) -> bool:
        now = time.time()
        with self.lock:
            if key in self.open_alerts:
                return False
            if now - self.cooldown.get(key, 0) < COOLDOWN:
                return False
            a = {"key": key, "camera_id": camera_id, "type": atype, "zone": zone, "track_id": track_id,
                 "message": message, "started": now, "resolved": None, "acked": 0}
            a["id"] = self.store.add_alert(a)
            self.open_alerts[key] = a
        self._emit(a)
        self.notifier.send(TITLES.get(atype, atype), message, atype)
        return True

    def _emit(self, a: dict) -> None:
        if self.listener is None:
            return
        try:
            self.listener(a["type"], a["message"], a.get("camera_id"), a.get("zone"), a["started"], a.get("track_id"))
        except Exception:
            log.exception("alert listener failed")

    def resolve(self, key: str) -> None:
        with self.lock:
            a = self.open_alerts.pop(key, None)
            if a is None:
                return
            self.cooldown[key] = time.time()
        self.store.resolve_alert(a["id"], time.time())

    def fire(self, key: str, atype: str, message: str, camera_id: str | None = None,
             zone: str | None = None, track_id: int | None = None) -> bool:
        """One-shot alert: recorded as already resolved, de-duplicated by key within the cooldown."""
        now = time.time()
        with self.lock:
            if now - self.cooldown.get(key, 0) < COOLDOWN:
                return False
            self.cooldown[key] = now
        a = {"key": key, "camera_id": camera_id, "type": atype, "zone": zone, "track_id": track_id,
             "message": message, "started": now}
        aid = self.store.add_alert(a)
        self.store.resolve_alert(aid, now)
        self._emit(a)
        self.notifier.send(TITLES.get(atype, atype), message, atype)
        return True

    def resolve_prefix(self, prefix: str) -> None:
        for key in [k for k in list(self.open_alerts) if k.startswith(prefix)]:
            self.resolve(key)

    def ack(self, alert_id: int) -> None:
        self.store.ack_alert(alert_id)
        with self.lock:
            for a in self.open_alerts.values():
                if a["id"] == alert_id:
                    a["acked"] = 1

    def resolve_id(self, alert_id: int) -> None:
        with self.lock:
            key = next((k for k, a in self.open_alerts.items() if a["id"] == alert_id), None)
        if key:
            self.resolve(key)
        else:
            self.store.resolve_alert(alert_id, time.time())

    def list_open(self) -> list[dict]:
        with self.lock:
            return sorted(self.open_alerts.values(), key=lambda a: -a["started"])
