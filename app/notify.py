"""Outbound notifications: Telegram, generic webhook, e-mail (SMTP). Sent from a background thread."""
from __future__ import annotations

import json
import logging
import queue
import smtplib
import threading
import time
import urllib.request
from email.message import EmailMessage

from .config import ConfigStore, NotifyConfig

log = logging.getLogger("notify")


class Notifier(threading.Thread):
    def __init__(self, config_store: ConfigStore):
        super().__init__(name="notifier", daemon=True)
        self.config_store = config_store
        self.q: queue.Queue = queue.Queue()
        self.last_error = ""
        self.sent = 0
        self.start()

    @property
    def cfg(self) -> NotifyConfig:
        return self.config_store.config.notifications

    # ---- public -----------------------------------------------------------
    def send(self, title: str, message: str, alert_type: str | None = None) -> None:
        if alert_type and alert_type not in self.cfg.types:
            return
        self.q.put({"title": title, "message": message, "type": alert_type, "ts": time.time()})

    def test(self) -> dict:
        """Synchronous test of every configured channel; returns per-channel result text."""
        item = {"title": "TangoEye-style test", "message": "Notifications are working.", "type": "test", "ts": time.time()}
        return self._deliver(item)

    def send_email(self, subject: str, body: str, to: str = "", attachment: tuple[str, bytes] | None = None) -> str:
        c = self.cfg
        to = to or c.email_to
        if not (c.smtp_host and to):
            return "email not configured"
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = c.email_from or c.smtp_user
        msg["To"] = to
        msg.set_content(body)
        if attachment:
            name, data = attachment
            msg.add_attachment(data, maintype="application",
                               subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=name)
        with smtplib.SMTP(c.smtp_host, c.smtp_port, timeout=20) as s:
            s.ehlo()
            if c.smtp_port != 25:
                try:
                    s.starttls()
                except smtplib.SMTPNotSupportedError:
                    pass
            if c.smtp_user:
                s.login(c.smtp_user, c.smtp_password)
            s.send_message(msg)
        return "sent"

    # ---- worker -----------------------------------------------------------
    def run(self) -> None:
        while True:
            item = self.q.get()
            try:
                self._deliver(item)
            except Exception as e:  # never let a notification kill the thread
                self.last_error = str(e)
                log.warning("notification failed: %s", e)

    def _deliver(self, item: dict) -> dict:
        c = self.cfg
        results: dict[str, str] = {}
        text = f"{item['title']}\n{item['message']}"
        if c.telegram_bot_token and c.telegram_chat_id:
            results["telegram"] = self._post_json(
                f"https://api.telegram.org/bot{c.telegram_bot_token}/sendMessage",
                {"chat_id": c.telegram_chat_id, "text": text})
        if c.webhook_url:
            results["webhook"] = self._post_json(c.webhook_url, item)
        if c.smtp_host and c.email_to:
            try:
                results["email"] = self.send_email(item["title"], item["message"])
            except Exception as e:
                results["email"] = f"error: {e}"
        if results:
            self.sent += 1
            errors = [v for v in results.values() if v.startswith("error")]
            self.last_error = errors[0] if errors else ""
        return results or {"none": "no channel configured"}

    @staticmethod
    def _post_json(url: str, payload: dict) -> str:
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return f"ok ({r.status})"
        except Exception as e:
            return f"error: {e}"
