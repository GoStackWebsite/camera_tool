"""Settings (.env) and the editable application config (data/config.json)."""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from dotenv import load_dotenv

log = logging.getLogger("config")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "app" / "static"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

ZONE_ROLES = ("display", "entrance", "checkout", "queue", "staff")
AGE_BANDS = (("Below 12", 0, 12), ("13-19", 13, 19), ("20-30", 20, 30), ("31-45", 31, 45), ("46-59", 46, 59), ("Above 60", 60, 999))
AGE_BAND_NAMES = tuple(b[0] for b in AGE_BANDS)


def age_band(age) -> str | None:
    if age is None:
        return None
    for name, lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return name
    return AGE_BANDS[-1][0]


def _env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    # python-dotenv already strips " # comments"; keep '#' inside passwords intact
    return raw.strip() or default


def build_rtsp_url(url: str, user: str, password: str) -> str:
    """Inject credentials into the RTSP URL unless it already carries them."""
    if not url:
        return url
    parts = urlsplit(url)
    if parts.username or not user:
        return url
    netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{parts.hostname}"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def mask_url(url: str) -> str:
    """Hide the password when logging or returning the URL."""
    parts = urlsplit(url)
    if not parts.password:
        return url
    netloc = f"{parts.username}:***@{parts.hostname}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


# ---------------------------------------------------------------- .env settings

@dataclass(frozen=True)
class Settings:
    default_rtsp_url: str
    model: str
    device: str
    conf: float
    imgsz: int
    target_fps: float
    host: str
    port: int
    jpeg_quality: int
    stream_width: int
    db_path: Path
    config_path: Path


def load_settings() -> Settings:
    return Settings(
        default_rtsp_url=build_rtsp_url(_env("RTSP_URL", ""), _env("RTSP_USER", ""), _env("RTSP_PASS", "")),
        model=_env("MODEL", "yolo11m.pt"),
        device=_env("DEVICE", "auto"),
        conf=float(_env("CONF", "0.35")),
        imgsz=int(_env("IMGSZ", "640")),
        target_fps=float(_env("TARGET_FPS", "12")),
        host=_env("HOST", "0.0.0.0"),
        port=int(_env("PORT", "8000")),
        jpeg_quality=int(_env("JPEG_QUALITY", "80")),
        stream_width=int(_env("STREAM_WIDTH", "1280")),
        db_path=DATA_DIR / "retail.db",
        config_path=DATA_DIR / "config.json",
    )


settings = load_settings()


# ---------------------------------------------------------------- app config (editable at runtime)

def _dc(cls, d: dict | None):
    """Build a dataclass from a dict, ignoring unknown keys and keeping defaults."""
    d = d or {}
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class ZoneConfig:
    name: str
    points: list            # normalised [[x, y], ...] in frame coordinates
    role: str = "display"   # display | entrance | checkout | queue | staff
    remote_id: int | None = None       # Congruence Eye zone id when the zone comes from the server
    remote_points: list | None = None  # the server's polygon as drawn (16:9 editor coordinates)


@dataclass
class CameraConfig:
    id: str
    name: str
    url: str
    enabled: bool = True
    target_fps: float = 0.0          # 0 = use TARGET_FPS from .env
    blur_faces: bool = True          # pixelate heads in every stream and snapshot
    line: list | None = None         # [[x, y], [x, y]] normalised; crossing to the + side = IN
    zones: list = field(default_factory=list)   # list[ZoneConfig]
    managed: bool = False            # created and kept in sync from Congruence Eye (Settings -> Congruence Eye)
    remote_id: int | None = None     # Congruence Eye camera id
    url_override: str = ""           # managed cameras: local RTSP URL that wins over the server's ("" = follow the server)
    remote_line: dict | None = None  # managed cameras: {"zone_id", "points", "in_direction"} as received from the server
    frame_w: int = 0                 # last decoded frame size (maps the server's 16:9 editor coordinates onto the frame)
    frame_h: int = 0
    passby_line: list | None = None  # [[x, y], [x, y]] on the street / mall side: everyone crossing it is a passer-by
    remote_passby: dict | None = None  # managed cameras: the server's "Pass-by" ENTRY_LINE zone {"zone_id", "points"}

    def __post_init__(self):
        self.zones = [z if isinstance(z, ZoneConfig) else _dc(ZoneConfig, z) for z in self.zones]
        for z in self.zones:
            if z.role not in ZONE_ROLES:
                z.role = "display"

    def to_dict(self, mask: bool = False) -> dict:
        d = asdict(self)
        if mask:
            d["url"] = mask_url(self.url)
            if self.url_override:
                d["url_override"] = mask_url(self.url_override)
        return d


@dataclass
class Thresholds:
    walk_speed: float = 0.06          # frame-heights / s: below = stationary
    run_speed: float = 0.5            # frame-heights / s: above = running
    loiter_seconds: float = 300.0     # lingering alert
    crowd_threshold: int = 10         # people in view
    engage_seconds: float = 30.0      # visit dwell >= this = engaged, else bounced
    zone_engage_seconds: float = 10.0 # zone stay >= this counts as engaged with the zone
    group_window: float = 2.0         # seconds between entrance crossings to form a group
    group_distance: float = 0.2       # max foot distance (fraction of frame) within a group
    queue_max: int = 4                # queue zone: people allowed before a breach
    queue_seconds: float = 60.0       # ... for this long
    unattended_seconds: float = 45.0  # customer in a zone with no staff nearby
    attend_distance: float = 0.25     # staff within this foot distance counts as attending
    phone_seconds: float = 8.0        # staff holding a phone for this long
    staff_zone_seconds: float = 20.0  # time spent in a staff zone to be classified as staff
    phone_everyone: bool = False      # phone alerts for everyone when no staff zone is defined
    line_margin: float = 0.015        # count line: feet closer than this (fraction of the frame) are "on the line"
    line_min_frames: int = 2          # count line: frames a person must stay on the new side before it counts
    line_cooldown_seconds: float = 1.5  # count line: minimum gap between two crossings of the same person


@dataclass
class StoreConfig:
    name: str = "My Store"
    open_time: str = "10:00"
    close_time: str = "21:00"
    grace_minutes: int = 15


@dataclass
class NotifyConfig:
    ntfy_url: str = "https://ntfy.sh"       # phone push: the ntfy app subscribes to the topic (self-hosted servers work too)
    ntfy_topic: str = ""
    ntfy_token: str = ""
    whatsapp_token: str = ""                # WhatsApp Business (Meta Cloud API) permanent token
    whatsapp_phone_id: str = ""             # the sending phone number id
    whatsapp_to: str = ""                   # recipients, comma separated, international format without +
    whatsapp_api_base: str = "https://graph.facebook.com/v21.0"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    webhook_url: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: str = ""
    types: list = field(default_factory=lambda: [
        "queue_breach", "unattended", "phone_usage", "crowd", "camera_offline",
        "store_open_late", "store_not_closed",
    ])


@dataclass
class ReportConfig:
    enabled: bool = False
    daily_time: str = "21:30"
    email_to: str = ""


@dataclass
class PosConfig:
    forward: bool = True          # forward bills to Congruence Eye (POST /api/pos/transactions with the edge key)
    inbox_enabled: bool = True    # watch data/pos_inbox/*.csv (POS exports)


@dataclass
class StaffConfig:
    after_minutes: float = 45.0   # anyone present this long is treated as staff (0 = off)
    uniforms: list = field(default_factory=list)   # [{"name": str, "h": 0-179, "s": 0-255, "v": 0-255}]


def uniform_kind(h: int, s: int, v: int) -> str:
    """How specific a uniform colour is: 'dark' and 'white' match a lot of ordinary clothing."""
    if v < 70:
        return "dark"
    if s < 55 and v > 160:
        return "white"
    return "distinct"


@dataclass
class DemographicsConfig:
    enabled: bool = True
    min_face: int = 24        # smallest face (px) worth estimating
    sample_every: int = 3     # run face analysis every N analysed frames
    max_samples: int = 40     # votes per person before sampling stops


@dataclass
class CongruenceConfig:
    """Congruence Eye edge connection (Settings -> Congruence Eye)."""
    enabled: bool = False
    base_url: str = "http://localhost:3000"
    api_key: str = ""                 # edge device key from Admin > Edge devices (masked in the API)
    managed: bool = False             # cameras, count line and zones come from the server (GET /api/ingest/config)
    url_overrides: dict = field(default_factory=dict)   # server camera id -> local RTSP URL (survives a camera being removed and re-added)
    cameras: dict = field(default_factory=dict)   # this app's camera id -> Congruence Eye camera id
    zones: dict = field(default_factory=dict)     # "<camera id>/<zone name>" -> Congruence Eye zone id
    heartbeat_seconds: int = 30
    flush_seconds: int = 5
    snapshot_seconds: int = 60
    heatmap_seconds: int = 300
    queue_seconds: int = 30


@dataclass
class AppConfig:
    store: StoreConfig = field(default_factory=StoreConfig)
    cameras: list = field(default_factory=list)      # list[CameraConfig]
    thresholds: Thresholds = field(default_factory=Thresholds)
    notifications: NotifyConfig = field(default_factory=NotifyConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    demographics: DemographicsConfig = field(default_factory=DemographicsConfig)
    staff: StaffConfig = field(default_factory=StaffConfig)
    congruence: CongruenceConfig = field(default_factory=CongruenceConfig)
    pos: PosConfig = field(default_factory=PosConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "AppConfig":
        return cls(
            store=_dc(StoreConfig, d.get("store")),
            cameras=[_dc(CameraConfig, c) for c in d.get("cameras", [])],
            thresholds=_dc(Thresholds, d.get("thresholds")),
            notifications=_dc(NotifyConfig, d.get("notifications")),
            report=_dc(ReportConfig, d.get("report")),
            demographics=_dc(DemographicsConfig, d.get("demographics")),
            staff=_dc(StaffConfig, d.get("staff")),
            congruence=_dc(CongruenceConfig, d.get("congruence")),
            pos=_dc(PosConfig, d.get("pos")),
        )

    def to_dict(self, mask: bool = False) -> dict:
        d = {
            "store": asdict(self.store),
            "cameras": [c.to_dict(mask) for c in self.cameras],
            "thresholds": asdict(self.thresholds),
            "notifications": asdict(self.notifications),
            "report": asdict(self.report),
            "demographics": asdict(self.demographics),
            "staff": asdict(self.staff),
            "congruence": asdict(self.congruence),
            "pos": asdict(self.pos),
        }
        if mask:
            n = d["notifications"]
            for k in ("telegram_bot_token", "smtp_password", "whatsapp_token", "ntfy_token"):
                if n.get(k):
                    n[k] = "***"
            if d["congruence"].get("api_key"):
                d["congruence"]["api_key"] = "***"
            d["congruence"]["url_overrides"] = {k: mask_url(v) for k, v in (d["congruence"].get("url_overrides") or {}).items()}
        return d

    def camera(self, cam_id: str) -> CameraConfig | None:
        return next((c for c in self.cameras if c.id == cam_id), None)


class ConfigStore:
    """Loads, migrates and saves data/config.json under a lock."""

    def __init__(self, path: Path, default_rtsp_url: str = ""):
        self.path = path
        self.lock = threading.RLock()
        self.config = self._load(default_rtsp_url)

    def _load(self, default_rtsp_url: str) -> AppConfig:
        raw: dict = {}
        for candidate in (self.path, self.path.with_suffix(".json.bak")):
            if candidate.exists():
                try:
                    raw = json.loads(candidate.read_text(encoding="utf-8"))
                    if candidate != self.path:
                        log.warning("Loaded config from backup %s", candidate)
                    break
                except Exception:
                    log.exception("Bad %s", candidate)
        if "cameras" not in raw:
            # migrate the single-camera v1 layout (line/zones/thresholds at top level)
            cam = {"id": "cam1", "name": "Camera 1", "url": default_rtsp_url,
                   "line": raw.get("line"), "zones": raw.get("zones", [])}
            thr = {k: raw[k] for k in ("walk_speed", "run_speed", "loiter_seconds", "crowd_threshold") if k in raw}
            raw = {"cameras": [cam] if default_rtsp_url else [], "thresholds": thr}
            cfg = AppConfig.from_dict(raw)
            self._write(cfg)
            log.info("Migrated config to multi-camera layout")
            return cfg
        return AppConfig.from_dict(raw)

    def _write(self, cfg: AppConfig) -> None:
        if self.path.exists():   # keep the previous version so a bad write never loses lines / zones
            try:
                self.path.with_suffix(".json.bak").write_bytes(self.path.read_bytes())
            except OSError:
                pass
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def save(self) -> None:
        with self.lock:
            self._write(self.config)
