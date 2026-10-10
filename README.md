# Retail Camera Analytics

A self-hosted, TangoEye-style retail intelligence platform: it turns existing RTSP
cameras into footfall, zone and SOP-compliance analytics, running locally on the GPU.
Multiple cameras per store, live annotated previews, a tabbed dashboard, alerts to
Telegram / webhook / e-mail, and a daily Excel report.

## Modules

**Traffic** (footfall)
- Walk-ins and exits over a counting line drawn on the entrance camera
- Group counting: people entering within a couple of seconds of each other count as one group
- Engaged vs bounced visits (by dwell time), average dwell, peak hours, weekday × hour heat grid
- Conversion = bills ÷ walk-ins (enter daily bill counts from your POS under Settings)

**Zone** (how the space is used)
- Zones drawn on the floor with a role: display / product area, entrance, checkout, queue, staff
- Per zone: footfall, unique visitors, average stay, engagement %, share of traffic, hot / warm / cold / dead ranking
- Shopper journeys (zone → zone transitions) and zone-to-checkout rate
- Live heatmaps of where people stand

**Trax** (store standards)
- Staff are told apart from customers four ways: a staff zone (behind the counter), a one-click "Staff" tag on a person in the
  camera view (which also learns their uniform colour), automatic uniform-colour matching after that, and a long-presence rule
  (anyone on the floor longer than N minutes). Staff are excluded from footfall
- Queue tab: live "N waiting · longest wait" per queue zone, breach status with an "open another till" prompt,
  queue length over time, served count, average and maximum wait, breaches and breach minutes per queue
- Queue breach alert: too many people waiting in a queue zone for too long
- Unattended customer: a customer waits in a display / checkout zone with no staff nearby
- Phone usage: a staff member holding a phone for N seconds (uses YOLO phone detection)
- Crowd, lingering visitor, camera offline, store not opened by opening time, people present after closing
- Alert centre with acknowledge / resolve, alert history, camera uptime

**Demographics** (anonymous)
- Age band (Below 12, 13-19, 20-30, 31-45, 46-59, Above 60) and gender per visit, estimated from the
  face with YuNet (OpenCV) + the InsightFace gender-age model, averaged over many frames
- Buyer analysis by age band, gender split, age × gender with engagement and dwell, in the Traffic tab and the daily report
- Needs a camera that sees faces (entrance camera at face height, faces ≥ 40 px). No face image or embedding is stored
- Model files download automatically into `models/` on first start (about 1.5 MB)
- Licence note: the InsightFace `genderage.onnx` weights are released for non-commercial research use. For a commercial
  deployment replace it with a licensed age/gender model (same 96×96 input, see `app/demographics.py`); YuNet is Apache-2.0

**Privacy**: faces are blurred by default on every camera (detected face box plus head area, coarsely pixelated in every stream and snapshot; can be switched off per camera).
Nothing identifiable is stored, only counts, timings and anonymous track IDs.

## Setup

Python 3.11+ and an NVIDIA GPU (CPU works, slower). Already done on this machine:

```bat
setup.bat
```

creates `.venv`, installs CUDA PyTorch and `requirements.txt`. `.env` holds server-level
settings (model, inference size, port). The first camera is taken from `RTSP_URL` /
`RTSP_USER` / `RTSP_PASS` in `.env` on first start; after that, cameras are managed in the
dashboard (Settings → Cameras) and stored in `data/config.json`.

## Run

```bat
run.bat
```

Open <http://localhost:8000> (or `http://<pc-ip>:8000` from another machine on the LAN). Logs go to
`logs\cam.log` as well as the console.

### Run as one stack with Congruence Eye (unattended)

`tools\stack.ps1` starts the local Congruence Eye server (node, port 3000) and this engine hidden, keeps
both alive (health check every 10 s, restart with backoff) and can register itself as a logon task:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File tools\stack.ps1 install
```

creates the task **Congruence Eye Stack** (runs at logon of the current user, so the GPU and this profile are
available; keep the PC logged on) and a desktop shortcut **Congruence Eye** that opens
<http://localhost:3000>. Other actions: `start` (foreground supervisor), `stop`, `status`, `uninstall`.
Supervisor log: `logs\stack.log`; server output: `logs\ce.out.log`.

## First-time configuration (5 minutes)

1. **Dashboard → click the camera** to open it.
2. **Draw count line** across the entrance: two clicks. The side labelled IN is inside the
   store. Use *Swap IN/OUT* if reversed. Walk-ins, groups and the funnel come from this line.
3. **Draw zone** on the floor (where feet are) for each area you care about and pick a role:
   - product areas → *display*
   - the billing counter → *checkout*
   - where customers queue → *queue* (enables queue-breach alerts)
   - behind the counter / staff-only space → *staff* (enables staff detection, unattended-customer and phone alerts)
4. **Settings → Store**: name and opening hours (drives the open / close checks).
5. **Settings → Notifications**: Telegram bot token + chat id, a webhook URL, or SMTP. Press *Send test*.
6. **Settings → Daily report**: enable and set a time; or download the Excel report any time.
7. **Settings → Sales**: enter the day's bill count to get conversion.

Thresholds (engaged time, queue size, unattended time, phone time, speeds, crowd size) are
under Settings → Thresholds and apply live.

## HTTP API

| Endpoint | Description |
|---|---|
| `GET /video/{cam}?mode=annotated\|raw&heat=0\|1&width=` | MJPEG stream per camera |
| `GET /snapshot/{cam}.jpg` | Current frame (same parameters) |
| `GET /api/stats`, `ws://host/ws` | Live store + per-camera stats (pushed twice a second) |
| `GET /api/traffic?hours=\|day=` | Funnel, hourly series, weekday × hour matrix, conversion |
| `GET /api/zones?hours=\|day=` | Zone ranking, journeys, zone-to-checkout |
| `GET /api/alerts`, `/api/alerts/summary`, `POST /api/alerts/{id}/ack\|resolve` | Alerts |
| `GET /api/visits`, `/api/events`, `/api/timeline`, `/api/health` | Raw history |
| `GET/POST/PUT/DELETE /api/cameras`, `PUT /api/cameras/{id}/layout`, `POST /api/cameras/test` | Camera management |
| `GET /api/config`, `PUT /api/config/{store\|thresholds\|notifications\|report}` | Configuration |
| `POST /api/notifications/test`, `GET /api/report/daily.xlsx?day=`, `POST /api/report/email` | Notifications, reports |
| `GET/POST /api/sales` | Daily bills for conversion |

`hours=0` or omitted means "today so far". Camera URLs and secrets are masked in responses.

## Project layout

```
app/
  config.py      .env settings + data/config.json (store, cameras, thresholds, notifications, report)
  camera.py      threaded RTSP reader (TCP, auto-reconnect, always the newest frame); while a camera is
                 offline the live counts fall to zero (open tracks close as visits) and streams/snapshots show a status card
  detector.py    YOLO11 person + phone detection, ByteTrack IDs
  analytics.py   per-camera: occupancy, dwell, staff, line counting + groups, zones + journeys, SOP checks, heatmap
  alerts.py      alert lifecycle (open / resolve / one-shot, de-duplicated, persisted, notified)
  notify.py      Telegram, webhook, SMTP
  manager.py     camera runtimes, health, store open/close, report schedule
  pipeline.py    frame loop: detect → analyse → blur → annotate → JPEG
  reports.py     daily Excel report
  store.py       SQLite (events, minute stats, visits, zone visits, crossings, alerts, health, sales)
  server.py      FastAPI: streams, WebSocket, REST, dashboard
  static/index.html   dashboard (Dashboard / Traffic / Zone / Trax / Settings)
tools/check_stream.py   RTSP connectivity test
data/            retail.db, config.json (git-ignored)
```

## Notes and limits

- One YOLO instance per camera; the RTX 5070 handles roughly 3 to 4 cameras at 12 fps with
  `yolo11m`. Use `MODEL=yolo11s.pt` in `.env` for more cameras.
- Track IDs can switch when someone is hidden for a while, which inflates visit counts a little.
  A high, front-facing entrance camera gives the cleanest footfall.
- Group detection, staff classification and SOP checks all depend on zones drawn at floor level.
- Age / gender demographics are not included (they need a face model and raise privacy questions).
- Conversion needs bill counts from your POS; there is no POS integration yet, only manual entry or `POST /api/sales`.

## Congruence Eye

**Settings → Congruence Eye** turns this PC into the store's edge box for a Congruence Eye server. Enter the
server URL and the store's edge API key (Congruence Eye → Admin → Edge devices), press **Test connection**,
then either map each camera here to one of the store's cameras (and each zone to one of that camera's zones)
and save, or, recommended, tick **Take cameras, count line and zones from Congruence Eye** ("managed"):

- every active camera of the store becomes a camera here (`ce<id>`), with the RTSP URL the server knows
  (camera linked to a recorder) or one entered here under **RTSP URL…**;
- the camera's ENTRY_LINE zone becomes the count line (IN = the server's "in direction"), SECTION / QUEUE /
  STAFF_ONLY zones become display / queue / staff zones; the id mapping is generated automatically;
- the setup is re-read every 5 minutes and on **Sync now**; drawing tools here are switched off, the
  hand-made layout is kept in `data/config.pre-managed.json`.

Snapshots are uploaded letterboxed to 16:9 so the server's zone editor shows the whole frame and its
coordinates map back exactly, whatever the camera's aspect ratio.
The connector sends, using the same `/api/ingest/*` contract as Congruence Eye's own AI service:

- walk-ins and exits (ENTRY / EXIT) and zone visits (ZONE_ENTER / ZONE_EXIT with dwell), each with the
  visitor's group id, staff flag, gender and age band
- queue length and average wait per queue zone every 30 s, a 32×18 heatmap every 5 min, a camera
  heartbeat every 30 s and a face-blurred snapshot every minute
- store opened / closed, and SOP alerts as store-ops events and AI Manager detections (phone use,
  unattended customer, queue breach, crowd, lingering, late opening)

Everything except heartbeats and snapshots goes through a write-ahead spool (`data/congruence.db`) and is
replayed in order once the server is reachable again; batches carry an idempotency id so retries never
duplicate data. The header shows the connection state, and the Settings card shows what was sent.
