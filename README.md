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
- Phone usage: a staff member holding a phone for N seconds (uses YOLO phone detection). **Off by default**
  since 2026-10-13: switch on *Settings › Thresholds › Detect phone use* (`thresholds.phone_detection`) where
  it is wanted. Off means the detector does not look for phones at all: no phone boxes or "Using phone"
  labels, no `phone` / `phone_demo` events, no phone alerts and no MOBILE_USAGE detections to Congruence Eye
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
   **Edit zones** (or *Edit* on a row of the zone table) changes zones afterwards, on the picture: click a
   zone to select it, drag a corner to reshape it, drag inside to move it, click a dot on an edge to add a
   corner, double-click a corner to remove it, rename it or change its role in the toolbar, then
   **Save changes**. A renamed zone keeps its history.
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
| `POST /api/cameras/{id}/zones/edit` | Zone edits from the camera view: `{changes: [{op: create\|update\|delete, zone, name, role, points}]}`; managed cameras need `X-CE-Token` from `POST /api/congruence/login` |
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
- the setup is re-read every 5 minutes and on **Sync now**; the hand-made layout is kept in
  `data/config.pre-managed.json`;
- **Edit zones** and **Draw zone** in the camera view still work: the change is written to Congruence Eye
  (zone shape in its 16:9 editor coordinates, name, type; *remove* switches the zone off there, so it can be
  restored in Admin › Zones) and synced straight back, so both sides always show the same zones. Saving asks
  for a Congruence Eye **Admin** sign-in, because the edge key cannot change zones; the password goes to the
  Congruence Eye server only, and the sign-in lives in that browser tab until it is closed. Zone roles are
  limited to display / queue / staff there. The count line is still changed in Congruence Eye.

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

## Staff recognition with a learned model

Colour rules alone miss uniforms with small prints (measured on a day of captures: 34 % recall). A small image
classifier trained on labelled person crops does much better, provided the labels are varied enough (see below):

```bat
.venv\Scripts\python tools\staff_train.py --build datasets\labels.csv --frames "<folder with captured frames>"
.venv\Scripts\python tools\staff_train.py --train
```

`labels.csv` has `file,x1,y1,x2,y2,label` (pixel box of a person, label `staff` or `customer`) and optionally
`split` (`train` / `val`); `file` may include a sub-folder per day. Training writes `models\staff_cls.pt`; the
engine uses it before the colour rules (Settings › Staff detection shows "learned uniform model") and **reloads
it by itself when the file changes**, so a retrain needs no restart. Copy the file to every store of the same
chain: no training needed there.

What the training set must contain (learned on 2026-10-12, when a model trained on 150 crops of one day called
19 % of unseen customers staff, because every customer in it happened to wear light clothes):

- **customers in dark clothing**, in black long-sleeved shirts, with black bags or scarves: without them the model
  learns "dark top = staff";
- **crops where no uniform can be seen**, labelled `customer`: people seen from straight above at the door (the
  crop is mostly hair) and heavily hidden people. The model must need positive evidence of the uniform;
- more than one day and, if the PC is ever pointed at another site, people from that scene too;
- only clear cases. A plain dark top that might be the uniform is left out, not guessed;
- a validation split by time (the last part of each day), so the score is measured on people it has not seen.

- **look-alikes of the uniform**: on the evening of 2026-10-12 a visitor in a black hoodie with an ID lanyard, and
  women in black tops, were read as staff until about 60 crops of them were added as `customer`.

`datasets\staff_labels_2026-10-12.csv` (about 1500 boxes over four days) is the current label file; the frames are the
snapshots the engine uploads to Congruence Eye (`App_Data\uploads\snapshots\<store>\<day>`, faces blurred). The
classifier never sees faces: the head band of every crop is pixelated, in training and live.

Retrain without touching the live model, compare, then install (the engine hot-loads both model files):

```
.venv\Scripts\python tools\staff_train.py --build datasets\staff_labels_2026-10-12.csv --frames <snapshots>\<store>
.venv\Scripts\python tools\staff_train.py --train --out models\staff_cls.candidate.pt
.venv\Scripts\python tools\staff_compare.py models\staff_cls.pt models\staff_cls.candidate.pt
copy models\staff_cls.candidate.pt models\staff_cls.pt
```

### Staff zones: where someone stands, not who they are

A staff zone is position evidence only, so these rules keep customers from turning into staff
(`tools\test_staff_rules.py` covers them):

- the presence must be **sustained**: 20 s in the zone, and time outside drains what was collected, so brushing a
  zone edge never adds up. When the model clearly sees no uniform it takes 30 times longer (10 minutes);
- staff by zone alone **lapses**: 45 s after leaving every staff zone the person is a customer again, unless the
  uniform model has confirmed them in the meantime (then they stay staff anywhere in the shop);
- the uniform model may change its mind about a person twice; after that their clothing is treated as ambiguous
  and only a zone or a manual tag makes them staff;
- someone who is lost and found again at the same spot inside a staff zone (a seated cashier of whom the camera
  sees only the head) keeps the zone time and staff status of the lost track instead of starting as a visitor.

The clothing score itself is the **average of two separately trained models** (the staff model and the brand
model's "not a customer"): either one alone calls some dark-clothed customers staff, rarely the same ones
(`tools\ensemble_eval.py` prints the scores per capture day). Crops that show no clothing are not scored at
all: a short box cut off by the frame bottom, or one covered by another person.

Known limit: the classifiers see 128 px crops cut from 1280 px snapshots, where a plain black top and the black
uniform polo differ by a few pixels of piping. A customer in a plain black top can still be read as staff until
crops of that person are in the label set (`tools\staff_events.py 17:30` lists every staff decision since a
time of day, to find such cases quickly).

Draw staff zones on the **floor where staff stand**, never over a counter or desk top: a customer leaning on a
counter or paying at a desk has hidden feet, so their position lands on the furniture. On 2026-10-12 both staff
zones of the shop floor camera covered such spots (44 of the 163 positions inside them that evening were in
customer areas); they were redrawn to the passage behind the counter and the seat behind the billing desk.
`tools\live_tracks.py` prints, for every person on screen, why they are staff (`staff_reason`, model score, seconds
banked in a staff zone).

### How fast people are identified, and when an entry is booked

- A person who comes into view is **identifying** (grey box) until staff or customer is decided: clothing is
  sampled every 0.15 s, and five clean, very sure samples are enough. With a clear view that is about one
  second; a less clear view takes three; with no usable view at all the person counts as a visitor after 6 s.
  People who are still being identified are not in the customer count.
- People enter under the camera, where no clothing can be seen. So an IN crossing is **held** until the person
  is identified (6 s at most) and then booked as a customer walk-in or as staff, with the time of the crossing.
  Staff walking back in do not raise the walk-in count; `staff_entries` / `staff_exits` count them separately.
- When the tracker gives a person a new number (it lost them for a moment), the new track takes over the
  verdict, the votes behind it, a manual tag and gender / age from the track that was lost at that spot.
  Not at the frame border, where one person leaving and another arriving look the same.

### One store, several cameras (store fusion)

Cameras of one store that see the same floor are told so in `data\fusion.json` (example:
`docs\fusion.example.json`): `primary` is the overview camera, whose track numbers and counts are the store's,
and each *shared region* is the same piece of floor drawn once per camera (normalised frame coordinates).

A person standing in a shared region on a second camera is linked to the person standing in that region on the
overview camera (the pairing has to hold for a second; with several people the pairs are made by clothing
colour, and one person on each side is still not linked when their clothing is clearly different). While linked:

- the person has **one id**: the overview camera's number on both screens and as Congruence Eye `visitorId`;
- there is **one verdict**: the clothing scores of both views are pooled (a polo that looks like the uniform
  from behind is seen from the front by the other camera). A region can name the camera with the better view
  of clothing there (`"trust": "ce32"`, the close-up of the billing desk): that camera's verdict is then the
  store's, and the other view is never used against it. A staff zone or a manual tag on either camera counts
  on both;
- the person is **counted once** in the store totals, and has one visit, which includes the zones of the
  second camera (the queue at the billing desk becomes part of the customer's journey).

People only a second camera can see keep that camera's number with a letter (`B-7`), so a bare number always
means the overview camera's person. `GET /api/stats` lists the live links under `fusion`.

## Tests and tooling

- `tools\e2e_test.py` replays a recorded day of real visits into an isolated Congruence Eye test store and checks
  traffic, demographics, zones, queue, AI detections with evidence snapshots, live, store-ops and infra health
  through the analytics API (report in `logs\e2e_report.json`).
- `tools\test_zone_edit.py` checks zone editing offline (hand-made camera, and a managed camera against a
  throwaway Congruence Eye look-alike): no cameras, server or credentials needed.
- `mcp_server\` is an MCP server ("congruence-eye") for Claude: analyse all stores, engine status, feature audit,
  staff recognition report, train/evaluate the staff model. See `mcp_server\README.md`.
- `docs\market-comparison.md` compares the platform with RetailNext, Sensormatic ShopperTrak, V-Count and Tango Eye
  and lists the roadmap.

## Market-parity features (2026-10-10)

- **Pass-by traffic and capture rate**: draw a second ENTRY_LINE named "Pass-by" (any name starting with
  "pass") on a street- or mall-facing camera in Congruence Eye; the engine counts everyone crossing it once
  and shows walk-ins / passers-by on the Live tiles (`/api/traffic` -> `passby`, `capture_rate_pct`).
- **Counting accuracy audit** (Settings): start, count by hand for 15-30 minutes, enter your numbers; the
  engine compares them with its own crossings and keeps the accuracy per camera (`/api/audits`).
- **Phone push and WhatsApp** (Settings > Notifications): ntfy topic (install the ntfy app) and WhatsApp
  Cloud API token / phone number id / recipients, next to Telegram, webhook and e-mail. Every alert links its
  evidence clip.
- **Live POS**: point the billing software at `POST /api/pos/transactions` (a bill, a list or
  `{"transactions": [...]}` with the usual field names) or drop CSV exports into `data\pos_inbox\`. Bills drive
  conversion here and are forwarded to Congruence Eye's POS API with the edge key.
- **Evidence clips**: every alert with a camera gets an MP4 (10 s before, 8 s after) under `data\clips\`,
  linked from the alert history, the notification and the Congruence Eye detection.
- Tests: `tools\e2e_features.py` (mock push/WhatsApp server, POS shapes + CSV, audit maths, pass-by zone
  round trip, clip from a recorded video) and `tools\e2e_test.py` (server pipeline).

## Brand attribution and phone demos (2026-10-10)

- **Which brand got the walk-in** (combined GoFix + GoGizmo stores): Settings > Staff detection > "Brands by logo
  colour" (`GoFix:155, GoGizmo:36`). The engine reads the logo colour on every staff member's chest and names the
  brand; each customer accumulates attended seconds per brand (staff within `attend_distance`); at the end of the
  visit the brand with the most time (>= `serve_seconds`, default 20 s) owns the walk-in. Live tile "Walk-ins by
  brand", `/api/traffic` -> `by_brand`, `served` events, visits.brand. In Congruence Eye create a SECTION zone named
  "Served by <brand>" on each camera: the engine sends the attributed visit there, so zone analytics show walk-ins,
  dwell and gender per brand.
- **Phone demo vs personal phone** (only with *Detect phone use* switched on): a staff phone episode with a customer within `attend_distance` for at least
  `demo_share` (50 %) of its duration is a product demo: event `phone_demo`, COMPLIED MOBILE_USAGE detection in
  Congruence Eye (evidence snapshot attached), no alert. The rest stays a personal-use breach. Next refinement:
  pose keypoints (head down / phone low = scrolling, phone held out = demo).
- Test: `tools\e2e_brand.py` (recorded shop video).
