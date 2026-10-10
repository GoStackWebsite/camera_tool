# congruence-eye MCP server

A stdio [MCP](https://modelcontextprotocol.io) server that lets Claude analyse **all stores** of the
Congruence Eye retail-analytics platform (NestJS REST API on `:3000`) together with the **local camera
engine** in `D:\cam` (FastAPI on `:8000` + `data/retail.db`), and manage the engine's learned
staff-uniform model (`models/staff_cls.pt`).

Entry point: `D:\cam\.venv\Scripts\python.exe D:\cam\mcp_server\server.py`

Requires the `mcp` package (works with both 1.x `FastMCP` and 2.x `MCPServer`; the venv has 2.3.0) and `requests`.

All logic lives in plain functions in `server.py`; the `@mcp.tool()` wrappers are thin, so
`tools/mcp_smoke.py` can call the functions directly without an MCP client.

## Configuration

`D:\cam\data\mcp.json` (path overridable with the environment variable `MCP_CONFIG`; the file is gitignored):

```json
{
  "ce_url": "http://localhost:3000",
  "ce_email": "admin@congruence.in",
  "ce_password": "Password@123",
  "engine_url": "http://localhost:8000",
  "engine_db": "D:/cam/data/retail.db"
}
```

Optional keys: `engine_root` (default `D:\cam`, used for `models/staff_cls.pt` and `tools/staff_train.py`),
`snapshots_dir` (default `D:\New folder\congruence-eye-smarterasp\App_Data\uploads\snapshots`).

The server logs in once, caches the bearer token and re-logs-in on 401. The password is never included in
tool output (secret-looking keys are masked).

## Registration

**Claude Code** – `D:\cam\.mcp.json` is already in place (project scope); open Claude Code in `D:\cam`
and approve the server, or register it for the user scope:

```
claude mcp add congruence-eye -s user -e MCP_CONFIG=D:\cam\data\mcp.json -- D:\cam\.venv\Scripts\python.exe D:\cam\mcp_server\server.py
```

**Claude Desktop** – add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "congruence-eye": {
      "command": "D:\\cam\\.venv\\Scripts\\python.exe",
      "args": ["D:\\cam\\mcp_server\\server.py"],
      "env": { "MCP_CONFIG": "D:\\cam\\data\\mcp.json" }
    }
  }
}
```

**VS Code** (Copilot agent mode / Claude extension) – `.vscode/mcp.json`:

```json
{
  "servers": {
    "congruence-eye": {
      "type": "stdio",
      "command": "D:\\cam\\.venv\\Scripts\\python.exe",
      "args": ["D:\\cam\\mcp_server\\server.py"],
      "env": { "MCP_CONFIG": "D:\\cam\\data\\mcp.json" }
    }
  }
}
```

## Tools

Dates are `YYYY-MM-DD` in Asia/Kolkata; defaults are today / the last 7 days. Every tool returns a JSON
object; HTTP or DB failures come back as `{"error": "..."}` and anomalies (inactive store, offline
cameras, zero data) in a `notes` list.

| Tool | What it returns |
|---|---|
| `help` | Description of all tools, config and conventions |
| `list_stores(include_inactive=False)` | id, code, name, region, city, isActive, camera/zone counts, edge last seen + version, infra health |
| `store_kpis(store_id, from_date, to_date)` | Merged overview + traffic + conversion + demographics + zones + queue + storeops + AI summary |
| `compare_stores(from_date, to_date, include_inactive=False)` | One row per store (footfall, uniqueVisitors, conversionPct, engagedPct, bounceRatePct, avgDwellSec, male/female, staffExcluded, lateOpenings, sopBreaches, cameraUptimePct, aiBreaches) sorted by footfall, rankings and flags |
| `live_status(store_id=None)` | Live occupancy/footfall/queues/camera status for one or all active stores |
| `detections(store_id, from_date, to_date, type=None, limit=50)` | AI-manager detections, optionally filtered by type |
| `engine_status()` | Engine health, Congruence connector status, staff uniforms + learned model (mtime), demographics, per camera: enabled, count line, zones by role, frame size |
| `engine_day_summary(day, camera_id=None)` | From `retail.db`: visits, staff, gender, age bands, hourly crossings, events by type, zone visits, camera health transitions, open/close; plus CE footfall of the mapped store with delta (note when > 20 %) |
| `staff_recognition_report(day)` | Staff events per hour/camera, is_staff share, live tracks with `staff_reason`/votes/logo %, uniform config, model presence; notes for over- (> 40 %) / under-detection (< 3 %) |
| `feature_audit()` | Table of engine features (count lines, zones, uniforms, staff zones, learned model, demographics, thresholds, notifier channels, connector, face blur) and CE features (alert email, bounceSec, engagedStoreDwellSec, aiServiceUrl, stores/cameras/zones/edges, zone types per store, AI rules) |
| `analyse_all_stores(from_date, to_date)` | `{headline, stores[], problems[], recommendations[]}` (offline cameras, late openings, zero-footfall days, high bounce, staff never detected, missing ENTRY_LINE, camera without zones, edge not seen > 1 h) |
| `train_staff_model(labels_csv, frames_dir, epochs=40)` | Runs `tools\staff_train.py --build` then `--train`; returns the parsed metrics line and model path |
| `evaluate_staff_model()` | Runs `tools\staff_train.py --eval` |
| `list_captures(store_id, day)` | Frame counts / time span per camera in the snapshots folder for that day |

## Training the staff model

1. `list_captures(7, "2026-10-09")` to see which cameras/days have enough frames.
2. Label person boxes in a CSV with columns `file,x1,y1,x2,y2,label` (`staff` | `customer`).
3. `train_staff_model("D:/labels/day1.csv", "D:/New folder/.../snapshots/7/2026-10-09", epochs=40)`
   builds `datasets/staff`, trains and writes `models/staff_cls.pt`. The result contains
   `metrics` = `{staff_recall_pct, precision_pct, customer_specificity}`.
4. `evaluate_staff_model()` re-runs validation on the existing model.

The engine picks up `models/staff_cls.pt` automatically; without it, staff are recognised by the
uniform HSV/logo rules in `config.json` and staff zones.

### Reusing the model at another store

The classifier is a small image model of "wears our uniform or not", so it transfers to any store with the
same uniform: copy `D:\cam\models\staff_cls.pt` into `models\` of the engine installation at the other
store (the file is gitignored, so ship it alongside the install), restart the engine, then run
`staff_recognition_report(<day>)` there after a day and check the staff share (3-40 % is normal).
If the other store has a different uniform, capture frames there (`list_captures`), label them and
retrain; mixing labelled frames from several stores in one CSV gives a single model valid for all.

## Smoke test

```
D:\cam\.venv\Scripts\python.exe D:\cam\tools\mcp_smoke.py
```

Calls the plain functions against the live servers and prints compact results.
