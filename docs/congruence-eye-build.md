# Building Congruence Eye (server + UI) from source

Verified on this PC on 2026-10-10 (Node 24, npm 11). Source: `D:\New folder\congruence-eye-src`.
Deployed form: one bundled `dist\main.js` + the Angular build in `public\` (what SmarterASP.NET and the
local launcher run).

## 1. Backend (NestJS) -> one file

The server must be compiled by **tsc first** (it emits the decorator metadata NestJS needs for dependency
injection) and only then bundled; bundling `src/main.ts` directly with esbuild starts but every guard
fails with `Cannot read properties of undefined (reading 'getAllAndOverride')`.

```bat
cd "D:\New folder\congruence-eye-src\backend"
npm ci
npm run build
npx esbuild@0.25.4 dist/main.js --bundle --platform=node --target=node20 --format=cjs --keep-names --minify --outfile=..\..\build\dist\main.js ^
  --external:@nestjs/microservices --external:@nestjs/websockets --external:@nestjs/platform-socket.io --external:@fastify/* ^
  --external:class-transformer/storage --external:@mikro-orm/* --external:@sap/hana-client --external:hdb-pool --external:mongodb ^
  --external:mysql --external:mysql2 --external:oracledb --external:pg --external:pg-native --external:pg-query-stream --external:redis ^
  --external:ioredis --external:better-sqlite3 --external:sqlite3 --external:sql.js --external:typeorm-aurora-data-api-driver ^
  --external:react-native-sqlite-storage --external:@google-cloud/spanner --external:libsql --external:@libsql/* ^
  --external:cpu-features --external:bufferutil --external:utf-8-validate --external:canvas --external:sharp
```

`mssql` must stay **inside** the bundle (it is the SQL Server driver). The externals are optional drivers
TypeORM / Nest probe for and never load here. Result: ~6.5 MB `main.js`.

Smoke test without touching the running server (same `.env`, other port):

```bat
cd "D:\New folder\congruence-eye-smarterasp"
set PORT=3100
node ..\..\build\dist\main.js
curl http://127.0.0.1:3100/api/health
```

## 2. Frontend (Angular 21)

```bat
cd "D:\New folder\congruence-eye-src\frontend"
npm ci
npx ng build --configuration production
```

Output: `frontend\dist\frontend\browser\` (index.html + hashed chunks). Build time ~7 s after install.

## 3. Serving the UI from the Node process

The deployed bundle serves `public\` when `SERVE_FRONTEND_DIR=./public` is set (web.config does this on
the host, the local launcher too). The exported source tree in `congruence-eye-src` did **not** contain
that code; `backend/src/main.ts` now has a `serveFrontend(app)` function (static files + SPA index
fallback for non-/api, non-/uploads GET requests). It still has to be called once in `bootstrap()`
right after `configureApp(app);`:

```ts
  configureApp(app);
  serveFrontend(app);
```

(That one-line edit was left for you; everything else is in place and type-checks.)

## 4. Putting a build live

Local PC (`D:\New folder\congruence-eye-smarterasp`): stop the stack, replace `dist\main.js`, replace the
contents of `public\` with `frontend\dist\frontend\browser\`, update `BUILD.txt`, start the stack
(`tools\stack.ps1` in D:\cam). Hosted site: commit the same two things to the `deploy` branch of the
GitHub repo and let the panel pull it, as in `DEPLOY.md`.

## 5. What to add server-side for the dashboard KPIs

- `PASS_BY` event type (`EVENT_TYPES` in `ingest.dto.ts`), hourly `passBy` on `HourlyStoreMetric`,
  `captureRatePct` in `analytics.service.ts` traffic/overview, a tile in the dashboard and traffic pages.
- `CountingAudit` entity + `POST /api/ingest/audits` (edge key) + accuracy % in `infra.controller.ts`
  health and the Infra page; the engine already stores audits locally (`/api/audits`).
- `clipUrl` on `DetectionDto` / the detection entity so the AI Manager page can play the engine's clip.
