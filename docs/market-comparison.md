# Congruence Eye + camera engine vs. the market leaders

Date: 2026-10-10. Scope: in-store video analytics for brick-and-mortar retail chains. Compared against the
vendors retailers of big brands actually buy: RetailNext, Sensormatic ShopperTrak (Johnson Controls),
V-Count, Tango Eye (India, Lenskart / Nykaa / Da Milano), Staqu JARVIS and Wobot. Sources at the end;
vendor claims (accuracy %, customer counts) are self-reported.

## 1. How the leaders are built

| Vendor | Sensing | Positioning | Notable |
|---|---|---|---|
| RetailNext | own stereo/AI sensors ("Aurora") + existing CCTV, cloud SaaS, mobile app | full stack: traffic, shopper journey, loss prevention, store ops, merchandising audits | pass-by + capture rate, staff exclusion by Wi-Fi/BLE tag, occupancy API, 100k+ sensors |
| Sensormatic ShopperTrak | stereo door counters, part of Sensormatic IQ | enterprise chains, malls; traffic benchmarks across tens of thousands of doors | STAR (shopper-to-associate ratio), labour scheduling from traffic, EAS/RFID integration |
| V-Count | own AI sensors (Nano/Ultima), BoostBI cloud | mid-market, 130+ countries | counting + age/gender + staff exclusion in one sensor, 200+ KPIs, AI recommendations, from $299/sensor |
| Tango Eye | edge mini-PC on existing CCTV, subscription | Indian retail chains | Tango Traffic (footfall excl. staff, engaged vs casual, groups, demographics), RevOp (bounced / missed / converted), StoreOps (open-close, SOP breaches, camera downtime), staff attendance and service time, no face recognition |
| Staqu JARVIS / Wobot | software on existing CCTV | India; Wobot leans to SOP compliance (food, fitness) | queue monitoring, unique visitor counting, compliance checklists |

The pattern: everyone sells (a) counting you can trust, (b) staff exclusion, (c) dwell/zones/queues,
(d) demographics, (e) store-ops compliance, (f) multi-store benchmarking, (g) alerts + mobile access,
(h) POS correlation. Hardware vendors win on counting accuracy; software vendors win on breadth and price.

## 2. Where Congruence Eye + the engine stand today

| Capability | Leaders | Congruence Eye + engine (2026-10-10) | Gap |
|---|---|---|---|
| Entrance counting (in/out) | 95-99 % with stereo sensors | virtual line on CCTV, segment-bounded, debounce; e2e verified server side (32/32 checks) | accuracy audit per store not yet measured; needs a manual-count calibration tool |
| Staff exclusion | tags (RetailNext), clothing AI (V-Count, Tango) | **learned uniform classifier** (new today, 88-94 % on held-out frames) + staff zones + long-presence rule + manual tag | model is per uniform; cross-chain needs the training tool (shipped) |
| Pass-by traffic / capture rate | RetailNext, V-Count, ShopperTrak | **engine**: a second ENTRY_LINE named "Pass-by" drawn in Congruence Eye becomes the street line; passers-by and capture rate on the Live tiles, `/api/traffic`, MCP `capture_rate` | KPI not yet on the Congruence Eye dashboard (needs a server build: PASS_BY event type + hourly metric) |
| Demographics (age, gender) | V-Count, Tango, RetailNext | face-based, anonymous, engine → server | accuracy unmeasured; faces rare at ceiling-camera angles (5 of 155 crops had a face) |
| Dwell, engaged vs bounced, zones, journey | all | zones (SECTION/QUEUE/STAFF_ONLY/SHELF), engaged %, bounce, journey, heatmap | shelf (SHELF) analytics is a zone only, no product-level interaction |
| Queue management | V-Count, Staqu, Tango | queue zones, wait time, unattended counter alerts | no predictive "open another counter" alert |
| Groups vs individuals | Tango, RetailNext | group detection at the line, billable entities | – |
| Store ops compliance | Tango StoreOps, Wobot | open/close, SOP breaches, mobile usage, unattended customer, checklists, tasks, tickets | **evidence snapshot on detections** (added today in the engine), hygiene/uniform rules exist server side but the engine does not emit them |
| Conversion & POS | RetailNext, ShopperTrak, V-Count | **live POS bridge in the engine**: webhook `POST /api/pos/transactions` (any common field names), CSV drop folder; bills feed conversion locally and are forwarded to the server's POS API with the edge key | adapters for specific POS products (Ginesys, GoFrugal, Zoho) are configuration on their side |
| Labour / STAR | ShopperTrak, RetailNext | customersPerStaff per hour, staffPresentAvg | no schedule import, no staffing recommendation |
| Multi-store benchmark | all | benchmark by region/format/city, infra health, uptime | no external market index |
| Alerts & mobile | all have native apps | PWA (installable); engine alerts by **phone push (ntfy)**, **WhatsApp Cloud API**, Telegram, webhook, e-mail, each with the evidence-clip link | no native store app yet (PWA wrap pending) |
| Loss prevention | RetailNext | loitering, running, lights-off, camera tamper; **evidence clip (10 s before / 8 s after) for every alert**, linked from the alert, the notification and the server detection | no sweep/concealment detection, no case export |
| Accuracy / audit tooling | vendors audit manually | **counting accuracy audit** in Settings (start, count by hand, compare: accuracy % per camera kept and exposed to the MCP), e2e replay tests, capture scan, training tool | per-store accuracy not yet shown on the Congruence Eye dashboard |
| Data privacy | anonymised counting, no video stored | faces blurred on upload, no video stored, demographics aggregate | document it for customers (GDPR/DPDP) |
| Open API / MCP | vendors: REST, some webhooks | REST + **MCP server for cross-store analysis** (new) | webhooks out of the server not yet |

## 3. What was fixed today, measured on yesterday's captures (277 frames, GoGizmo shop)

| Problem reported | Cause found | Fix | Result |
|---|---|---|---|
| staff not identified (≈30 %) | colour rule demanded ≥5 % white print in the chest crop; small logos and background in the crop failed; faces rarely visible (5/155) | trained `models/staff_cls.pt` on 150 labelled crops (`tools/staff_train.py`), engine uses it before the colour rules | recall 34 % → 88-94 %, precision 94 % (held-out frames) |
| entry points wrong | the count line was still the office door; people enter from under the camera | zone 73 redrawn along the frame bottom, direction UP | to be verified live (cameras offline from this network) |
| areas (zones) meaningless | zones were drawn for the office | shop zones drawn for both cameras: Display counter, Accessories wall, Billing queue, Counter staff side, Billing counter, Staff desk, Accessories shelf | synced into the engine |
| walk-ins 23 vs engine 123 | server counts every ENTRY it receives (verified); the engine logged crossings the server never got; the connector ignored per-event rejections | connector now counts and logs rejected events (see `rejected` in its status) | root cause visible next time it happens |
| "not market ready" | see table above | MCP server, e2e test, training tool, evidence snapshots | roadmap below |

## 4. Built on 2026-10-10 and tested end to end (`tools/e2e_features.py`)

| Feature | Where | Test |
|---|---|---|
| Phone push + WhatsApp alerts | engine notifier (ntfy, Meta Cloud API) | mock server receives the topic POST and two Cloud-API messages with bearer token |
| Live POS | engine `POST /api/pos/transactions`, CSV inbox, forward to the server | bills in three shapes accepted, CSV imported, conversion computed, server accepts the forwarded format and shows the transactions |
| Counting accuracy audit | engine `/api/audits`, Settings card | audit against a recorded hour: 100 % in / 95.8 % out as designed |
| Pass-by line + capture rate | server zone "Pass-by" -> engine `passby_line` | zone created on the server reaches the engine, count line untouched, removed again |
| Evidence clips | engine ring buffer + `data/clips`, links everywhere | recorded shop video replayed through the first camera until an alert produced an MP4 |

## 4b. Roadmap, remaining (priority order)

1. **Server-side KPIs** (needs a Congruence Eye build): PASS_BY event type + hourly passers-by and capture
   rate on the dashboard; counting-accuracy % per store in Infra health; clip links on detections as a
   first-class field.
2. **Staff model lifecycle**: labelling screen in the engine UI (click a person -> staff/customer), model
   versioning per chain; attendance (first/last seen per staff uniform per day) and service time.
3. **Alert routing per role** and web push inside the PWA (ntfy covers phones today).
4. **POS adapters** for the common Indian products (Ginesys, GoFrugal, Zoho) on top of the generic webhook.
5. **Loss prevention pack**: sweep/concealment heuristics, after-hours presence, case export of clips.
6. **Native mobile shell** (Capacitor) for store presence with push.
7. **Benchmark index**: anonymised cross-tenant traffic index once there are enough stores.

## 5. Sources

- RetailNext platform and solution comparison: https://retailnext.net/ , https://retailnext.net/platform/solution-comparison
- Sensormatic ShopperTrak / Store Guest Behaviors Analytics: https://www.sensormatic.com/resources/pr/2026/rts-2026-innovations-press-release , https://www.businesswire.com/news/home/20251209116602/en
- V-Count: https://v-count.com/ , https://v-count.com/retail-people-counting-foot-traffic-analytics-2026-guide/ , https://v-count.com/vs-retailnext/ , https://v-count.com/vs-shoppertrak
- Tango Eye: https://tangoeye.ai/ , https://app.dealroom.co/companies/tangoeye
- Staqu, Wobot: https://www.staqu.com/blog-best-video-analytics-software-retail-customer-analytics/ , https://www.cbinsights.com/compare/staqu-vs-wobot-intelligence
- Market round-ups: https://storerocket.io/learn/retail-analytics-software , https://www.guideflow.com/blog/footfall-counting-software , https://www.getdor.com/compare/best-people-counters
- KPI definitions (capture rate, bounce, STAR): https://www.pygmalios.com/insights/store-intelligence-metrics , https://www.sensormatic.com/resources/ar/2023/traffic-optimizing-labor-blog , https://retailnext.net/blog/digital-kpis-irl-e-commerce-metrics-for-brick-and-mortar
