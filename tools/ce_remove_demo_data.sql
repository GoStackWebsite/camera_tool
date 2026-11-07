/* Congruence Eye (local): remove the seeded demo data and the E2E test store, keep the live store.
   Run with:  sqlcmd ... -v MODE=DRYRUN   (everything is rolled back)
              sqlcmd ... -v MODE=COMMIT
   Kept: store 7 (LIVE-01) with all its camera data, its region, the admin account, settings,
         the global alert rules (the live store's alerts are raised by them), audit log, job leases. */
SET NOCOUNT ON;
SET XACT_ABORT ON;
SET QUOTED_IDENTIFIER ON;
SET LOCK_TIMEOUT 60000;
SET DEADLOCK_PRIORITY LOW;

DECLARE @live int = 7, @admin int = 1;
IF NOT EXISTS (SELECT 1 FROM stores WHERE id = @live AND code = N'LIVE-01')
    THROW 50001, 'Store 7 is not the live store LIVE-01; nothing was changed.', 1;
IF NOT EXISTS (SELECT 1 FROM users WHERE id = @admin AND email = N'admin@congruence.in')
    THROW 50002, 'User 1 is not admin@congruence.in; nothing was changed.', 1;

DECLARE @log TABLE (step int IDENTITY(1,1), what nvarchar(120), n int);

BEGIN TRANSACTION;

/* ---- 1. the admin's notifications that point at alerts / detections of the stores being removed */
DELETE n FROM notifications n
WHERE n.userId = @admin
  AND (EXISTS (SELECT 1 FROM alerts a WHERE a.storeId <> @live AND n.link = '/alerts/' + CAST(a.id AS varchar(12)))
    OR EXISTS (SELECT 1 FROM ai_detections d WHERE d.storeId <> @live AND n.link = '/ai-manager/detections?id=' + CAST(d.id AS varchar(12))));
INSERT @log VALUES (N'notifications of the admin about demo/test stores', @@ROWCOUNT);

/* ---- 2. every row that belongs to another store, children before parents */
DECLARE @tables TABLE (ord int IDENTITY(1,1), name sysname);
INSERT @tables (name) VALUES
 (N'checklist_assignments'), (N'checklist_submissions'), (N'tasks'), (N'tickets'), (N'vendor_visits'),
 (N'ai_detections'), (N'alerts'), (N'camera_status_log'), (N'heatmap_frames'), (N'hourly_store_metrics'),
 (N'hourly_zone_metrics'), (N'ingest_batches'), (N'pos_transactions'), (N'queue_snapshots'), (N'recaps'),
 (N'snapshots'), (N'storeops_events'), (N'visitor_events'), (N'visits'), (N'zone_transitions'),
 (N'discovered_devices'), (N'probes'), (N'zones'), (N'cameras'), (N'recorders'), (N'edge_devices'),
 (N'staff'), (N'user_stores');
/* a table with a storeId column that is not listed above would be a surprise: stop */
IF EXISTS (SELECT 1 FROM sys.tables t JOIN sys.columns c ON c.object_id = t.object_id AND c.name = 'storeId'
           WHERE t.name NOT IN (SELECT name FROM @tables) AND t.name NOT IN (N'alert_rules', N'documents'))
    THROW 50003, 'A table with a storeId column is not covered by this script; nothing was changed.', 1;

DECLARE @i int = 1, @max int = (SELECT MAX(ord) FROM @tables), @t sysname, @sql nvarchar(400), @n int;
WHILE @i <= @max
BEGIN
    SELECT @t = name FROM @tables WHERE ord = @i;
    SET @sql = N'DELETE FROM ' + QUOTENAME(@t) + N' WHERE storeId <> @live; SET @n = @@ROWCOUNT;';
    EXEC sp_executesql @sql, N'@live int, @n int OUTPUT', @live = @live, @n = @n OUTPUT;
    INSERT @log VALUES (@t + N' (other stores)', @n);
    SET @i += 1;
END;

DELETE FROM alert_rules WHERE storeId IS NOT NULL AND storeId <> @live;
INSERT @log VALUES (N'alert_rules scoped to other stores', @@ROWCOUNT);

/* ---- 3. demo content that is not tied to one store */
DELETE FROM documents;                                   INSERT @log VALUES (N'documents (seeded SOP / circular / planogram)', @@ROWCOUNT);
DELETE FROM ai_manager_rules
WHERE NOT EXISTS (SELECT 1 FROM OPENJSON(storeIds) j WHERE TRY_CAST(j.value AS int) = @live);
                                                         INSERT @log VALUES (N'ai_manager_rules for demo stores', @@ROWCOUNT);
DELETE FROM checklist_schedules;                         INSERT @log VALUES (N'checklist_schedules', @@ROWCOUNT);
DELETE FROM checklist_questions;                         INSERT @log VALUES (N'checklist_questions', @@ROWCOUNT);
DELETE FROM checklist_templates;                         INSERT @log VALUES (N'checklist_templates', @@ROWCOUNT);
DELETE FROM report_schedules;                            INSERT @log VALUES (N'report_schedules', @@ROWCOUNT);
DELETE FROM vendors;                                     INSERT @log VALUES (N'vendors', @@ROWCOUNT);

/* ---- 4. the stores themselves, then regions left without a store */
DELETE FROM stores WHERE id <> @live;                    INSERT @log VALUES (N'stores (6 demo + E2E test)', @@ROWCOUNT);
DELETE r FROM regions r WHERE NOT EXISTS (SELECT 1 FROM stores s WHERE s.regionId = r.id);
                                                         INSERT @log VALUES (N'regions without a store', @@ROWCOUNT);

/* ---- 5. demo sign-in accounts that have never been used (their notifications go with them) */
DELETE u FROM users u
WHERE u.id <> @admin AND NOT EXISTS (SELECT 1 FROM audit_logs l WHERE l.userId = u.id);
                                                         INSERT @log VALUES (N'demo user accounts never signed in', @@ROWCOUNT);

/* ---- 6. test rows inside the live store: the four E2E bills and two unused "Pass-by (e2e)" zones */
DECLARE @bills TABLE ([date] date, [hour] int);
DELETE FROM pos_transactions OUTPUT deleted.[date], deleted.[hour] INTO @bills
WHERE storeId = @live AND billNo LIKE N'E2E-%';
INSERT @log VALUES (N'E2E test bills in the live store', @@ROWCOUNT);
UPDATE h SET transactions = 0, revenue = 0, itemsSold = 0
FROM hourly_store_metrics h
WHERE h.storeId = @live AND EXISTS (SELECT 1 FROM @bills b WHERE b.[date] = h.[date] AND b.[hour] = h.[hour])
  AND NOT EXISTS (SELECT 1 FROM pos_transactions p WHERE p.storeId = @live AND p.[date] = h.[date] AND p.[hour] = h.[hour]);
INSERT @log VALUES (N'hourly rows reset to 0 sales (were the E2E bills)', @@ROWCOUNT);

DELETE z FROM zones z
WHERE z.storeId = @live AND z.isActive = 0 AND z.name = N'Pass-by (e2e)'
  AND NOT EXISTS (SELECT 1 FROM visitor_events v WHERE v.zoneId = z.id)
  AND NOT EXISTS (SELECT 1 FROM hourly_zone_metrics m WHERE m.zoneId = z.id)
  AND NOT EXISTS (SELECT 1 FROM queue_snapshots q WHERE q.zoneId = z.id)
  AND NOT EXISTS (SELECT 1 FROM ai_detections d WHERE d.zoneId = z.id)
  AND NOT EXISTS (SELECT 1 FROM zone_transitions x WHERE x.fromZoneId = z.id OR x.toZoneId = z.id);
INSERT @log VALUES (N'inactive unused "Pass-by (e2e)" zones in the live store', @@ROWCOUNT);

/* ---- report */
PRINT '== removed';
SELECT step, what, n FROM @log WHERE n <> 0 ORDER BY step;
SELECT SUM(n) AS total_rows_removed_or_reset FROM @log;

PRINT '== what is left';
SELECT t.name AS tbl, SUM(p.rows) AS rows
FROM sys.tables t JOIN sys.partitions p ON p.object_id = t.object_id AND p.index_id IN (0,1)
GROUP BY t.name HAVING SUM(p.rows) > 0 ORDER BY t.name;
SELECT id, code, name, isActive FROM stores;
SELECT id, email, role FROM users;
SELECT id, name FROM regions;

IF '$(MODE)' = 'COMMIT'
BEGIN
    COMMIT TRANSACTION;
    PRINT '== COMMITTED';
END
ELSE
BEGIN
    ROLLBACK TRANSACTION;
    PRINT '== DRY RUN: rolled back, nothing changed';
END
