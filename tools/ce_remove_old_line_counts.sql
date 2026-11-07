/* Congruence Eye (local): remove today's IN/OUT counted on the old, wrong count line.
   Run with:  sqlcmd -S "127.0.0.1,1433" -E -C -d CongruenceEye -b -v MODE=DRYRUN -i tools\ce_remove_old_line_counts.sql
              ... -v MODE=COMMIT   (anything else is a dry run: everything is rolled back)

   Until 18:20:08 on 2026-10-14 (12:50:08 UTC, when D:\cam picked up the new line) the count line of
   "Shop floor (Camera 04)" (camera 31, store 7) ran along the bottom of the picture instead of across the
   glass door, so the walk-ins and exits it produced that day are not real door crossings.

   Removes, for store 7 / 2026-10-14 / before the switch:
     * ENTRY and EXIT rows of camera 31 in visitor_events (customers and staff),
     * the visits built from those ENTRY rows (one per ENTRY: unique visitors, dwell, gender / age, groups).
   Then corrects the hourly totals the dashboard reads (hourly_store_metrics):
     * footfallIn / footfallOut / staffEntries of every affected hour are recounted from the rows left,
     * an hour with no crossings and no visits left gets its visitor columns set to 0, as Congruence Eye's
       own recompute would do (its 5-minute job only revisits the current and the previous hour).
     An hour that still has new-line visits is finished by that 5-minute job (today until 20:00).
   Keeps: zone enter/exit rows, zone and queue figures, crossings after the switch, every other day.
   Safe to run twice: the second run finds nothing to remove.
   Backup taken before: MSSQL\Backup\CongruenceEye_before_old_line_cleanup_2026-10-14.bak */
SET NOCOUNT ON;
SET XACT_ABORT ON;
SET QUOTED_IDENTIFIER ON;
SET LOCK_TIMEOUT 30000;
SET DEADLOCK_PRIORITY LOW;

DECLARE @store int = 7, @cam int = 31, @date varchar(10) = '2026-10-14';
DECLARE @cutoff datetime2 = '2026-10-14T12:50:08';            -- UTC (18:20:08 India time)
DECLARE @log TABLE (step int IDENTITY(1,1), what nvarchar(120), n int);
DECLARE @hours TABLE (hour int PRIMARY KEY);
DECLARE @visitors TABLE (visitorId nvarchar(100) PRIMARY KEY);

PRINT '== Before: IN / OUT rows of camera 31 today';
SELECT CASE WHEN ts < @cutoff THEN 'old line (removed)' ELSE 'door line (kept)' END AS line, type,
       SUM(CASE WHEN isStaff = 1 THEN 0 ELSE 1 END) AS customers, SUM(CASE WHEN isStaff = 1 THEN 1 ELSE 0 END) AS staff
FROM visitor_events
WHERE storeId = @store AND cameraId = @cam AND date = @date AND type IN (N'ENTRY', N'EXIT')
GROUP BY CASE WHEN ts < @cutoff THEN 'old line (removed)' ELSE 'door line (kept)' END, type
ORDER BY 1 DESC, 2;
SELECT hour, footfallIn, footfallOut, staffEntries, uniqueVisitors, individuals, [groups], engaged, bounced, staffPresentAvg
FROM hourly_store_metrics WHERE storeId = @store AND date = @date ORDER BY hour;

BEGIN TRANSACTION;

INSERT @visitors (visitorId)
SELECT DISTINCT visitorId FROM visitor_events
WHERE storeId = @store AND cameraId = @cam AND date = @date AND type = N'ENTRY' AND ts < @cutoff AND visitorId IS NOT NULL;

INSERT @hours (hour)
SELECT hour FROM visitor_events
WHERE storeId = @store AND cameraId = @cam AND date = @date AND type IN (N'ENTRY', N'EXIT') AND ts < @cutoff
UNION
SELECT hour FROM visits
WHERE storeId = @store AND date = @date AND entryTs < @cutoff AND visitorId IN (SELECT visitorId FROM @visitors);

DELETE FROM visits
WHERE storeId = @store AND date = @date AND entryTs < @cutoff AND visitorId IN (SELECT visitorId FROM @visitors);
INSERT @log VALUES (N'visits built from old-line walk-ins', @@ROWCOUNT);

DELETE FROM visitor_events
WHERE storeId = @store AND cameraId = @cam AND date = @date AND type IN (N'ENTRY', N'EXIT') AND ts < @cutoff;
INSERT @log VALUES (N'old-line ENTRY / EXIT rows', @@ROWCOUNT);

-- footfall of each affected hour, recounted from the rows that are left (Congruence Eye's own rule)
UPDATE m SET
    footfallIn   = (SELECT COUNT(*) FROM visitor_events e WHERE e.storeId = m.storeId AND e.date = m.date AND e.hour = m.hour AND e.type = N'ENTRY' AND e.isStaff = 0),
    footfallOut  = (SELECT COUNT(*) FROM visitor_events e WHERE e.storeId = m.storeId AND e.date = m.date AND e.hour = m.hour AND e.type = N'EXIT' AND e.isStaff = 0),
    staffEntries = (SELECT COUNT(*) FROM visitor_events e WHERE e.storeId = m.storeId AND e.date = m.date AND e.hour = m.hour AND e.type = N'ENTRY' AND e.isStaff = 1),
    updatedAt    = SYSUTCDATETIME()
FROM hourly_store_metrics m
WHERE m.storeId = @store AND m.date = @date AND m.hour IN (SELECT hour FROM @hours);
INSERT @log VALUES (N'hourly rows: footfall recounted', @@ROWCOUNT);

-- an hour with no visits left has no visitors, groups, dwell, demographics or staff presence from visits
UPDATE m SET
    uniqueVisitors = 0, individuals = 0, [groups] = 0, billableEntities = 0, engaged = 0, engagedEntities = 0,
    bounced = 0, repeatVisitors = 0, male = 0, female = 0, genderUnknown = 0,
    age_0_17 = 0, age_18_24 = 0, age_25_34 = 0, age_35_44 = 0, age_45_54 = 0, age_55_plus = 0, ageUnknown = 0,
    male_0_17 = 0, male_18_24 = 0, male_25_34 = 0, male_35_44 = 0, male_45_54 = 0, male_55_plus = 0,
    female_0_17 = 0, female_18_24 = 0, female_25_34 = 0, female_35_44 = 0, female_45_54 = 0, female_55_plus = 0,
    totalDwellSec = 0, dwellCount = 0, staffPresentAvg = 0, updatedAt = SYSUTCDATETIME()
FROM hourly_store_metrics m
WHERE m.storeId = @store AND m.date = @date AND m.hour IN (SELECT hour FROM @hours)
  AND NOT EXISTS (SELECT 1 FROM visits v WHERE v.storeId = m.storeId AND v.date = m.date AND v.hour = m.hour)
  AND NOT EXISTS (SELECT 1 FROM visits v WHERE v.storeId = m.storeId AND v.date = m.date AND v.isStaff = 1
                  AND v.entryTs < DATEADD(hour, m.hour + 1, CAST(CAST(m.date AS date) AS datetime2)) AT TIME ZONE 'India Standard Time' AT TIME ZONE 'UTC');
INSERT @log VALUES (N'hourly rows with no visits left: visitor figures set to 0', @@ROWCOUNT);

SELECT step, what, n FROM @log ORDER BY step;
PRINT '== After';
SELECT type, SUM(CASE WHEN isStaff = 1 THEN 0 ELSE 1 END) AS customers, SUM(CASE WHEN isStaff = 1 THEN 1 ELSE 0 END) AS staff
FROM visitor_events WHERE storeId = @store AND date = @date AND type IN (N'ENTRY', N'EXIT') GROUP BY type ORDER BY type;
SELECT hour, footfallIn, footfallOut, staffEntries, uniqueVisitors, individuals, [groups], engaged, bounced, staffPresentAvg
FROM hourly_store_metrics WHERE storeId = @store AND date = @date ORDER BY hour;

DECLARE @localHour int = DATEPART(hour, SYSUTCDATETIME() AT TIME ZONE 'UTC' AT TIME ZONE 'India Standard Time');
IF EXISTS (SELECT 1 FROM @hours h WHERE h.hour < @localHour - 1
           AND EXISTS (SELECT 1 FROM visits v WHERE v.storeId = @store AND v.date = @date AND v.hour = h.hour))
    PRINT '== Note: an hour that still has visits is older than the 5-minute job looks at; its visitor columns stay as they are';

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
