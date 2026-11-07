/* Congruence Eye (local): remove everything about mobile-phone usage.
   Run with:  sqlcmd ... -v MODE=DRYRUN   (everything is rolled back)
              sqlcmd ... -v MODE=COMMIT
   Removes: AI Manager detections of type MOBILE_USAGE (all stores), the "Mobile Usage" alerts raised
            from them, the notifications that point at either, and the AI Manager rules of that type.
   Keeps:   the evidence snapshots (plain camera frames) and every other detection type. */
SET NOCOUNT ON;
SET XACT_ABORT ON;
SET QUOTED_IDENTIFIER ON;
SET LOCK_TIMEOUT 30000;
SET DEADLOCK_PRIORITY LOW;

DECLARE @log TABLE (step int IDENTITY(1,1), what nvarchar(120), n int);
DECLARE @det TABLE (id int PRIMARY KEY, alertId int NULL);

BEGIN TRANSACTION;

INSERT @det (id, alertId) SELECT id, alertId FROM ai_detections WHERE type = N'MOBILE_USAGE';

DELETE n FROM notifications n
WHERE EXISTS (SELECT 1 FROM @det d
              WHERE n.link = '/ai-manager/detections?id=' + CAST(d.id AS varchar(12))
                 OR (d.alertId IS NOT NULL AND n.link = '/alerts/' + CAST(d.alertId AS varchar(12))));
INSERT @log VALUES (N'notifications about mobile usage', @@ROWCOUNT);

DELETE a FROM alerts a
WHERE a.metric = N'AI_MANAGER_BREACH'
  AND a.id IN (SELECT alertId FROM @det WHERE alertId IS NOT NULL)
  AND NOT EXISTS (SELECT 1 FROM ai_detections o WHERE o.alertId = a.id AND o.type <> N'MOBILE_USAGE')
  AND NOT EXISTS (SELECT 1 FROM tickets t WHERE t.alertId = a.id);
INSERT @log VALUES (N'"Mobile Usage" alerts', @@ROWCOUNT);

DELETE FROM ai_detections WHERE id IN (SELECT id FROM @det);
INSERT @log VALUES (N'MOBILE_USAGE detections', @@ROWCOUNT);

DELETE FROM ai_manager_rules WHERE type = N'MOBILE_USAGE';
INSERT @log VALUES (N'AI Manager rules of type MOBILE_USAGE', @@ROWCOUNT);

SELECT step, what, n FROM @log ORDER BY step;
SELECT type, COUNT(*) AS detections_left FROM ai_detections GROUP BY type ORDER BY type;

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
