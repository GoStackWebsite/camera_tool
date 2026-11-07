/* Congruence Eye (local): extend the count line of "Shop floor (Camera 04)" (zone 73 "Main door", camera 31)
   to the right-hand door frame, so people walking out past the end of the glass counter count as OUT.
   Run with:  sqlcmd -S "127.0.0.1,1433" -E -C -d CongruenceEye -b -v MODE=DRYRUN -i tools\ce_extend_door_line.sql
              ... -v MODE=COMMIT   apply      ... -v MODE=UNDO   put the line back as it was
   (anything else is a dry run: rolled back)

   Why: on 14 Oct evening the line ended at the counter corner (0.7837, 0.4006). People step through the
   right part of the doorway and cross that line's direction at 1.04-1.23 of its length; the engine accepts
   up to 1.1, so walk-ins (about 1.08) counted and walk-outs (1.11-1.23) did not. Replaying 10 minutes of
   tracks: the line extended along the same direction to (0.85, 0.4313) counts all of them (IN 0.66-0.91,
   OUT 0.69-0.70 of its length). The new end stays just left of the "Counter staff side" zone.
   Same left end, same IN side (DOWN = into the shop). Only changes the row when it holds the line it expects.
   D:\cam picks it up on its next sync (every 5 min, or Settings > Congruence Eye > Sync now). */
SET NOCOUNT ON;
SET XACT_ABORT ON;

DECLARE @cur nvarchar(200) = N'[[0.6758,0.3506],[0.7837,0.4006]]';
DECLARE @new nvarchar(200) = N'[[0.6758,0.3506],[0.85,0.4313]]';
DECLARE @from nvarchar(200) = CASE WHEN '$(MODE)' = 'UNDO' THEN @new ELSE @cur END;
DECLARE @to   nvarchar(200) = CASE WHEN '$(MODE)' = 'UNDO' THEN @cur ELSE @new END;

PRINT '== Before';
SELECT id, name, type, polygon, inDirection, isActive FROM zones WHERE id = 73;

BEGIN TRANSACTION;
UPDATE zones SET polygon = @to
WHERE id = 73 AND storeId = 7 AND cameraId = 31 AND type = N'ENTRY_LINE' AND inDirection = N'DOWN' AND polygon = @from;
DECLARE @n int = @@ROWCOUNT;

PRINT '== After';
SELECT id, name, type, polygon, inDirection, isActive FROM zones WHERE id = 73;

IF @n = 0
BEGIN
    ROLLBACK TRANSACTION;
    PRINT '== Nothing changed: zone 73 does not hold the expected line (already changed, or moved by hand)';
END
ELSE IF '$(MODE)' IN ('COMMIT', 'UNDO')
BEGIN
    COMMIT TRANSACTION;
    PRINT '== ' + '$(MODE)' + ': zone 73 updated';
END
ELSE
BEGIN
    ROLLBACK TRANSACTION;
    PRINT '== DRY RUN: rolled back, nothing changed';
END
