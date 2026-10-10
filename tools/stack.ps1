<#
Congruence Eye stack on this PC: the Congruence Eye server (node, port 3000) plus the camera engine
(this folder's app, port 8000), started hidden and kept alive by a small supervisor.

  powershell -NoProfile -ExecutionPolicy Bypass -File tools\stack.ps1 start      start both hidden and supervise (blocks)
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\stack.ps1 stop       stop the supervisor and both apps
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\stack.ps1 status     health of both, task state
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\stack.ps1 install    logon task "Congruence Eye Stack" + desktop shortcut, then start
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\stack.ps1 uninstall  remove the task and the shortcut (apps keep running until "stop")

The task runs at logon of the current user (not as SYSTEM): the engine needs this user's profile, its
virtual environment and the GPU of the interactive session. Keep the store PC logged on (auto-logon + lock).
Logs: logs\stack.log (supervisor), logs\cam.log (engine), logs\ce.out.log / ce.err.log (server).
#>
param(
    [ValidateSet('start', 'stop', 'status', 'install', 'uninstall')] [string]$Action = 'status',
    [string]$CeRoot = 'D:\New folder\congruence-eye-smarterasp',
    [int]$CePort = 3000,
    [int]$CamPort = 8000
)

$ErrorActionPreference = 'Stop'
$CamRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $CamRoot 'logs'
$LogFile = Join-Path $LogDir 'stack.log'
$PidFile = Join-Path $LogDir 'stack.pid'
$TaskName = 'Congruence Eye Stack'
$ShortcutPath = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Congruence Eye.url'
$PythonW = Join-Path $CamRoot '.venv\Scripts\pythonw.exe'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log([string]$Message) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    try {
        if ((Test-Path $LogFile) -and ((Get-Item $LogFile).Length -gt 2MB)) { Move-Item -Force $LogFile ($LogFile + '.1') }
        Add-Content -Path $LogFile -Value $line -Encoding UTF8
    } catch { }
    if ($Action -ne 'start') { Write-Output $line }
}

function Test-Health([string]$Url, [bool]$NeedOk) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
        if ($r.StatusCode -ne 200) { return $false }
        if ($NeedOk) { return [bool]($r.Content -match '"status"\s*:\s*"ok"') }
        return $true
    } catch { return $false }
}

function Get-PortOwners([int]$Port) {
    @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique | Where-Object { $_ -gt 0 })
}

function Stop-Tree([int]$ProcId) {
    # Never let taskkill's stderr (e.g. "process not found") become a terminating error (PowerShell 5.1 + Stop).
    if ($ProcId -le 0 -or $ProcId -eq $PID) { return }
    if (-not (Get-Process -Id $ProcId -ErrorAction SilentlyContinue)) { return }
    try {
        Start-Process -FilePath 'taskkill.exe' -ArgumentList "/PID $ProcId /T /F" -NoNewWindow -Wait `
            -RedirectStandardOutput (Join-Path $LogDir 'taskkill.out') -RedirectStandardError (Join-Path $LogDir 'taskkill.err')
    } catch { }
}

function Start-CeServer {
    $env:CE_NO_BROWSER = '1'
    if (-not (Test-Path (Join-Path $CeRoot 'local-launcher.js'))) { throw "local-launcher.js not found in $CeRoot" }
    $p = Start-Process -FilePath 'node' -ArgumentList 'local-launcher.js' -WorkingDirectory $CeRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $LogDir 'ce.out.log') -RedirectStandardError (Join-Path $LogDir 'ce.err.log')
    Write-Log "started Congruence Eye server (node pid $($p.Id))"
    return $p
}

function Start-CamEngine {
    if (-not (Test-Path $PythonW)) { throw "pythonw.exe not found at $PythonW (run setup.bat first)" }
    $p = Start-Process -FilePath $PythonW -ArgumentList 'main.py' -WorkingDirectory $CamRoot -PassThru
    Write-Log "started camera engine (pythonw pid $($p.Id))"
    return $p
}

function Invoke-Supervise {
    $mutex = New-Object System.Threading.Mutex($false, 'Global\CongruenceEyeStack')
    if (-not $mutex.WaitOne(0)) { Write-Log 'another supervisor is already running; nothing to do'; return }
    Set-Content -Path $PidFile -Value $PID -Encoding ASCII
    Write-Log "supervisor started (pid $PID)"
    $apps = @(
        @{ name = 'Congruence Eye'; url = "http://127.0.0.1:$CePort/api/health"; needOk = $true; port = $CePort; start = { Start-CeServer } },
        @{ name = 'camera engine'; url = "http://127.0.0.1:$CamPort/api/health"; needOk = $false; port = $CamPort; start = { Start-CamEngine } }
    )
    foreach ($a in $apps) {
        $a.fails = 0; $a.backoff = 10; $a.next = 0; $a.proc = $null; $a.healthySince = 0
        $a.started = [double](Get-Date -UFormat %s)
        if (Test-Health $a.url $a.needOk) { Write-Log "$($a.name) already answers on port $($a.port)" }
        else { try { $a.proc = & $a.start } catch { Write-Log "could not start $($a.name): $($_.Exception.Message)" } }
    }
    while ($true) {
        Start-Sleep -Seconds 10
        $now = [double](Get-Date -UFormat %s)
        foreach ($a in $apps) {
            try {
                if (Test-Health $a.url $a.needOk) {
                    if ($a.fails -gt 0) { Write-Log "$($a.name) healthy again" }
                    $a.fails = 0
                    if ($a.healthySince -eq 0) { $a.healthySince = $now }
                    if ($now - $a.healthySince -gt 600) { $a.backoff = 10 }
                    continue
                }
                $a.healthySince = 0
                $a.fails++
                if ($a.fails -lt 3 -or ($now - $a.started) -lt 120 -or $now -lt $a.next) { continue }
                Write-Log "$($a.name) failed $($a.fails) health checks: restarting (next retry in $($a.backoff)s)"
                if ($a.proc) { Stop-Tree $a.proc.Id }
                foreach ($owner in (Get-PortOwners $a.port)) { Stop-Tree $owner }
                Start-Sleep -Seconds 2
                try { $a.proc = & $a.start } catch { Write-Log "could not start $($a.name): $($_.Exception.Message)" }
                $a.started = $now
                $a.next = $now + $a.backoff
                $a.backoff = [Math]::Min($a.backoff * 2, 300)
                $a.fails = 0
            } catch {
                Write-Log "supervisor error while checking $($a.name): $($_.Exception.Message)"
            }
        }
    }
}

function Stop-Stack {
    if (Test-Path $PidFile) {
        $sup = [int](Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($sup -gt 0 -and (Get-Process -Id $sup -ErrorAction SilentlyContinue)) { Write-Log "stopping supervisor (pid $sup) and its apps"; Stop-Tree $sup }
        Remove-Item -Force $PidFile -ErrorAction SilentlyContinue
    }
    foreach ($port in @($CePort, $CamPort)) {
        foreach ($owner in (Get-PortOwners $port)) {
            $name = (Get-Process -Id $owner -ErrorAction SilentlyContinue).ProcessName
            Write-Log "stopping $name (pid $owner) on port $port"
            Stop-Tree $owner
        }
    }
    Write-Log 'stack stopped'
}

function Show-Status {
    $ce = Test-Health "http://127.0.0.1:$CePort/api/health" $true
    $cam = Test-Health "http://127.0.0.1:$CamPort/api/health" $false
    Write-Output ("Congruence Eye server  (port {0}): {1}" -f $CePort, $(if ($ce) { 'healthy' } else { 'DOWN' }))
    Write-Output ("camera engine          (port {0}): {1}" -f $CamPort, $(if ($cam) { 'healthy' } else { 'DOWN' }))
    $sup = 0
    if (Test-Path $PidFile) { $sup = [int](Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1) }
    $alive = $sup -gt 0 -and (Get-Process -Id $sup -ErrorAction SilentlyContinue)
    Write-Output ("supervisor: {0}" -f $(if ($alive) { "running (pid $sup)" } else { 'not running' }))
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Write-Output ("logon task '{0}': {1}" -f $TaskName, $(if ($task) { $task.State } else { 'not installed' }))
    Write-Output ("desktop shortcut: {0}" -f $(if (Test-Path $ShortcutPath) { $ShortcutPath } else { 'none' }))
}

function Install-Task {
    $vbs = Join-Path $PSScriptRoot 'stack-hidden.vbs'
    if (-not (Test-Path $vbs)) { throw "missing $vbs" }
    $action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument ('"{0}"' -f $vbs) -WorkingDirectory $PSScriptRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
        -Description 'Congruence Eye: starts the local server and the camera engine hidden at logon and keeps them alive.' -Force | Out-Null
    Set-Content -Path $ShortcutPath -Value "[InternetShortcut]`r`nURL=http://localhost:$CePort/`r`n" -Encoding ASCII
    Write-Log "installed logon task '$TaskName' and desktop shortcut $ShortcutPath"
    Start-ScheduledTask -TaskName $TaskName
    Write-Log 'task started'
}

function Uninstall-Task {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Log "removed logon task '$TaskName'"
    }
    if (Test-Path $ShortcutPath) { Remove-Item -Force $ShortcutPath; Write-Log 'removed the desktop shortcut' }
}

switch ($Action) {
    'start' { Invoke-Supervise }
    'stop' { Stop-Stack }
    'status' { Show-Status }
    'install' { Install-Task; Start-Sleep -Seconds 3; Show-Status }
    'uninstall' { Uninstall-Task; Show-Status }
}
