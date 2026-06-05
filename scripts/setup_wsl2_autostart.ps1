# =============================================================================
# scripts/setup_wsl2_autostart.ps1 — WSL2 autostart via Windows Task Scheduler
#
# Creates a Task Scheduler task that launches WSL2 (Ubuntu-22.04) at Windows
# login and keeps it alive with a persistent background process.
# Once WSL2 is running, systemd starts — which brings up cron automatically.
#
# REQUIREMENTS:
#   - Run PowerShell as Administrator (right-click → Run as Administrator)
#   - WSL2 with Ubuntu-22.04 already installed and configured
#
# USAGE:
#   powershell.exe -ExecutionPolicy Bypass -File setup_wsl2_autostart.ps1
#
# WHAT IT CREATES:
#   Task name : QuantLab WSL2 Autostart
#   Trigger   : At user logon (fires for the current user)
#   Action    : wsl.exe --distribution Ubuntu-22.04 -- sleep infinity
#   Window    : Hidden (cmd /B suppresses the console window)
#   Priority  : Highest privilege, no timeout, one instance at a time
# =============================================================================

#Requires -RunAsAdministrator

$TaskName   = "QuantLab WSL2 Autostart"
$DistroName = "Ubuntu-22.04"

Write-Host ""
Write-Host "QuantLab WSL2 Autostart Setup"
Write-Host "==============================`n"

# ── Remove existing task if present ──────────────────────────────────────────
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Removing existing task: $TaskName"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# ── Action ────────────────────────────────────────────────────────────────────
# cmd.exe /c start "" /B wsl.exe ...
#   /B  — runs wsl.exe without creating a new console window (no flash)
#   ""  — required empty title when /B is used with a program name
# "sleep infinity" keeps the distro alive so systemd (and cron) keep running.

$WslArgs = "--distribution $DistroName -- sleep infinity"
$Action  = New-ScheduledTaskAction `
    -Execute  "cmd.exe" `
    -Argument "/c start `"`" /B wsl.exe $WslArgs"

# ── Trigger ───────────────────────────────────────────────────────────────────
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# ── Settings ──────────────────────────────────────────────────────────────────
$Settings = New-ScheduledTaskSettingsSet `
    -Hidden                                    `  # don't show in Task Scheduler UI
    -ExecutionTimeLimit ([System.TimeSpan]::Zero)  `  # no timeout — runs indefinitely
    -MultipleInstances  IgnoreNew              `  # skip if already running
    -StartWhenAvailable                           # start if missed at logon

# ── Principal (run as current user, highest privilege) ───────────────────────
$Principal = New-ScheduledTaskPrincipal `
    -UserId   $env:USERNAME `
    -LogonType Interactive  `
    -RunLevel Highest

# ── Register ──────────────────────────────────────────────────────────────────
$task = Register-ScheduledTask `
    -TaskName   $TaskName `
    -Description "Keeps WSL2 (Ubuntu-22.04) running so QuantLab cron jobs fire automatically after Windows login." `
    -Action     $Action `
    -Trigger    $Trigger `
    -Settings   $Settings `
    -Principal  $Principal `
    -Force

Write-Host "  [OK] Task created: $TaskName"
Write-Host "       Trigger    : At logon for $env:USERNAME"
Write-Host "       Action     : wsl.exe --distribution $DistroName -- sleep infinity"

# ── Start immediately (no need to log out and back in) ───────────────────────
Write-Host ""
Write-Host "  Starting WSL2 now..."
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 3   # give WSL2 a moment to initialise

$state = (Get-ScheduledTask -TaskName $TaskName).State
Write-Host "  [OK] Task state  : $state"

# ── Verify WSL2 is running ────────────────────────────────────────────────────
$wslStatus = & wsl.exe --list --running 2>&1 | Select-String $DistroName
if ($wslStatus) {
    Write-Host "  [OK] WSL2 distro : $DistroName is Running"
} else {
    Write-Host "  [!!] $DistroName not yet in running list — may need a moment to start"
}

# ── Next steps ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "══════════════════════════════════════════════════════"
Write-Host "  NEXT STEP — run this once inside WSL2:"
Write-Host ""
Write-Host "  cd ~/projects/quantlab-project"
Write-Host "  bash scripts/wsl2_keepalive.sh --install"
Write-Host ""
Write-Host "  This installs /etc/profile.d/quantlab-keepalive.sh"
Write-Host "  so cron is verified every time a terminal opens."
Write-Host "══════════════════════════════════════════════════════"
Write-Host ""
Write-Host "To verify cron is firing:"
Write-Host "  wsl.exe -d $DistroName -- bash -c 'systemctl is-active cron'"
Write-Host ""
