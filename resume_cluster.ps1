#Requires -Version 5.1
<#
.SYNOPSIS
    Resume cluster setup from bootstrap step 7.

.DESCRIPTION
    Starts/waits for existing Multipass VMs using a guest command probe instead
    of trusting "multipass list" state, then bootstraps, transfers workload code,
    verifies metrics, prints a summary, and optionally starts the dashboard.

.PARAMETER NoUI
    Skip launching the Streamlit dashboard at the end.

.EXAMPLE
    .\resume_cluster.ps1
    .\resume_cluster.ps1 -NoUI
#>

[CmdletBinding()]
param(
    [switch]$NoUI
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Fleet = @(
    [PSCustomObject]@{ Name='lb-node';        CPUs=1; Memory='512M';  DiskGB=5;  Role='load-balancer' }
    [PSCustomObject]@{ Name='compute-node-1'; CPUs=2; Memory='1024M'; DiskGB=10; Role='compute'       }
    [PSCustomObject]@{ Name='compute-node-2'; CPUs=2; Memory='1024M'; DiskGB=10; Role='compute'       }
    [PSCustomObject]@{ Name='data-node-1';    CPUs=1; Memory='2G';    DiskGB=20; Role='data'          }
    [PSCustomObject]@{ Name='data-node-2';    CPUs=1; Memory='2G';    DiskGB=20; Role='data'          }
)
$ScriptDir = $PSScriptRoot
$BootstrapMarker = '/var/local/rojg-bootstrap.done'

function Write-Log  ($msg) { Write-Host "  [>] $msg" -ForegroundColor Cyan    }
function Write-Ok   ($msg) { Write-Host "  [+] $msg" -ForegroundColor Green   }
function Write-Warn ($msg) { Write-Host "  [!] $msg" -ForegroundColor Yellow  }
function Write-Err  ($msg) { Write-Host "  [x] $msg" -ForegroundColor Red     }
function Write-Hdr  ($msg) {
    Write-Host ""
    Write-Host ("  == " + $msg + " ==") -ForegroundColor Magenta
    Write-Host ""
}
function Exit-Fatal ($msg) { Write-Err $msg; exit 1 }

function Invoke-Cmd {
    param([string]$Exe, [string[]]$Args, [int]$TimeoutSec = 300, [switch]$PassThru)
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName               = $Exe
    $psi.Arguments              = ($Args | ForEach-Object { "`"$_`"" }) -join ' '
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.UseShellExecute        = $false
    $psi.CreateNoWindow         = $true

    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    $null = $proc.Start()

    if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
        try { $proc.Kill() } catch { }
        throw "[$Exe] timed out after ${TimeoutSec}s"
    }

    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()

    if ($proc.ExitCode -ne 0) { throw "[$Exe] failed (rc=$($proc.ExitCode)): $stderr" }
    if ($PassThru) { return $stdout.Trim() }
    return $stdout.Trim()
}

function Invoke-Mp {
    param([string[]]$Args, [int]$TimeoutSec = 300, [switch]$PassThru, [switch]$NoThrow)
    try   { return Invoke-Cmd -Exe 'multipass' -Args $Args -TimeoutSec $TimeoutSec -PassThru:$PassThru }
    catch { if (-not $NoThrow) { throw } ; return '' }
}

function Test-VmExists ([string]$Name) {
    & multipass info $Name 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Test-VmReady ([string]$Name) {
    & multipass exec $Name -- true 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Test-VmFile ([string]$Name, [string]$Path) {
    & multipass exec $Name -- test -f $Path 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Get-VmIPv4 ([string]$Name) {
    $info = Invoke-Mp -Args @('info', $Name) -TimeoutSec 30 -PassThru -NoThrow
    foreach ($line in ($info -split "`n")) {
        if ($line -match '^\s*IPv4:\s*(\S+)') { return $Matches[1] }
    }
    return '-'
}

function Get-VmIPv4FromList ([string]$Name) {
    $rows = & multipass list 2>$null
    foreach ($row in $rows) {
        $parts = @($row -split '\s+' | Where-Object { $_ })
        if ($parts.Count -ge 3 -and $parts[0] -eq $Name) { return $parts[2] }
    }
    return '-'
}

function Invoke-VmMetric ([string]$Name, [string]$Snippet) {
    $output = & multipass exec $Name -- python3 -c $Snippet 2>&1
    $text = ($output | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw $text
    }
    return $text
}

function Ensure-DashboardDeps {
    $ReqFile = Join-Path $ScriptDir 'requirements.txt'
    if (-not (Test-Path $ReqFile)) {
        Write-Warn "requirements.txt not found at $ReqFile -- dashboard dependencies may be missing"
    } else {
        Write-Log 'Ensuring local Python dashboard dependencies are installed...'
        & python -m pip install --quiet --upgrade pip
        if ($LASTEXITCODE -ne 0) { Exit-Fatal 'pip upgrade failed. Check your local Python install.' }
        & python -m pip install --quiet -r $ReqFile
        if ($LASTEXITCODE -ne 0) { Exit-Fatal 'pip install -r requirements.txt failed.' }
    }

    & python -m streamlit --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Exit-Fatal 'Streamlit is not available. Run: python -m pip install streamlit'
    }
}

function Wait-ForVmReady ([string]$Name, [int]$MaxWaitSec = 240) {
    if (-not (Test-VmExists $Name)) {
        Write-Err "$Name does not exist. Run .\bootstrap_cluster.ps1 first to create it."
        return $false
    }

    if (-not (Test-VmReady $Name)) {
        Write-Warn "$Name is not accepting guest commands -- attempting start"
        Invoke-Mp -Args @('start', $Name) -TimeoutSec 180 -NoThrow | Out-Null
    }

    $waited = 0
    while ($waited -le $MaxWaitSec) {
        if (Test-VmReady $Name) {
            Write-Ok "$Name is ready"
            return $true
        }
        Write-Log "$Name guest not ready ... waiting ($waited/$MaxWaitSec s)"
        Start-Sleep -Seconds 5
        $waited += 5
    }

    Write-Err "$Name did not accept guest commands within ${MaxWaitSec}s"
    return $false
}

Clear-Host
Write-Hdr 'Resume Cluster Setup'

if (-not (Get-Command multipass -ErrorAction SilentlyContinue)) {
    Exit-Fatal 'multipass is required. Run bootstrap_cluster.ps1 first or install Multipass.'
}

# ------------------------------------------------------------------------------
# STEP 7 -- WAIT FOR GUEST READINESS
# ------------------------------------------------------------------------------
Write-Hdr 'Step 7 -- Waiting for VM guest readiness'

$ReadyVms = @{}
foreach ($vm in $Fleet) {
    $ReadyVms[$vm.Name] = Wait-ForVmReady -Name $vm.Name
}

$notReady = @($Fleet | Where-Object { -not $ReadyVms[$_.Name] })
if ($notReady.Count -gt 0) {
    Write-Warn "Continuing with ready VMs only. Not ready: $($notReady.Name -join ', ')"
}

# ------------------------------------------------------------------------------
# STEP 8 -- BOOTSTRAP VMs
# ------------------------------------------------------------------------------
Write-Hdr 'Step 8 -- Bootstrapping VMs (apt + python3 + psutil + stress-ng)'

$BootstrapScript = @'
set -e
export DEBIAN_FRONTEND=noninteractive
echo "[vm-bootstrap] apt update..."
sudo apt-get update -qq
echo "[vm-bootstrap] Installing python3 + pip + psutil + stress-ng + sysstat..."
sudo apt-get install -y -qq python3 python3-pip python3-psutil stress-ng sysstat
echo "[vm-bootstrap] Installing psutil..."
pip3 install psutil --quiet --break-system-packages 2>/dev/null || pip3 install psutil --quiet
echo "[vm-bootstrap] Verifying psutil..."
python3 -c "import psutil; print('psutil', psutil.__version__, 'OK')"
echo "[vm-bootstrap] Writing completion marker..."
sudo mkdir -p /var/local
date -Is | sudo tee /var/local/rojg-bootstrap.done >/dev/null
echo "[vm-bootstrap] Done."
'@

$jobs = @()
foreach ($vm in $Fleet) {
    if (-not $ReadyVms[$vm.Name]) { continue }
    if (Test-VmFile -Name $vm.Name -Path $BootstrapMarker) {
        Write-Ok "$($vm.Name) already bootstrapped -- skipping"
        continue
    }

    Write-Log "Bootstrapping $($vm.Name) (background)..."
    $vmName = $vm.Name
    $jobs += Start-Job -ScriptBlock {
        param($name, $script)
        $out = & multipass exec $name -- bash -c $script 2>&1
        return [PSCustomObject]@{ VM=$name; Output=$out; RC=$LASTEXITCODE }
    } -ArgumentList $vmName, $BootstrapScript
}

foreach ($job in $jobs) {
    $result = Receive-Job -Job $job -Wait -AutoRemoveJob
    if ($result.RC -eq 0) {
        Write-Ok "$($result.VM) bootstrapped"
    } else {
        Write-Err "$($result.VM) bootstrap FAILED"
        Write-Host $result.Output -ForegroundColor DarkGray
    }
}

# ------------------------------------------------------------------------------
# STEP 9 -- PUSH workload_generator.py
# ------------------------------------------------------------------------------
Write-Hdr 'Step 9 -- Distributing workload_generator.py'

$WlSrc = Join-Path $ScriptDir 'workload_generator.py'
if (-not (Test-Path $WlSrc)) {
    Write-Warn "workload_generator.py not found at $WlSrc -- skipping"
} else {
    foreach ($vm in $Fleet) {
        if (-not $ReadyVms[$vm.Name]) { continue }
        try {
            & multipass transfer $WlSrc "$($vm.Name):/tmp/workload_generator.py"
            if ($LASTEXITCODE -ne 0) { throw "multipass transfer exited $LASTEXITCODE" }
            Write-Ok "workload_generator.py -> $($vm.Name)"
        } catch {
            Write-Warn "Transfer failed for $($vm.Name): $_"
        }
    }
}

# ------------------------------------------------------------------------------
# STEP 10 -- METRIC VERIFICATION
# ------------------------------------------------------------------------------
Write-Hdr 'Step 10 -- Metric verification'

$MetricSnippet = 'import psutil,json; m=psutil.virtual_memory(); print(json.dumps({"cpu":psutil.cpu_percent(interval=1),"mem_pct":m.percent,"mem_mb":m.total//1024//1024}))'
$VerifyPass = 0
$VerifyFail = 0

foreach ($vm in $Fleet) {
    if (-not $ReadyVms[$vm.Name]) {
        Write-Warn "$($vm.Name) not ready -- skipping metric check"
        continue
    }
    try {
        $raw = Invoke-VmMetric -Name $vm.Name -Snippet $MetricSnippet
        $d = $raw | ConvertFrom-Json
        Write-Ok ("{0,-22}  CPU: {1,5:F1}%   Mem: {2,5:F1}%   RAM: {3} MB" -f `
            $vm.Name, $d.cpu, $d.mem_pct, $d.mem_mb)
        $VerifyPass++
    } catch {
        Write-Err "$($vm.Name) -- metric check failed: $_"
        $VerifyFail++
    }
}

# ------------------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------------------
Write-Hdr 'Cluster Summary'

& multipass list 2>$null
Write-Host ''

Write-Host ('  {0,-22} {1,-8} {2,-8} {3,-10} {4,-16} {5}' -f `
    'VM NAME','vCPUs','MEM','DISK','ROLE','IPv4') -ForegroundColor Gray
Write-Host ('  {0,-22} {1,-8} {2,-8} {3,-10} {4,-16} {5}' -f `
    '----------------------','------','------','--------','----------------','---------------') -ForegroundColor DarkGray

foreach ($vm in $Fleet) {
    $ip = if ($ReadyVms[$vm.Name]) { Get-VmIPv4 $vm.Name } else { '-' }
    if ($ip -eq '-') { $ip = Get-VmIPv4FromList $vm.Name }
    Write-Host ('  {0,-22} {1,-8} {2,-8} {3,-10} {4,-16} {5}' -f `
        $vm.Name, "$($vm.CPUs) vCPU", $vm.Memory, "$($vm.DiskGB) GB", $vm.Role, $ip) -ForegroundColor White
}

Write-Host ''
Write-Host "  Metrics OK: $VerifyPass / $($VerifyPass + $VerifyFail) VMs" -ForegroundColor Green
Write-Host ''
Write-Host '  Useful commands:' -ForegroundColor Gray
Write-Host '    multipass list'
Write-Host '    multipass shell <vm-name>'
Write-Host '    multipass exec <vm-name> -- <cmd>'
Write-Host '    python dispatch_workload.py --role all --mode ramp --duration 120'
Write-Host ''

# ------------------------------------------------------------------------------
# STEP 11 -- LAUNCH DASHBOARD
# ------------------------------------------------------------------------------
if ($NoUI) {
    Write-Ok 'Cluster resume complete. Dashboard skipped (-NoUI). Run: streamlit run dashboard.py'
    exit 0
}

Write-Hdr 'Step 11 -- Launching Streamlit dashboard'
Write-Host '  Dashboard URL : http://localhost:8501' -ForegroundColor Cyan
Write-Host '  Stop with     : Ctrl-C' -ForegroundColor Gray
Write-Host ''

$DashboardPath = Join-Path $ScriptDir 'dashboard.py'
if (-not (Test-Path $DashboardPath)) {
    Exit-Fatal "dashboard.py not found at $DashboardPath"
}

Ensure-DashboardDeps

& python -m streamlit run $DashboardPath `
    --server.port 8501 `
    --server.address 0.0.0.0 `
    --browser.gatherUsageStats false `
    --theme.base dark `
    --theme.backgroundColor '#040d14' `
    --theme.secondaryBackgroundColor '#0a1a26' `
    --theme.primaryColor '#00d4ff' `
    --theme.textColor '#e8f4fd'
