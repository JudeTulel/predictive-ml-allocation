#Requires -Version 5.1
<#
.SYNOPSIS
    DataCenter AI Controller — Full cluster bootstrap for a fresh Windows machine.

.DESCRIPTION
    1. Checks execution policy and elevates if needed
    2. Installs Winget (if missing) and uses it to install prerequisites
    3. Installs Python 3 (via winget) if missing
    4. Installs Multipass (via winget) if missing
    5. Installs Python dashboard dependencies (pip)
    6. Launches the 5-VM fleet (lb-node, compute-node-1/2, data-node-1/2)
    7. Waits for every VM to reach Running state
    8. Bootstraps each VM (apt update + python3 + psutil + stress-ng)
    9. Pushes workload_generator.py into every VM
   10. Verifies metrics can be pulled from every VM
   11. Prints a summary and (optionally) starts the Streamlit dashboard

.PARAMETER NoUI
    Skip launching the Streamlit dashboard at the end.

.PARAMETER Destroy
    Delete and purge every cluster VM — useful for a clean slate.

.EXAMPLE
    .\bootstrap_cluster.ps1
    .\bootstrap_cluster.ps1 -NoUI
    .\bootstrap_cluster.ps1 -Destroy

.NOTES
    Run from the project folder that also contains dashboard.py,
    workload_generator.py, and requirements.txt.
    Requires an internet connection on first run.
#>

[CmdletBinding()]
param(
    [switch]$NoUI,
    [switch]$Destroy
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# -- Fleet definition ----------------------------------------------------------
$Fleet = @(
    [PSCustomObject]@{ Name='lb-node';        CPUs=1; Memory='512M';  DiskGB=5;  Role='load-balancer' }
    [PSCustomObject]@{ Name='compute-node-1'; CPUs=2; Memory='1024M'; DiskGB=10; Role='compute'       }
    [PSCustomObject]@{ Name='compute-node-2'; CPUs=2; Memory='1024M'; DiskGB=10; Role='compute'       }
    [PSCustomObject]@{ Name='data-node-1';    CPUs=1; Memory='2G';    DiskGB=20; Role='data'          }
    [PSCustomObject]@{ Name='data-node-2';    CPUs=1; Memory='2G';    DiskGB=20; Role='data'          }
)
$UbuntuImage = '22.04'
$ScriptDir   = $PSScriptRoot
$BootstrapMarker = '/var/local/rojg-bootstrap.done'

# -- Colour helpers ------------------------------------------------------------
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

# -- Banner --------------------------------------------------------------------
Clear-Host
Write-Host @"

  ██████╗  ██████╗    ██████╗██╗     ██╗   ██╗███████╗████████╗███████╗██████╗
  ██╔══██╗██╔════╝   ██╔════╝██║     ██║   ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗
  ██║  ██║██║        ██║     ██║     ██║   ██║███████╗   ██║   █████╗  ██████╔╝
  ██║  ██║██║        ██║     ██║     ██║   ██║╚════██║   ██║   ██╔══╝  ██╔══██╗
  ██████╔╝╚██████╗   ╚██████╗███████╗╚██████╔╝███████║   ██║   ███████╗██║  ██║
  ╚═════╝  ╚═════╝    ╚═════╝╚══════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

"@ -ForegroundColor Cyan
Write-Host "  DataCenter AI Controller -- Cluster Bootstrap (PowerShell)" -ForegroundColor White
Write-Host "  Predictive Resource Allocation * Two-Stage ML Pipeline * Energy Optimization"
Write-Host ""

# ------------------------------------------------------------------------------
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

    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit($TimeoutSec * 1000) | Out-Null

    if ($PassThru) { return $stdout.Trim() }
    if ($proc.ExitCode -ne 0) { throw "[$Exe] failed (rc=$($proc.ExitCode)): $stderr" }
    return $stdout.Trim()
}

function Invoke-Mp {
    param([string[]]$Args, [int]$TimeoutSec = 300, [switch]$PassThru, [switch]$NoThrow)
    try   { return Invoke-Cmd -Exe 'multipass' -Args $Args -TimeoutSec $TimeoutSec -PassThru:$PassThru }
    catch { if (-not $NoThrow) { throw } ; return '' }
}

# -- Get VM state from multipass list CSV --------------------------------------
function Get-VmState ([string]$Name) {
    $csv = Invoke-Mp -Args @('list','--format','csv') -PassThru -NoThrow
    foreach ($line in ($csv -split "`n")) {
        $cols = $line -split ','
        if ($cols[0].Trim() -eq $Name) { return $cols[1].Trim() }
    }
    return $null
}

function Get-VmIPv4 ([string]$Name) {
    $csv = Invoke-Mp -Args @('list','--format','csv') -PassThru -NoThrow
    foreach ($line in ($csv -split "`n")) {
        $cols = $line -split ','
        if ($cols[0].Trim() -eq $Name -and $cols.Count -ge 3) { return $cols[2].Trim() }
    }
    return '-'
}

function Test-VmFile ([string]$Name, [string]$Path) {
    & multipass exec $Name -- test -f $Path 2>$null
    return ($LASTEXITCODE -eq 0)
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

# ------------------------------------------------------------------------------
# DESTROY MODE
# ------------------------------------------------------------------------------
if ($Destroy) {
    Write-Hdr 'DESTROY MODE -- tearing down fleet'
    Write-Warn 'This will DELETE all cluster VMs. You have 5 seconds to press Ctrl-C.'
    Start-Sleep -Seconds 5

    foreach ($vm in $Fleet) {
        $state = Get-VmState $vm.Name
        if ($state) {
            Write-Log "Deleting $($vm.Name) (state: $state)..."
            Invoke-Mp -Args @('delete', $vm.Name) -NoThrow | Out-Null
        } else {
            Write-Warn "$($vm.Name) not found -- skipping"
        }
    }
    Write-Log 'Purging disk images...'
    Invoke-Mp -Args @('purge') -NoThrow | Out-Null
    Write-Ok 'All cluster VMs deleted and purged.'
    exit 0
}

# ------------------------------------------------------------------------------
# STEP 1 -- EXECUTION POLICY
# ------------------------------------------------------------------------------
Write-Hdr 'Step 1 -- Execution Policy'

$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -in @('Restricted','AllSigned')) {
    Write-Warn "Execution policy is '$policy' -- setting to RemoteSigned for CurrentUser..."
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
    Write-Ok 'Execution policy updated'
} else {
    Write-Ok "Execution policy: $policy"
}

# ------------------------------------------------------------------------------
# STEP 2 -- WINGET
# ------------------------------------------------------------------------------
Write-Hdr 'Step 2 -- Winget (Windows Package Manager)'

function Test-Winget {
    $null = Get-Command winget -ErrorAction SilentlyContinue
    return $?
}

if (-not (Test-Winget)) {
    Write-Warn 'winget not found -- attempting to install via Microsoft Store AppInstaller...'
    # The recommended way on fresh Windows 10/11 is to update App Installer from the Store
    Write-Host '  Opening Microsoft Store App Installer page...' -ForegroundColor Yellow
    Start-Process 'ms-windows-store://pdp/?ProductId=9NBLGGH4NNS1'
    Write-Host ''
    Write-Host '  Please install/update "App Installer" from the Store, then re-run this script.' -ForegroundColor Yellow
    Write-Host '  Alternatively, download winget from: https://github.com/microsoft/winget-cli/releases' -ForegroundColor Yellow
    Exit-Fatal 'winget is required. Install it and re-run.'
} else {
    $wgVer = (winget --version 2>$null) -replace '^v',''
    Write-Ok "winget $wgVer"
}

# ------------------------------------------------------------------------------
# STEP 3 -- PYTHON
# ------------------------------------------------------------------------------
Write-Hdr 'Step 3 -- Python 3'

function Test-Python {
    try {
        $ver = & python --version 2>&1
        return ($ver -match 'Python 3\.[9-9]|Python 3\.1[0-9]')
    } catch { return $false }
}

if (-not (Test-Python)) {
    Write-Log 'Installing Python 3.11 via winget...'
    winget install --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable('PATH','Machine') + ';' +
                [System.Environment]::GetEnvironmentVariable('PATH','User')
    if (-not (Test-Python)) {
        Exit-Fatal 'Python install failed. Install from https://python.org and re-run.'
    }
    Write-Ok 'Python installed'
} else {
    $pyVer = & python --version 2>&1
    Write-Ok $pyVer
}

# ------------------------------------------------------------------------------
# STEP 4 -- MULTIPASS
# ------------------------------------------------------------------------------
Write-Hdr 'Step 4 -- Multipass'

function Test-Multipass {
    $null = Get-Command multipass -ErrorAction SilentlyContinue
    return $?
}

if (-not (Test-Multipass)) {
    Write-Log 'Installing Multipass via winget...'
    winget install --id Canonical.Multipass --silent --accept-package-agreements --accept-source-agreements
    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable('PATH','Machine') + ';' +
                [System.Environment]::GetEnvironmentVariable('PATH','User')
    if (-not (Test-Multipass)) {
        Exit-Fatal 'Multipass install failed. Install from https://multipass.run and re-run.'
    }
    Write-Ok 'Multipass installed'
    Write-Warn 'A reboot may be required for Hyper-V/VirtualBox backend -- if VMs fail to launch, reboot and re-run.'
} else {
    $mpVer = (multipass version 2>$null | Select-Object -First 1)
    Write-Ok "Multipass -- $mpVer"
}

# ------------------------------------------------------------------------------
# STEP 5 -- PYTHON DASHBOARD DEPENDENCIES
# ------------------------------------------------------------------------------
Write-Hdr 'Step 5 -- Python dashboard dependencies'

$ReqFile = Join-Path $ScriptDir 'requirements.txt'
if (-not (Test-Path $ReqFile)) {
    Write-Warn 'requirements.txt not found -- writing defaults...'
    @'
streamlit>=1.35.0
plotly>=5.20.0
pandas>=2.0.0
psutil>=5.9.0
lightgbm>=4.3.0
scikit-learn>=1.4.0
numpy>=1.26.0
'@ | Set-Content $ReqFile -Encoding UTF8
}

Write-Log 'pip install -r requirements.txt...'
& python -m pip install --quiet --upgrade pip
& python -m pip install --quiet -r $ReqFile
if ($LASTEXITCODE -ne 0) { Exit-Fatal 'pip install failed.' }
Write-Ok 'Python dependencies installed'

# ------------------------------------------------------------------------------
# STEP 6 -- LAUNCH VM FLEET
# ------------------------------------------------------------------------------
Write-Hdr 'Step 6 -- Launching VM fleet'

# Print fleet table
Write-Host ('  {0,-22} {1,-8} {2,-8} {3,-10} {4}' -f 'NAME','vCPUs','MEM','DISK','ROLE') -ForegroundColor Gray
Write-Host ('  {0,-22} {1,-8} {2,-8} {3,-10} {4}' -f '----------------------','------','------','--------','------------') -ForegroundColor DarkGray
foreach ($vm in $Fleet) {
    Write-Host ('  {0,-22} {1,-8} {2,-8} {3,-10} {4}' -f `
        $vm.Name, "$($vm.CPUs) vCPU", $vm.Memory, "$($vm.DiskGB) GB", $vm.Role) -ForegroundColor White
}
Write-Host ''

$Launched = 0
$Skipped  = 0
$Started  = 0
$LaunchFailed = 0

foreach ($vm in $Fleet) {
    $existing = Get-VmState $vm.Name
    if ($existing) {
        if ($existing -ne 'Running') {
            Write-Warn "$($vm.Name) already exists (state: $existing) -- starting"
            try {
                Invoke-Mp -Args @('start', $vm.Name) -TimeoutSec 180 | Out-Null
                Write-Ok "Started $($vm.Name)"
                $Started++
            } catch {
                Write-Err "Failed to start $($vm.Name): $_"
                $LaunchFailed++
            }
        } else {
            Write-Warn "$($vm.Name) already exists (state: $existing) -- skipping launch"
        }
        $Skipped++
        continue
    }

    Write-Log "Launching $($vm.Name) ($($vm.CPUs) vCPU * $($vm.Memory) RAM * $($vm.DiskGB)G disk)..."
    try {
        $launchArgs = @(
            'launch', $UbuntuImage,
            '--name',   $vm.Name,
            '--cpus',   "$($vm.CPUs)",
            '--memory', $vm.Memory,
            '--disk',   "$($vm.DiskGB)G",
            '--timeout','300'
        )
        & multipass @launchArgs
        if ($LASTEXITCODE -ne 0) { throw "multipass launch exited $LASTEXITCODE" }
        Write-Ok "Launched $($vm.Name)"
        $Launched++
    } catch {
        Write-Err "Failed to launch $($vm.Name): $_"
        $LaunchFailed++
    }
}

Write-Ok "Fleet launch complete (launched=$Launched, started=$Started, skipped=$Skipped, failed=$LaunchFailed)"

# ------------------------------------------------------------------------------
# STEP 7 -- WAIT FOR RUNNING STATE
# ------------------------------------------------------------------------------
Write-Hdr 'Step 7 -- Waiting for all VMs to reach Running state'

function Wait-ForVm ([string]$Name, [int]$MaxWaitSec = 180) {
    $waited = 0
    while ($waited -lt $MaxWaitSec) {
        $state = Get-VmState $Name
        if ($state -eq 'Running') {
            Write-Ok "$Name is Running"
            return $true
        }
        Write-Log "$Name state=$state ... waiting ($waited/$MaxWaitSec s)"
        Start-Sleep -Seconds 5
        $waited += 5
    }
    Write-Err "$Name did not reach Running within ${MaxWaitSec}s"
    return $false
}

$allRunning = $true
foreach ($vm in $Fleet) {
    if (-not (Wait-ForVm $vm.Name)) { $allRunning = $false }
}
if (-not $allRunning) { Write-Warn "Some VMs may not be Running -- check: multipass list" }

# ------------------------------------------------------------------------------
# STEP 8 -- BOOTSTRAP VMs (parallel jobs)
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
    $state = Get-VmState $vm.Name
    if ($state -ne 'Running') {
        Write-Warn "Skipping bootstrap for $($vm.Name) (state: $state)"
        continue
    }
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

# Collect job results
foreach ($job in $jobs) {
    $result = Receive-Job -Job $job -Wait -AutoRemoveJob
    if ($result.RC -eq 0) {
        Write-Ok "$($result.VM) bootstrapped"
    } else {
        Write-Err "$($result.VM) bootstrap FAILED"
        Write-Host $result.Output -ForegroundColor DarkGray
    }
}

Write-Ok 'All VM bootstraps complete'

# ------------------------------------------------------------------------------
# STEP 9 -- PUSH workload_generator.py
# ------------------------------------------------------------------------------
Write-Hdr 'Step 9 -- Distributing workload_generator.py'

$WlSrc = Join-Path $ScriptDir 'workload_generator.py'
if (-not (Test-Path $WlSrc)) {
    Write-Warn "workload_generator.py not found at $WlSrc -- skipping"
} else {
    foreach ($vm in $Fleet) {
        $state = Get-VmState $vm.Name
        if ($state -ne 'Running') { continue }
        try {
            & multipass transfer $WlSrc "$($vm.Name):/tmp/workload_generator.py"
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
    $state = Get-VmState $vm.Name
    if ($state -ne 'Running') {
        Write-Warn "$($vm.Name) not running -- skipping metric check"
        continue
    }
    try {
        $raw = Invoke-VmMetric -Name $vm.Name -Snippet $MetricSnippet
        $d   = $raw | ConvertFrom-Json
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
    $ip = Get-VmIPv4 $vm.Name
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
    Write-Ok 'Cluster ready. Dashboard skipped (-NoUI). Run: streamlit run dashboard.py'
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
