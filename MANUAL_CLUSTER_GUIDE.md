# Manual Cluster Setup Guide

This guide shows how to do by hand what `bootstrap_cluster.ps1` and `resume_cluster.ps1` automate.

Run these commands from the project folder:

```powershell
cd G:\RojG
```

## 1. Allow Local PowerShell Scripts

Check the current policy:

```powershell
Get-ExecutionPolicy -Scope CurrentUser
```

If it is `Restricted` or `AllSigned`, set it to `RemoteSigned`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

If Windows marks a script as downloaded, unblock it:

```powershell
Unblock-File .\bootstrap_cluster.ps1
Unblock-File .\resume_cluster.ps1
```

## 2. Install Required Windows Tools

Check `winget`:

```powershell
winget --version
```

If `winget` is missing, install or update **App Installer** from the Microsoft Store.

Check Python:

```powershell
python --version
```

If Python 3.9 or newer is missing:

```powershell
winget install --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
```

Check Multipass:

```powershell
multipass version
```

If Multipass is missing:

```powershell
winget install --id Canonical.Multipass --silent --accept-package-agreements --accept-source-agreements
```

After installing Python or Multipass, restart PowerShell so your `PATH` is refreshed.

## 3. Install Dashboard Dependencies

Install the Python packages used by the local dashboard:

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

If `requirements.txt` is missing, create it with:

```text
streamlit>=1.35.0
plotly>=5.20.0
pandas>=2.0.0
psutil>=5.9.0
lightgbm>=4.3.0
scikit-learn>=1.4.0
numpy>=1.26.0
```

## 4. Create the VM Fleet

The current low-memory VM layout is:

| VM | vCPU | Memory | Disk | Role |
| --- | ---: | ---: | ---: | --- |
| `lb-node` | 1 | 512M | 5G | load-balancer |
| `compute-node-1` | 2 | 1024M | 10G | compute |
| `compute-node-2` | 2 | 1024M | 10G | compute |
| `data-node-1` | 1 | 2G | 20G | data |
| `data-node-2` | 1 | 2G | 20G | data |

Create each VM:

```powershell
multipass launch 22.04 --name lb-node --cpus 1 --memory 512M --disk 5G --timeout 300
multipass launch 22.04 --name compute-node-1 --cpus 2 --memory 1024M --disk 10G --timeout 300
multipass launch 22.04 --name compute-node-2 --cpus 2 --memory 1024M --disk 10G --timeout 300
multipass launch 22.04 --name data-node-1 --cpus 1 --memory 2G --disk 20G --timeout 300
multipass launch 22.04 --name data-node-2 --cpus 1 --memory 2G --disk 20G --timeout 300
```

If a VM already exists, do not recreate it. Start it instead:

```powershell
multipass start lb-node
```

Check all VMs:

```powershell
multipass list
```

## 5. Wait Until VMs Are Really Ready

Do not rely only on `multipass list`. A better readiness test is whether the VM accepts a guest command:

```powershell
multipass exec lb-node -- true
multipass exec compute-node-1 -- true
multipass exec compute-node-2 -- true
multipass exec data-node-1 -- true
multipass exec data-node-2 -- true
```

If a VM fails that check, start it and try again:

```powershell
multipass start <vm-name>
multipass exec <vm-name> -- true
```

Repeat until the command exits cleanly.

## 6. Bootstrap Each VM

Run this for each VM name:

```powershell
multipass exec <vm-name> -- bash -lc "set -e; export DEBIAN_FRONTEND=noninteractive; sudo apt-get update -qq; sudo apt-get install -y -qq python3 python3-pip stress-ng sysstat; pip3 install psutil --quiet --break-system-packages 2>/dev/null || pip3 install psutil --quiet; python3 -c 'import psutil; print(psutil.__version__)'; sudo mkdir -p /var/local; date -Is | sudo tee /var/local/rojg-bootstrap.done >/dev/null"
```

Example:

```powershell
multipass exec lb-node -- bash -lc "set -e; export DEBIAN_FRONTEND=noninteractive; sudo apt-get update -qq; sudo apt-get install -y -qq python3 python3-pip stress-ng sysstat; pip3 install psutil --quiet --break-system-packages 2>/dev/null || pip3 install psutil --quiet; python3 -c 'import psutil; print(psutil.__version__)'; sudo mkdir -p /var/local; date -Is | sudo tee /var/local/rojg-bootstrap.done >/dev/null"
```

To check whether a VM was already bootstrapped:

```powershell
multipass exec <vm-name> -- test -f /var/local/rojg-bootstrap.done
```

If that command exits successfully, you can skip bootstrapping that VM.

## 7. Copy the Workload Generator

Copy `workload_generator.py` into every ready VM:

```powershell
multipass transfer .\workload_generator.py lb-node:/tmp/workload_generator.py
multipass transfer .\workload_generator.py compute-node-1:/tmp/workload_generator.py
multipass transfer .\workload_generator.py compute-node-2:/tmp/workload_generator.py
multipass transfer .\workload_generator.py data-node-1:/tmp/workload_generator.py
multipass transfer .\workload_generator.py data-node-2:/tmp/workload_generator.py
```

## 8. Verify Metrics

Run this against each VM:

```powershell
multipass exec <vm-name> -- python3 -c "import psutil,json; m=psutil.virtual_memory(); print(json.dumps({'cpu':psutil.cpu_percent(interval=1),'mem_pct':m.percent,'mem_mb':m.total//1024//1024}))"
```

Example:

```powershell
multipass exec compute-node-1 -- python3 -c "import psutil,json; m=psutil.virtual_memory(); print(json.dumps({'cpu':psutil.cpu_percent(interval=1),'mem_pct':m.percent,'mem_mb':m.total//1024//1024}))"
```

You should see JSON containing CPU percentage, memory percentage, and total RAM in MB.

## 9. Print a Cluster Summary

Show VM status and IP addresses:

```powershell
multipass list
multipass info lb-node
multipass info compute-node-1
multipass info compute-node-2
multipass info data-node-1
multipass info data-node-2
```

Useful manual commands:

```powershell
multipass shell <vm-name>
multipass exec <vm-name> -- <command>
python dispatch_workload.py --role all --mode ramp --duration 120
```

## 10. Launch the Dashboard

Start Streamlit:

```powershell
streamlit run .\dashboard.py `
    --server.port 8501 `
    --server.address 0.0.0.0 `
    --browser.gatherUsageStats false `
    --theme.base dark `
    --theme.backgroundColor '#040d14' `
    --theme.secondaryBackgroundColor '#0a1a26' `
    --theme.primaryColor '#00d4ff' `
    --theme.textColor '#e8f4fd'
```

Open:

```text
http://localhost:8501
```

Stop the dashboard with `Ctrl-C`.

## Resume Checklist

If setup fails halfway through:

1. Run `multipass list`.
2. Start any stopped VM with `multipass start <vm-name>`.
3. Check readiness with `multipass exec <vm-name> -- true`.
4. Skip bootstrapping VMs where this succeeds:

```powershell
multipass exec <vm-name> -- test -f /var/local/rojg-bootstrap.done
```

5. Continue from the first VM that does not have the marker file.

## Destroy and Recreate

Use this only when you want a clean slate:

```powershell
multipass delete lb-node compute-node-1 compute-node-2 data-node-1 data-node-2
multipass purge
```

Then recreate the fleet from step 4.
