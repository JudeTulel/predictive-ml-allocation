"""
multipass_controller.py
Thin wrapper around the `multipass` CLI for the dashboard.

Responsibilities
────────────────
• List VMs and their states / resource info
• Start / stop / suspend individual VMs
• Push the workload generator into a VM and execute it
• Stream metrics from a running VM via `multipass exec`
• Scale-out (launch) and scale-in (delete) VMs
"""

from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class VMInfo:
    name:    str
    state:   str          # Running / Stopped / Suspended / Starting
    ipv4:    str = ""
    cpus:    int = 0
    mem_gb:  float = 0.0
    disk_gb: float = 0.0
    role:    str = "unknown"   # load-balancer | compute | data
    # live metrics (populated by MetricsCollector)
    cpu_pct:  float = 0.0
    mem_pct:  float = 0.0
    disk_io:  float = 0.0
    net_kbps: float = 0.0


# ── VM role catalogue (matches your Multipass setup) ─────────────────────────

VM_ROLES: dict[str, str] = {
    # fill in your actual VM names; partial-match used below
    "lb":      "load-balancer",
    "compute": "compute",
    "data":    "data",
}

def _infer_role(name: str) -> str:
    n = name.lower()
    for key, role in VM_ROLES.items():
        if key in n:
            return role
    return "compute"   # sane default


# ── subprocess helper ─────────────────────────────────────────────────────────

def _run(cmd: list[str], *, timeout: int = 30) -> tuple[int, str, str]:
    """Run a command; return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 1, "", "Timeout"


def _mp(*args, **kwargs) -> tuple[int, str, str]:
    return _run(["multipass", *args], **kwargs)


# ── Core controller ───────────────────────────────────────────────────────────

class MultipassController:
    """High-level interface to the local Multipass daemon."""

    # ── inventory ─────────────────────────────────────────────────────────────

    def list_vms(self) -> list[VMInfo]:
        rc, out, err = _mp("list", "--format", "json")
        if rc != 0:
            return []
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return []

        vms = []
        for item in data.get("list", []):
            name  = item.get("name", "")
            state = item.get("state", "Unknown")
            ipv4_list = item.get("ipv4", [])
            ipv4  = ipv4_list[0] if ipv4_list else ""
            vm = VMInfo(
                name=name,
                state=state,
                ipv4=ipv4,
                role=_infer_role(name),
            )
            vms.append(vm)
        return vms

    def info(self, name: str) -> dict:
        rc, out, _ = _mp("info", name, "--format", "json")
        if rc != 0:
            return {}
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {}

    # ── power operations ──────────────────────────────────────────────────────

    def start(self, name: str) -> bool:
        rc, _, _ = _mp("start", name, timeout=60)
        return rc == 0

    def stop(self, name: str) -> bool:
        rc, _, _ = _mp("stop", name, timeout=60)
        return rc == 0

    def suspend(self, name: str) -> bool:
        rc, _, _ = _mp("suspend", name, timeout=60)
        return rc == 0

    def restart(self, name: str) -> bool:
        rc, _, _ = _mp("restart", name, timeout=90)
        return rc == 0

    # ── scaling ───────────────────────────────────────────────────────────────

    def scale_out(
        self,
        name: str,
        cpus: int = 2,
        mem_gb: int = 4,
        disk_gb: int = 10,
        image: str = "22.04",
    ) -> bool:
        """Launch a new VM."""
        rc, _, err = _mp(
            "launch",
            image,
            "--name",  name,
            "--cpus",  str(cpus),
            "--memory", f"{mem_gb}G",
            "--disk",   f"{disk_gb}G",
            timeout=180,
        )
        if rc != 0:
            print(f"[controller] scale_out failed: {err}")
        return rc == 0

    def scale_in(self, name: str, purge: bool = True) -> bool:
        """Delete (and optionally purge) a VM."""
        cmds = [["multipass", "delete", name]]
        if purge:
            cmds.append(["multipass", "purge"])
        for cmd in cmds:
            _run(cmd, timeout=60)
        return True

    # ── file transfer ─────────────────────────────────────────────────────────

    def push_file(self, local_path: str, vm_name: str, remote_path: str) -> bool:
        rc, _, err = _mp("transfer", local_path, f"{vm_name}:{remote_path}", timeout=30)
        if rc != 0:
            print(f"[controller] push_file failed: {err}")
        return rc == 0

    # ── workload dispatch ─────────────────────────────────────────────────────

    def run_workload(
        self,
        vm_name: str,
        mode: str = "balanced",
        duration: int = 60,
        intensity: float | None = None,
        scenario: str | None = None,
        on_finish: Callable[[str, bool], None] | None = None,
    ) -> threading.Thread:
        """
        Upload workload_generator.py to the VM and run it asynchronously.
        Returns the background thread so callers can track it.
        """
        import os

        local_wl = os.path.join(os.path.dirname(__file__), "workload_generator.py")

        def _worker():
            # 1. push file
            ok = self.push_file(local_wl, vm_name, "/tmp/workload_generator.py")
            if not ok:
                if on_finish:
                    on_finish(vm_name, False)
                return

            # 2. build command
            extra = []
            if scenario:
                extra += ["--scenario", scenario]
            else:
                extra += ["--mode", mode]
            if intensity is not None:
                extra += ["--intensity", str(intensity)]
            extra += ["--duration", str(duration)]

            cmd = ["multipass", "exec", vm_name, "--",
                   "python3", "/tmp/workload_generator.py"] + extra

            rc, _, _ = _run(cmd, timeout=duration + 30)
            if on_finish:
                on_finish(vm_name, rc == 0)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t

    # ── live metrics (via multipass exec + python psutil) ────────────────────

    def fetch_metrics(self, vm_name: str) -> dict:
        """
        Run a one-shot psutil snippet inside the VM and return a dict with:
          cpu_pct, mem_pct, mem_used_mb, mem_total_mb,
          disk_read_mb, disk_write_mb, net_sent_kb, net_recv_kb
        """
        snippet = (
            "import psutil, json, sys;"
            "d=psutil.disk_io_counters();"
            "n=psutil.net_io_counters();"
            "m=psutil.virtual_memory();"
            "print(json.dumps({"
            "'cpu':psutil.cpu_percent(interval=1),"
            "'mem_pct':m.percent,"
            "'mem_used':m.used//1024//1024,"
            "'mem_total':m.total//1024//1024,"
            "'disk_r':d.read_bytes//1024//1024 if d else 0,"
            "'disk_w':d.write_bytes//1024//1024 if d else 0,"
            "'net_s':n.bytes_sent//1024 if n else 0,"
            "'net_r':n.bytes_recv//1024 if n else 0,"
            "}))"
        )
        cmd = ["multipass", "exec", vm_name, "--", "python3", "-c", snippet]
        rc, out, _ = _run(cmd, timeout=10)
        if rc != 0 or not out:
            return {}
        try:
            return json.loads(out.strip())
        except json.JSONDecodeError:
            return {}

    def install_psutil(self, vm_name: str) -> bool:
        """Ensure psutil is available in the VM."""
        cmd = ["multipass", "exec", vm_name, "--",
               "pip3", "install", "psutil", "--quiet", "--break-system-packages"]
        rc, _, _ = _run(cmd, timeout=60)
        return rc == 0


# ── Background metrics collector ─────────────────────────────────────────────

class MetricsCollector:
    """
    Polls all running VMs every *interval* seconds and keeps a
    rolling history of `max_history` samples per VM.
    """

    def __init__(self, controller: MultipassController, interval: float = 15.0, max_history: int = 60):
        self._ctrl     = controller
        self._interval = interval
        self._max      = max_history
        self._lock     = threading.Lock()
        self._history: dict[str, list[dict]] = {}
        self._running  = False
        self._thread: threading.Thread | None = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_history(self, vm_name: str) -> list[dict]:
        with self._lock:
            return list(self._history.get(vm_name, []))

    def latest(self, vm_name: str) -> dict:
        hist = self.get_history(vm_name)
        return hist[-1] if hist else {}

    def _loop(self):
        while self._running:
            vms = self._ctrl.list_vms()
            for vm in vms:
                if vm.state.lower() != "running":
                    continue
                try:
                    m = self._ctrl.fetch_metrics(vm.name)
                    if m:
                        m["ts"] = time.time()
                        m["vm"] = vm.name
                        with self._lock:
                            hist = self._history.setdefault(vm.name, [])
                            hist.append(m)
                            if len(hist) > self._max:
                                hist.pop(0)
                except Exception:
                    pass
            time.sleep(self._interval)