#!/usr/bin/env python3
"""
Workload Generator — runs inside a Multipass VM.
Simulates realistic data-center workload patterns:
  - compute-intensive  (CPU burn)
  - data-intensive     (disk I/O)
  - balanced           (mixed)
  - idle               (cool-down)

Usage (inside VM):
  python3 workload_generator.py --mode compute --duration 60 --intensity 0.8
"""
import argparse
import math
import os
import random
import signal
import socket
import sys
import tempfile
import threading
import time

# ── helpers ──────────────────────────────────────────────────────────────────

def cpu_burn(stop_event: threading.Event, intensity: float = 1.0):
    """Burn CPU at *intensity* fraction (0‑1) via a tight loop with sleep."""
    cycle_s = 0.05
    work_s  = cycle_s * intensity
    rest_s  = cycle_s - work_s
    while not stop_event.is_set():
        end = time.perf_counter() + work_s
        while time.perf_counter() < end:
            _ = math.sqrt(random.random())  # purposeful work
        if rest_s > 0:
            time.sleep(rest_s)


def disk_burn(stop_event: threading.Event, chunk_kb: int = 512):
    """Write-then-read temp files repeatedly."""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "wl_gen.bin")
    data = os.urandom(chunk_kb * 1024)
    while not stop_event.is_set():
        with open(path, "wb") as f:
            f.write(data)
        with open(path, "rb") as f:
            _ = f.read()
        time.sleep(0.01)


def mem_pressure(stop_event: threading.Event, mb: int = 128):
    """Hold a large byte-string in memory and touch it periodically."""
    buf = bytearray(mb * 1024 * 1024)
    while not stop_event.is_set():
        # touch pages so the OS doesn't swap them out immediately
        for i in range(0, len(buf), 4096):
            buf[i] = (buf[i] + 1) % 256
        time.sleep(0.5)


def net_loopback(stop_event: threading.Event, port: int = 19999):
    """
    Minimal loopback traffic: a server thread echoes, a client sends.
    Uses a random available port if 19999 is taken.
    """
    # find a free port
    try:
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        actual_port = srv.getsockname()[1]
        srv.listen(5)
    except OSError:
        return  # skip net workload if socket fails

    def _server():
        srv.settimeout(1.0)
        while not stop_event.is_set():
            try:
                conn, _ = srv.accept()
                with conn:
                    conn.settimeout(0.5)
                    try:
                        d = conn.recv(4096)
                        conn.sendall(d)
                    except Exception:
                        pass
            except socket.timeout:
                pass
        srv.close()

    def _client():
        payload = os.urandom(1024)
        while not stop_event.is_set():
            try:
                c = socket.create_connection(("127.0.0.1", actual_port), timeout=1)
                with c:
                    c.sendall(payload)
                    c.recv(2048)
            except Exception:
                pass
            time.sleep(0.05)

    threading.Thread(target=_server, daemon=True).start()
    threading.Thread(target=_client, daemon=True).start()


# ── mode definitions ──────────────────────────────────────────────────────────

MODES = {
    "compute": {
        "cpu_threads": 2,
        "cpu_intensity": 0.9,
        "disk": False,
        "mem_mb": 64,
        "net": False,
        "description": "Compute-intensive: heavy multi-threaded CPU burn",
    },
    "data": {
        "cpu_threads": 1,
        "cpu_intensity": 0.3,
        "disk": True,
        "mem_mb": 256,
        "net": True,
        "description": "Data-intensive: disk I/O + network + light CPU",
    },
    "balanced": {
        "cpu_threads": 1,
        "cpu_intensity": 0.55,
        "disk": True,
        "mem_mb": 128,
        "net": True,
        "description": "Balanced: moderate CPU + I/O + network",
    },
    "idle": {
        "cpu_threads": 0,
        "cpu_intensity": 0.0,
        "disk": False,
        "mem_mb": 16,
        "net": False,
        "description": "Idle: minimal resource usage",
    },
    "spike": {
        "cpu_threads": 4,
        "cpu_intensity": 1.0,
        "disk": True,
        "mem_mb": 512,
        "net": True,
        "description": "Spike: maximum resource stress (all subsystems)",
    },
}


def run_workload(mode: str, duration: float, intensity_override: float | None):
    cfg = MODES[mode]
    print(f"[workload_generator] mode={mode}  duration={duration}s")
    print(f"  {cfg['description']}")

    stop = threading.Event()
    threads = []

    # ── CPU ──
    n_cpu = cfg["cpu_threads"]
    eff_intensity = intensity_override if intensity_override is not None else cfg["cpu_intensity"]
    for _ in range(n_cpu):
        t = threading.Thread(target=cpu_burn, args=(stop, eff_intensity), daemon=True)
        t.start()
        threads.append(t)

    # ── Memory ──
    if cfg["mem_mb"] > 0:
        t = threading.Thread(target=mem_pressure, args=(stop, cfg["mem_mb"]), daemon=True)
        t.start()
        threads.append(t)

    # ── Disk ──
    if cfg["disk"]:
        t = threading.Thread(target=disk_burn, args=(stop,), daemon=True)
        t.start()
        threads.append(t)

    # ── Network ──
    if cfg["net"]:
        net_loopback(stop)

    # ── wait ──
    def _handle_sig(sig, frame):
        print("\n[workload_generator] Interrupted — shutting down.")
        stop.set()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join(timeout=2)
    print("[workload_generator] Done.")


# ── scenario runner ───────────────────────────────────────────────────────────

def run_scenario(scenario: str, total_duration: float):
    """
    Pre-canned multi-phase scenarios that mimic real DC workload patterns.
    Each phase is (mode, fraction_of_total).
    """
    SCENARIOS = {
        "ramp":   [("idle", .1), ("balanced", .2), ("compute", .4), ("spike", .2), ("idle", .1)],
        "sawtooth":[("idle",.15),("compute",.35),("idle",.15),("compute",.35)],
        "steady": [("balanced", 1.0)],
        "burst":  [("idle",.3),("spike",.1),("idle",.3),("spike",.1),("idle",.2)],
    }
    phases = SCENARIOS.get(scenario)
    if not phases:
        print(f"Unknown scenario '{scenario}'. Choose from: {list(SCENARIOS)}")
        sys.exit(1)

    print(f"[workload_generator] Running scenario '{scenario}' for {total_duration}s")
    for mode, frac in phases:
        phase_dur = total_duration * frac
        print(f"  → phase {mode} for {phase_dur:.1f}s")
        run_workload(mode, phase_dur, None)
    print("[workload_generator] Scenario complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="VM Workload Generator")
    ap.add_argument("--mode",      default="balanced",
                    choices=list(MODES), help="Workload mode")
    ap.add_argument("--scenario",  default=None,
                    choices=["ramp","sawtooth","steady","burst"],
                    help="Multi-phase scenario (overrides --mode)")
    ap.add_argument("--duration",  type=float, default=60.0,
                    help="Total duration in seconds")
    ap.add_argument("--intensity", type=float, default=None,
                    help="CPU intensity override 0-1 (mode default used if omitted)")
    args = ap.parse_args()

    if args.scenario:
        run_scenario(args.scenario, args.duration)
    else:
        run_workload(args.mode, args.duration, args.intensity)


if __name__ == "__main__":
    main()