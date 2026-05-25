#!/usr/bin/env python3
"""
dispatch_workload.py
────────────────────
Convenience CLI to dispatch workloads to Multipass VMs by ROLE
without opening the Streamlit dashboard.

Examples
────────
# Fire a 'ramp' scenario at all compute nodes for 3 minutes
python3 dispatch_workload.py --role compute --scenario ramp --duration 180

# Spike all data nodes for 45 s
python3 dispatch_workload.py --role data --mode spike --duration 45

# Blast every running VM
python3 dispatch_workload.py --role all --mode balanced --duration 120

# Scale-out a new compute node then immediately stress it
python3 dispatch_workload.py --scale-out my-compute-4 --cpus 2 --mem 4 \\
    && python3 dispatch_workload.py --vm my-compute-4 --mode compute --duration 120
"""

import argparse
import sys
import time
import threading

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from multipass_controller import MultipassController, _infer_role

ctrl = MultipassController()


def main():
    ap = argparse.ArgumentParser(description="Multipass Workload Dispatcher")

    # targeting
    tgt = ap.add_mutually_exclusive_group()
    tgt.add_argument("--role", choices=["compute","data","load-balancer","all"],
                     help="Target VMs by role")
    tgt.add_argument("--vm",  help="Target a specific VM by name")

    # workload
    ap.add_argument("--mode",     default="balanced",
                    choices=["balanced","compute","data","idle","spike"])
    ap.add_argument("--scenario", default=None,
                    choices=["ramp","sawtooth","steady","burst"],
                    help="Multi-phase scenario (overrides --mode)")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--intensity", type=float, default=None)

    # scaling
    ap.add_argument("--scale-out", metavar="NAME",
                    help="Launch a new VM with this name before dispatching")
    ap.add_argument("--scale-in",  metavar="NAME",
                    help="Delete this VM after workload completes")
    ap.add_argument("--cpus",  type=int, default=2)
    ap.add_argument("--mem",   type=int, default=4, help="Memory in GB")
    ap.add_argument("--disk",  type=int, default=10, help="Disk in GB")

    args = ap.parse_args()

    # ── optional scale-out first ──
    if args.scale_out:
        print(f"[dispatch] Launching VM '{args.scale_out}' ({args.cpus} vCPU, {args.mem}G RAM)…")
        ok = ctrl.scale_out(args.scale_out, args.cpus, args.mem, args.disk)
        if not ok:
            print("[dispatch] ✗ Launch failed. Aborting.")
            sys.exit(1)
        print("[dispatch] ✓ VM launched. Waiting 5 s for boot…")
        time.sleep(5)
        # install psutil
        ctrl.install_psutil(args.scale_out)
        target_names = [args.scale_out]
    elif args.vm:
        target_names = [args.vm]
    elif args.role:
        all_vms = ctrl.list_vms()
        running = [v for v in all_vms if v.state.lower() == "running"]
        if args.role == "all":
            target_names = [v.name for v in running]
        else:
            target_names = [v.name for v in running if v.role == args.role]
    else:
        ap.print_help()
        sys.exit(0)

    if not target_names:
        print("[dispatch] No matching VMs found (or none are running).")
        sys.exit(0)

    print(f"[dispatch] Target VMs: {target_names}")
    done_events: dict[str, bool] = {}
    threads = []

    def _done(name, ok):
        done_events[name] = ok
        print(f"[dispatch] {'✓' if ok else '✗'} {name} workload {'done' if ok else 'FAILED'}")

    for vm_name in target_names:
        t = ctrl.run_workload(
            vm_name,
            mode=args.mode,
            duration=int(args.duration),
            intensity=args.intensity,
            scenario=args.scenario,
            on_finish=_done,
        )
        threads.append(t)
        print(f"[dispatch] → Dispatched to {vm_name}")

    # wait for all
    for t in threads:
        t.join()

    # ── optional scale-in ──
    if args.scale_in:
        print(f"[dispatch] Deleting VM '{args.scale_in}'…")
        ctrl.scale_in(args.scale_in)
        print("[dispatch] ✓ VM deleted.")


if __name__ == "__main__":
    main()