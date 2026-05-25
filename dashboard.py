"""
dashboard.py — DataCenter AI Controller
Streamlit frontend for the Two-Stage Chained Prediction Pipeline.

Pickle files expected alongside this script:
  stage1_model.pkl  — best_rf_model_stage1   (RandomForestRegressor, StandardScaler input)
  stage2_model.pkl  — lgbm_model_stage2_oracle (LGBMRegressor, StandardScaler input + predicted CPU appended)
  scaler.pkl        — StandardScaler fitted on X_train (needed to transform live psutil metrics)

Run:
  streamlit run dashboard.py
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multipass_controller import MultipassController, MetricsCollector

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DataCenter AI Controller",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg-deep:   #040d14;
    --bg-card:   #0a1a26;
    --bg-panel:  #0f2133;
    --accent1:   #00d4ff;
    --accent2:   #7fff6e;
    --accent3:   #ff6b35;
    --warn:      #ffd166;
    --text-hi:   #e8f4fd;
    --text-lo:   #4a7a9b;
    --border:    rgba(0,212,255,0.18);
    --glow:      0 0 20px rgba(0,212,255,0.25);
}
html, body, .stApp { background: var(--bg-deep) !important; color: var(--text-hi) !important; font-family: 'Syne', sans-serif; }
[data-testid="stSidebar"] { background: var(--bg-card) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text-hi) !important; }
[data-testid="metric-container"] { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; box-shadow: var(--glow); }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 2rem !important; color: var(--accent1) !important; }
[data-testid="stMetricLabel"] { color: var(--text-lo) !important; }
.stButton > button { background: transparent !important; border: 1px solid var(--accent1) !important; color: var(--accent1) !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 600 !important; border-radius: 6px !important; transition: all .2s !important; }
.stButton > button:hover { background: rgba(0,212,255,0.12) !important; box-shadow: var(--glow) !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: var(--text-hi) !important; }
hr { border-color: var(--border) !important; }
.vm-running  { color: #7fff6e; font-weight: 700; }
.vm-stopped  { color: #ff6b35; font-weight: 700; }
.vm-starting { color: #ffd166; font-weight: 700; }
.log-box { background: #020a10; border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--accent1); max-height: 240px; overflow-y: auto; line-height: 1.6; white-space: pre-wrap; }
.pred-card { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; }
.pred-label { font-family: 'JetBrains Mono', monospace; font-size: .7rem; color: var(--text-lo); letter-spacing: .12em; text-transform: uppercase; }
.pred-value { font-family: 'JetBrains Mono', monospace; font-size: 1.7rem; font-weight: 700; margin: 4px 0 2px; }
.pred-sub   { font-size: .75rem; color: var(--text-lo); }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — Linear Power Model (Stage 2 training derivation)
# ══════════════════════════════════════════════════════════════════════════════

POWER_IDLE = 80.0   # W  — adjust to match your training derivation
POWER_MAX  = 200.0  # W

def cpu_to_watts(cpu_pct: float) -> float:
    return POWER_IDLE + (POWER_MAX - POWER_IDLE) * (cpu_pct / 100.0)

def watts_to_kwh(watts: float, hours: float) -> float:
    return watts * hours / 1000.0

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# Replicates the column order used during training.
# psutil gives us: cpu, mem_pct, mem_used, mem_total, disk_r, disk_w, net_s, net_r
#
# Your training features (X_train_scaled_standard columns):
#   Memory util ratio, Disk I/O total, Network total,
#   + lag features, rolling stats, time features, VM_ID embedding
#
# For live inference we only have the base telemetry features — lag and
# rolling stats require a history window, so we approximate them from the
# rolling history collected by MetricsCollector.
# ══════════════════════════════════════════════════════════════════════════════

# Number of features the Stage 1 model was trained on.
# Set this to match len(X_train.columns) from your notebook.
# The inference function will zero-pad or warn if the live vector is shorter.
STAGE1_N_FEATURES = None   # set automatically when scaler is loaded

def build_feature_vector(latest: dict, history: list[dict]) -> Optional[np.ndarray]:
    """
    Construct a single-row feature vector from live psutil metrics + rolling history.
    Returns None if there is not enough data yet.

    Feature layout (must match training column order):
      0  mem_util_ratio      — mem_pct / 100
      1  disk_io_total_mb    — disk_r + disk_w (MB)
      2  net_total_kb        — net_s + net_r (KB)
      3  cpu_lag1            — CPU% one poll ago  (15 s)
      4  cpu_lag2            — CPU% two polls ago (30 s)
      5  cpu_lag3            — CPU% three polls ago (45 s)
      6  cpu_roll_mean_5     — mean of last 5 CPU readings
      7  cpu_roll_std_5      — std  of last 5 CPU readings
      8  mem_roll_mean_5     — mean of last 5 mem readings
      9  hour_sin            — sin(2π·hour/24)
     10  hour_cos            — cos(2π·hour/24)
     11  minute_sin          — sin(2π·minute/60)
     12  minute_cos          — cos(2π·minute/60)

    If your training had more features (VM_ID embedding, more lags, etc.)
    drop the scaler.pkl in next to dashboard.py so the n_features mismatch
    is caught cleanly rather than silently producing wrong predictions.
    """
    if not latest:
        return None

    now_hour   = datetime.now().hour
    now_minute = datetime.now().minute

    cpu_hist = [h.get("cpu", 0.0) for h in history[-6:]] if history else []
    mem_hist = [h.get("mem_pct", 0.0) for h in history[-5:]] if history else []

    def _lag(n):
        idx = -(n + 1)
        return cpu_hist[idx] if len(cpu_hist) >= n + 1 else latest.get("cpu", 0.0)

    cpu_roll_vals = cpu_hist[-5:] if len(cpu_hist) >= 2 else [latest.get("cpu", 0.0)]
    mem_roll_vals = mem_hist[-5:] if len(mem_hist) >= 2 else [latest.get("mem_pct", 0.0)]

    vec = np.array([
        latest.get("mem_pct", 0.0) / 100.0,                              # 0 mem_util_ratio
        latest.get("disk_r", 0.0) + latest.get("disk_w", 0.0),           # 1 disk_io_total_mb
        latest.get("net_s", 0.0)  + latest.get("net_r", 0.0),            # 2 net_total_kb
        _lag(1),                                                           # 3 cpu_lag1
        _lag(2),                                                           # 4 cpu_lag2
        _lag(3),                                                           # 5 cpu_lag3
        float(np.mean(cpu_roll_vals)),                                     # 6 cpu_roll_mean_5
        float(np.std(cpu_roll_vals)) if len(cpu_roll_vals) > 1 else 0.0, # 7 cpu_roll_std_5
        float(np.mean(mem_roll_vals)),                                     # 8 mem_roll_mean_5
        np.sin(2 * np.pi * now_hour   / 24),                              # 9 hour_sin
        np.cos(2 * np.pi * now_hour   / 24),                              # 10 hour_cos
        np.sin(2 * np.pi * now_minute / 60),                              # 11 minute_sin
        np.cos(2 * np.pi * now_minute / 60),                              # 12 minute_cos
    ], dtype=np.float32)

    return vec.reshape(1, -1)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER  (cached — loads once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """
    Load stage1_model.pkl, stage2_model.pkl, and optionally scaler.pkl
    from the same directory as this script.
    Returns a dict with keys: stage1, stage2, scaler, errors, warnings
    """
    base = os.path.dirname(os.path.abspath(__file__))
    result = {"stage1": None, "stage2": None, "scaler": None,
              "errors": [], "warnings": []}

    for key, fname in [("stage1", "stage1_model.pkl"),
                        ("stage2", "stage2_model.pkl"),
                        ("scaler", "scaler.pkl")]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    result[key] = pickle.load(f)
            except Exception as e:
                result["errors"].append(f"Could not load {fname}: {e}")
        else:
            if key != "scaler":   # scaler is optional but strongly recommended
                result["warnings"].append(
                    f"{fname} not found — predictions disabled for this stage.")
            else:
                result["warnings"].append(
                    "scaler.pkl not found. Live features will be passed unscaled "
                    "(predictions will be inaccurate). Export your fitted StandardScaler "
                    "from the training notebook: pickle.dump(scaler, open('scaler.pkl','wb'))")

    if result["scaler"] is None:
        alt_scaler = os.path.join(base, "scaler_standard.pkl")
        if os.path.exists(alt_scaler):
            try:
                with open(alt_scaler, "rb") as f:
                    result["scaler"] = pickle.load(f)
                result["warnings"].append("Loaded scaler_standard.pkl as the StandardScaler fallback.")
            except Exception as e:
                result["errors"].append(f"Could not load scaler_standard.pkl: {e}")

    for key, fname in [("stage1", "stage1_model.pkl"),
                       ("stage2", "stage2_model.pkl"),
                       ("scaler", "scaler.pkl")]:
        if result[key] is None:
            continue
        try:
            check_is_fitted(result[key])
        except Exception as e:
            result[key] = None
            result["errors"].append(f"{fname} is not fitted or cannot be used: {e}")

    # validate feature count against scaler if available
    if result["scaler"] is not None:
        try:
            n = result["scaler"].n_features_in_
            result["n_features"] = n
        except AttributeError:
            result["n_features"] = None
    else:
        result["n_features"] = None

    if result["n_features"] is None:
        result["n_features"] = getattr(result["stage1"], "n_features_in_", None)
    if result["n_features"] is None and result["stage2"] is not None:
        stage2_n = getattr(result["stage2"], "n_features_in_", None)
        result["n_features"] = stage2_n - 1 if stage2_n else None

    stage2_n = getattr(result["stage2"], "n_features_in_", None)
    if result["n_features"] is not None and stage2_n and stage2_n != result["n_features"] + 1:
        result["errors"].append(
            f"Feature count mismatch: Stage 2 expects {stage2_n} columns, "
            f"but Stage 1/scaler features imply {result['n_features']} + 1."
        )

    return result


def align_feature_count(X: np.ndarray, n_expected: Optional[int]) -> np.ndarray:
    """Pad or truncate one-row live telemetry to the model feature count."""
    if n_expected is None or X.shape[1] == n_expected:
        return X
    if X.shape[1] < n_expected:
        pad = np.zeros((X.shape[0], n_expected - X.shape[1]), dtype=X.dtype)
        return np.hstack([X, pad])
    return X[:, :n_expected]


def run_inference(models: dict, latest: dict, history: list[dict]) -> dict:
    """
    Run the two-stage pipeline on a single observation.
    Returns a dict with prediction results and diagnostics.
    """
    out = {
        "cpu_pred":    None,   # Stage 1 output: predicted CPU% at t+5min
        "energy_pred": None,   # Stage 2 output: predicted Energy (W) at t+5min
        "energy_baseline": None,  # linear model baseline using current CPU
        "stage1_ok": False,
        "stage2_ok": False,
        "error":     None,
    }

    # baseline energy from linear model (always available)
    out["energy_baseline"] = cpu_to_watts(latest.get("cpu", 0.0))

    try:
        X_raw = build_feature_vector(latest, history)
        if X_raw is None:
            out["error"] = "Not enough history yet — waiting for at least 3 polling cycles."
            return out

        # ── scale ──
        X_raw = align_feature_count(X_raw, models.get("n_features"))

        if models["scaler"] is not None:
            n_expected = models.get("n_features")
            if n_expected is not None and X_raw.shape[1] != n_expected:
                # Pad or truncate to match scaler expectation
                if X_raw.shape[1] < n_expected:
                    pad = np.zeros((1, n_expected - X_raw.shape[1]), dtype=np.float32)
                    X_raw = np.hstack([X_raw, pad])
                else:
                    X_raw = X_raw[:, :n_expected]
            X_scaled = models["scaler"].transform(X_raw)
        else:
            X_scaled = X_raw   # unscaled — warn shown in UI

        # ── Stage 1: predict CPU% at t+5min ──
        if models["stage1"] is not None:
            cpu_pred = float(models["stage1"].predict(X_scaled)[0])
            cpu_pred = float(np.clip(cpu_pred, 0.0, 100.0))
            out["cpu_pred"]  = cpu_pred
            out["stage1_ok"] = True
        else:
            # fall back to current CPU as pass-through
            cpu_pred = latest.get("cpu", 0.0)

        # ── Stage 2: predict Energy (W) at t+5min ──
        # Stage 2 input = scaled Stage 1 features + predicted CPU appended as last column
        # This matches: X_test_stage2_pipeline = np.hstack((X_test_scaled_standard, y_pred.reshape(-1,1)))
        if models["stage2"] is not None:
            X_stage2 = np.hstack([X_scaled, [[cpu_pred]]])
            energy_pred = float(models["stage2"].predict(X_stage2)[0])
            energy_pred = max(0.0, energy_pred)
            out["energy_pred"] = energy_pred
            out["stage2_ok"]   = True

    except Exception as e:
        out["error"] = str(e)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# CONTROLLER + COLLECTOR  (cached singletons)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_controller():
    return MultipassController()

@st.cache_resource
def get_collector(_ctrl):
    c = MetricsCollector(_ctrl, interval=15)
    c.start()
    return c

ctrl      = get_controller()
collector = get_collector(ctrl)
models    = load_models()

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("logs", []),
    ("active_workloads", {}),
    ("last_refresh", 0.0),
    ("scale_count", 0),
    ("pred_history", {}),   # vm_name → list of {ts, cpu_pred, energy_pred, cpu_actual}
]:
    if key not in st.session_state:
        st.session_state[key] = default


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")
    if len(st.session_state.logs) > 300:
        st.session_state.logs = st.session_state.logs[-300:]


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="display:flex; align-items:center; gap:16px; padding:8px 0 20px;">
  <div style="font-size:2.8rem;">⚡</div>
  <div>
    <div style="font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800;
         background:linear-gradient(90deg,#00d4ff,#7fff6e); -webkit-background-clip:text;
         -webkit-text-fill-color:transparent; line-height:1.1;">
      DataCenter AI Controller
    </div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:.78rem; color:#4a7a9b; margin-top:4px;">
      Predictive Resource Allocation &nbsp;·&nbsp; Two-Stage Chained Pipeline &nbsp;·&nbsp; Energy Optimization
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🖥️ VM Fleet Controls")
    st.markdown("---")

    auto_refresh = st.checkbox("Auto-refresh (15 s)", value=True)
    if st.button("🔄 Refresh Now"):
        st.rerun()

    st.markdown("---")
    st.markdown("### 🚀 Workload Dispatch")

    vms_raw  = ctrl.list_vms()
    vm_names = [v.name for v in vms_raw]
    running  = [v.name for v in vms_raw if v.state.lower() == "running"]

    target_vms  = st.multiselect("Target VMs", options=vm_names,
                                  default=running[:1] if running else [])
    wl_mode     = st.selectbox("Workload Mode",
                                ["balanced", "compute", "data", "idle", "spike"])
    wl_scenario = st.selectbox("— or Scenario (overrides mode) —",
                                ["(none)", "ramp", "sawtooth", "steady", "burst"])
    wl_duration = st.slider("Duration (s)", 30, 600, 120, step=30)
    wl_intensity= st.slider("CPU Intensity override", 0.0, 1.0, 0.0, step=0.05,
                              help="0 = use mode default")

    c1, c2 = st.columns(2)
    launch_btn = c1.button("▶ Launch")
    _stop_btn  = c2.button("■ Stop All")

    if launch_btn:
        if not target_vms:
            st.warning("Select at least one VM.")
        else:
            scenario  = None if wl_scenario == "(none)" else wl_scenario
            intensity = wl_intensity if wl_intensity > 0 else None
            for vm in target_vms:
                def _done(name, ok, _vm=vm):
                    log(f"Workload on {_vm} {'completed ✓' if ok else 'FAILED ✗'}")
                    st.session_state.active_workloads.pop(_vm, None)
                ctrl.run_workload(vm, mode=wl_mode, duration=wl_duration,
                                  intensity=intensity, scenario=scenario, on_finish=_done)
                st.session_state.active_workloads[vm] = {
                    "mode": scenario or wl_mode,
                    "started": time.time(),
                    "duration": wl_duration,
                }
                log(f"Dispatched '{scenario or wl_mode}' → {vm} ({wl_duration}s)")
            st.success(f"Workload launched on {len(target_vms)} VM(s)")

    st.markdown("---")
    st.markdown("### ⚖️ Scale Fleet")

    with st.expander("Scale-Out: Launch new VM"):
        new_name  = st.text_input("VM name", value=f"compute-node-{st.session_state.scale_count+3}")
        new_cpus  = st.selectbox("vCPUs",       [1, 2, 4, 8], index=1)
        new_mem   = st.selectbox("Memory (GB)", [1, 2, 4, 8], index=1)
        new_disk  = st.selectbox("Disk (GB)",   [5, 10, 20, 40], index=1)
        new_image = st.selectbox("Ubuntu image", ["22.04", "24.04"])
        if st.button("🚀 Launch VM", key="launch_vm"):
            log(f"Launching '{new_name}' ({new_cpus} vCPU, {new_mem}GB)…")
            if ctrl.scale_out(new_name, new_cpus, new_mem, new_disk, new_image):
                st.session_state.scale_count += 1
                log(f"VM '{new_name}' launched ✓")
                st.success("Launched!")
            else:
                st.error("Launch failed — check multipass logs.")

    with st.expander("Scale-In: Remove VM"):
        del_vm = st.selectbox("VM to remove", ["(select)"] + vm_names)
        purge  = st.checkbox("Purge after delete", value=True)
        if st.button("🗑 Delete VM", key="delete_vm"):
            if del_vm == "(select)":
                st.warning("Pick a VM first.")
            else:
                ctrl.scale_in(del_vm, purge=purge)
                log(f"Deleted '{del_vm}' (purge={purge})")
                st.success(f"'{del_vm}' deleted.")

    st.markdown("---")
    st.markdown("### ⚙️ Bootstrap")
    if st.button("Install psutil on all running VMs"):
        for vm in running:
            ok = ctrl.install_psutil(vm)
            log(f"psutil → {vm}: {'ok' if ok else 'failed'}")

    # ── model status in sidebar ──
    st.markdown("---")
    st.markdown("### 🧠 Model Status")
    s1_ok = models["stage1"] is not None
    s2_ok = models["stage2"] is not None
    sc_ok = models["scaler"] is not None
    st.markdown(
        f"{'🟢' if s1_ok else '🔴'} Stage 1 (RF) &nbsp;&nbsp; "
        f"{'🟢' if s2_ok else '🔴'} Stage 2 (LGBM) &nbsp;&nbsp; "
        f"{'🟢' if sc_ok else '🟡'} Scaler",
        unsafe_allow_html=True
    )
    if not sc_ok:
        st.caption("⚠️ scaler.pkl missing — export it from your notebook.")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_fleet, tab_metrics, tab_predictions, tab_energy, tab_pipeline, tab_logs = st.tabs([
    "🖥 Fleet Overview",
    "📊 Live Metrics",
    "🔮 Live Predictions",
    "⚡ Energy Model",
    "🧠 ML Pipeline",
    "📋 Activity Log",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FLEET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_fleet:
    vms = ctrl.list_vms()
    n_total     = len(vms)
    n_running   = sum(1 for v in vms if v.state.lower() == "running")
    n_stopped   = n_total - n_running
    n_active_wl = len(st.session_state.active_workloads)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total VMs",        n_total)
    k2.metric("Running",          n_running, delta=f"{n_running - n_stopped} vs stopped")
    k3.metric("Active Workloads", n_active_wl)
    k4.metric("Est. Fleet Power", f"{n_running * POWER_IDLE:.0f} W",
              help="Idle baseline — rises with CPU load")

    st.markdown("---")
    st.markdown("#### VM Status")

    if not vms:
        st.info("No VMs found. Ensure `multipass list` works from this terminal.")
    else:
        for vm in vms:
            cols = st.columns([3, 2, 2, 3, 2, 2])
            sc   = {"running": "vm-running", "stopped": "vm-stopped"}.get(vm.state.lower(), "vm-starting")
            cols[0].markdown(f"**{vm.name}** &nbsp;<span style='color:#4a7a9b;font-size:.8rem'>{vm.role}</span>",
                             unsafe_allow_html=True)
            cols[1].markdown(f"<span class='{sc}'>{vm.state}</span>", unsafe_allow_html=True)
            cols[2].write(vm.ipv4 or "—")

            aw = st.session_state.active_workloads.get(vm.name)
            if aw:
                rem = max(0, aw["duration"] - int(time.time() - aw["started"]))
                cols[3].markdown(f"<span style='color:#ffd166'>⚙ {aw['mode']} ({rem}s left)</span>",
                                 unsafe_allow_html=True)
            else:
                cols[3].write("idle")

            with cols[4]:
                if vm.state.lower() == "running":
                    if st.button("⏸ Stop", key=f"stop_{vm.name}"):
                        ctrl.stop(vm.name); log(f"Stopped {vm.name}"); st.rerun()
                else:
                    if st.button("▶ Start", key=f"start_{vm.name}"):
                        ctrl.start(vm.name); log(f"Started {vm.name}"); st.rerun()
            with cols[5]:
                if st.button("🔁 Restart", key=f"restart_{vm.name}"):
                    ctrl.restart(vm.name); log(f"Restarted {vm.name}"); st.rerun()
            st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE METRICS
# ══════════════════════════════════════════════════════════════════════════════

with tab_metrics:
    st.markdown("#### Real-Time VM Telemetry")
    st.caption("Polls every 15 s via `multipass exec`. First reading takes ~5 s per VM.")

    vms_now     = ctrl.list_vms()
    running_vms = [v for v in vms_now if v.state.lower() == "running"]

    if not running_vms:
        st.info("No running VMs to poll.")
    else:
        rows = []
        for vm in running_vms:
            m = collector.latest(vm.name) or ctrl.fetch_metrics(vm.name)
            if m:
                rows.append({
                    "VM":           vm.name,
                    "CPU %":        round(m.get("cpu", 0), 1),
                    "Mem %":        round(m.get("mem_pct", 0), 1),
                    "Mem Used MB":  m.get("mem_used", 0),
                    "Disk Read MB": m.get("disk_r", 0),
                    "Net Recv KB":  m.get("net_r", 0),
                    "Power (W)":    round(cpu_to_watts(m.get("cpu", 0)), 1),
                })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            fig_g = make_subplots(rows=1, cols=len(rows),
                                  specs=[[{"type": "indicator"}] * len(rows)])
            for i, r in enumerate(rows):
                fig_g.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=r["CPU %"],
                    title={"text": r["VM"], "font": {"size": 11, "color": "#4a7a9b"}},
                    gauge={
                        "axis":  {"range": [0, 100], "tickcolor": "#4a7a9b"},
                        "bar":   {"color": "#00d4ff"},
                        "bgcolor": "#0f2133",
                        "steps": [{"range": [0, 40], "color": "#0a2a1e"},
                                  {"range": [40, 75], "color": "#1a2a0a"},
                                  {"range": [75, 100], "color": "#2a1010"}],
                        "threshold": {"line": {"color": "#ff6b35", "width": 3},
                                      "thickness": 0.75, "value": 80},
                    },
                    number={"suffix": "%", "font": {"color": "#00d4ff"}},
                ), row=1, col=i + 1)
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

        st.markdown("#### Historical CPU & Memory")
        for vm in running_vms:
            hist = collector.get_history(vm.name)
            if len(hist) < 2:
                st.caption(f"{vm.name}: collecting data…")
                continue
            df_h = pd.DataFrame(hist)
            df_h["time"] = pd.to_datetime(df_h["ts"], unit="s")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_h["time"], y=df_h["cpu"],
                                     name="CPU %", line=dict(color="#00d4ff", width=2)))
            fig.add_trace(go.Scatter(x=df_h["time"], y=df_h["mem_pct"],
                                     name="Mem %", line=dict(color="#7fff6e", width=2)))
            fig.update_layout(
                title=vm.name, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(4,13,20,1)", font=dict(color="#4a7a9b"),
                xaxis=dict(gridcolor="#0f2133"), yaxis=dict(gridcolor="#0f2133", range=[0, 100]),
                legend=dict(bgcolor="rgba(0,0,0,0)"), height=220,
                margin=dict(l=0, r=0, t=36, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE PREDICTIONS  (new tab)
# ══════════════════════════════════════════════════════════════════════════════

with tab_predictions:
    st.markdown("#### Two-Stage Pipeline — Live Inference")
    st.caption(
        "Stage 1 (Random Forest) predicts CPU% at t+5 min. "
        "Stage 2 (LightGBM) uses that prediction as an extra feature to forecast Energy (W) at t+5 min."
    )

    # show model/scaler status warnings
    for w in models["warnings"]:
        st.warning(w)
    for e in models["errors"]:
        st.error(e)

    vms_pred    = ctrl.list_vms()
    running_pred = [v for v in vms_pred if v.state.lower() == "running"]

    if not running_pred:
        st.info("No running VMs — start a VM and trigger a workload to see predictions.")
    elif models["stage1"] is None and models["stage2"] is None:
        st.error("No models loaded. Place `stage1_model.pkl` and `stage2_model.pkl` in the project folder.")
    else:
        for vm in running_pred:
            latest  = collector.latest(vm.name)
            history = collector.get_history(vm.name)

            with st.container():
                st.markdown(f"##### {vm.name} &nbsp;<span style='font-size:.8rem;color:#4a7a9b'>{vm.role}</span>",
                            unsafe_allow_html=True)

                if not latest:
                    st.caption("Waiting for first metric poll…")
                    continue

                preds = run_inference(models, latest, history)

                # ── store prediction history ──
                ph = st.session_state.pred_history.setdefault(vm.name, [])
                ph.append({
                    "ts":           time.time(),
                    "cpu_actual":   latest.get("cpu", 0.0),
                    "cpu_pred":     preds["cpu_pred"],
                    "energy_actual":cpu_to_watts(latest.get("cpu", 0.0)),
                    "energy_pred":  preds["energy_pred"],
                    "energy_base":  preds["energy_baseline"],
                })
                if len(ph) > 120:
                    st.session_state.pred_history[vm.name] = ph[-120:]

                if preds["error"]:
                    st.warning(f"Inference: {preds['error']}")

                # ── prediction cards ──
                pc1, pc2, pc3, pc4 = st.columns(4)

                current_cpu = latest.get("cpu", 0.0)
                pred_cpu    = preds["cpu_pred"]
                pred_energy = preds["energy_pred"]
                base_energy = preds["energy_baseline"]

                with pc1:
                    st.markdown(f"""
                    <div class="pred-card">
                      <div class="pred-label">CPU Now</div>
                      <div class="pred-value" style="color:#00d4ff">{current_cpu:.1f}%</div>
                      <div class="pred-sub">current utilisation</div>
                    </div>""", unsafe_allow_html=True)

                with pc2:
                    if pred_cpu is not None:
                        delta_cpu = pred_cpu - current_cpu
                        arrow = "▲" if delta_cpu > 1 else ("▼" if delta_cpu < -1 else "—")
                        col   = "#ff6b35" if delta_cpu > 5 else ("#7fff6e" if delta_cpu < -5 else "#ffd166")
                        st.markdown(f"""
                        <div class="pred-card">
                          <div class="pred-label">CPU at t+5 min (Stage 1)</div>
                          <div class="pred-value" style="color:{col}">{pred_cpu:.1f}%</div>
                          <div class="pred-sub">{arrow} {abs(delta_cpu):.1f}% vs now</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="pred-card"><div class="pred-label">CPU at t+5 min</div>'
                                    '<div class="pred-value" style="color:#4a7a9b">—</div>'
                                    '<div class="pred-sub">stage1_model.pkl not loaded</div></div>',
                                    unsafe_allow_html=True)

                with pc3:
                    st.markdown(f"""
                    <div class="pred-card">
                      <div class="pred-label">Energy Now (linear model)</div>
                      <div class="pred-value" style="color:#7fff6e">{base_energy:.1f} W</div>
                      <div class="pred-sub">P_idle + ΔP × CPU%</div>
                    </div>""", unsafe_allow_html=True)

                with pc4:
                    if pred_energy is not None:
                        delta_e = pred_energy - base_energy
                        ecol    = "#ff6b35" if delta_e > 10 else ("#7fff6e" if delta_e < -10 else "#ffd166")
                        st.markdown(f"""
                        <div class="pred-card">
                          <div class="pred-label">Energy at t+5 min (Stage 2)</div>
                          <div class="pred-value" style="color:{ecol}">{pred_energy:.1f} W</div>
                          <div class="pred-sub">{'+' if delta_e >= 0 else ''}{delta_e:.1f} W vs now</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="pred-card"><div class="pred-label">Energy at t+5 min</div>'
                                    '<div class="pred-value" style="color:#4a7a9b">—</div>'
                                    '<div class="pred-sub">stage2_model.pkl not loaded</div></div>',
                                    unsafe_allow_html=True)

                # ── prediction trend chart ──
                if len(ph) >= 2:
                    df_ph = pd.DataFrame(ph)
                    df_ph["time"] = pd.to_datetime(df_ph["ts"], unit="s")

                    fig_pred = make_subplots(rows=1, cols=2,
                                             subplot_titles=("CPU% — actual vs predicted t+5",
                                                             "Energy (W) — actual vs predicted t+5"))
                    fig_pred.add_trace(go.Scatter(x=df_ph["time"], y=df_ph["cpu_actual"],
                                                  name="CPU actual", line=dict(color="#00d4ff", width=2)),
                                       row=1, col=1)
                    if df_ph["cpu_pred"].notna().any():
                        fig_pred.add_trace(go.Scatter(x=df_ph["time"], y=df_ph["cpu_pred"],
                                                      name="CPU predicted", line=dict(color="#ffd166", width=2, dash="dot")),
                                           row=1, col=1)
                    fig_pred.add_trace(go.Scatter(x=df_ph["time"], y=df_ph["energy_actual"],
                                                  name="Energy actual", line=dict(color="#7fff6e", width=2)),
                                       row=1, col=2)
                    if df_ph["energy_pred"].notna().any():
                        fig_pred.add_trace(go.Scatter(x=df_ph["time"], y=df_ph["energy_pred"],
                                                      name="Energy predicted", line=dict(color="#ff6b35", width=2, dash="dot")),
                                           row=1, col=2)

                    fig_pred.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(4,13,20,1)",
                        font=dict(color="#4a7a9b"), height=260,
                        margin=dict(l=0, r=0, t=36, b=0),
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                    )
                    fig_pred.update_xaxes(gridcolor="#0f2133")
                    fig_pred.update_yaxes(gridcolor="#0f2133")
                    st.plotly_chart(fig_pred, use_container_width=True)

                st.markdown("---")

        # ── fleet-wide prediction summary table ──
        if any(st.session_state.pred_history.get(v.name) for v in running_pred):
            st.markdown("#### Fleet Prediction Summary")
            summary_rows = []
            for vm in running_pred:
                ph = st.session_state.pred_history.get(vm.name, [])
                if not ph:
                    continue
                last = ph[-1]
                summary_rows.append({
                    "VM":               vm.name,
                    "CPU now (%)":      round(last["cpu_actual"], 1),
                    "CPU t+5 pred (%)": round(last["cpu_pred"], 1) if last["cpu_pred"] is not None else None,
                    "Energy now (W)":   round(last["energy_actual"], 1),
                    "Energy t+5 (W)":   round(last["energy_pred"], 1) if last["energy_pred"] is not None else None,
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ENERGY MODEL
# ══════════════════════════════════════════════════════════════════════════════

with tab_energy:
    st.markdown("#### Linear Power Model &nbsp; `P(W) = P_idle + (P_max − P_idle) × CPU%`")
    st.caption(f"P_idle = {POWER_IDLE:.0f} W &nbsp;|&nbsp; P_max = {POWER_MAX:.0f} W  "
               f"— these constants must match the derivation used to create your Stage 2 training labels.")

    cpu_vals   = list(range(0, 101, 5))
    power_vals = [cpu_to_watts(c) for c in cpu_vals]
    fig_pow = go.Figure()
    fig_pow.add_trace(go.Scatter(
        x=cpu_vals, y=power_vals, mode="lines+markers",
        line=dict(color="#00d4ff", width=3), marker=dict(color="#7fff6e", size=7),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
    ))
    fig_pow.update_layout(
        xaxis_title="CPU Utilisation (%)", yaxis_title="Power (W)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(4,13,20,1)",
        font=dict(color="#4a7a9b"),
        xaxis=dict(gridcolor="#0f2133"), yaxis=dict(gridcolor="#0f2133"),
        height=280, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_pow, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Fleet Energy Budget")

    vms_e     = ctrl.list_vms()
    running_e = [v for v in vms_e if v.state.lower() == "running"]
    e_rows    = []
    total_w   = 0.0

    for vm in running_e:
        m   = collector.latest(vm.name)
        cpu = m.get("cpu", 0.0) if m else 0.0
        w   = cpu_to_watts(cpu)
        total_w += w

        # also show predicted energy if available
        ph      = st.session_state.pred_history.get(vm.name, [])
        pred_w  = ph[-1]["energy_pred"] if ph and ph[-1]["energy_pred"] is not None else None
        e_rows.append({
            "VM":             vm.name,
            "CPU %":          round(cpu, 1),
            "Power now (W)":  round(w, 1),
            "Pred t+5 (W)":  round(pred_w, 1) if pred_w else "—",
        })

    if e_rows:
        st.dataframe(pd.DataFrame(e_rows), use_container_width=True, hide_index=True)

        df_e = pd.DataFrame([r for r in e_rows if isinstance(r["Pred t+5 (W)"], float)])
        if not df_e.empty:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name="Now", x=df_e["VM"], y=df_e["Power now (W)"],
                                     marker_color="#00d4ff"))
            fig_cmp.add_trace(go.Bar(name="Predicted t+5", x=df_e["VM"], y=df_e["Pred t+5 (W)"],
                                     marker_color="#ff6b35"))
            fig_cmp.update_layout(
                barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(4,13,20,1)", font=dict(color="#4a7a9b"),
                xaxis=dict(gridcolor="#0f2133"), yaxis=dict(gridcolor="#0f2133"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                height=280, margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Fleet Power Now",    f"{total_w:.0f} W")
        ec2.metric("Hourly consumption", f"{watts_to_kwh(total_w, 1):.3f} kWh")
        ec3.metric("Daily (est.)",       f"{watts_to_kwh(total_w, 24):.2f} kWh")
    else:
        st.info("No running VMs with metrics yet.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

with tab_pipeline:
    st.markdown("#### Two-Stage Chained Prediction Pipeline")

    st.markdown("""
<div style="display:grid; grid-template-columns:1fr 48px 1fr; gap:0; align-items:center; margin:24px 0;">

  <div style="background:#0a1a26; border:1px solid rgba(0,212,255,0.3); border-radius:12px; padding:20px;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:.7rem; color:#4a7a9b; letter-spacing:.15em;">STAGE 1</div>
    <div style="font-size:1.15rem; font-weight:800; color:#00d4ff; margin:6px 0;">Workload Forecasting</div>
    <div style="font-size:.8rem; color:#7fff6e; margin-bottom:10px;">Target: CPU usage [%] at t+5 min</div>
    <div style="font-size:.75rem; color:#4a7a9b; line-height:1.8;">
      Input: Memory util ratio · Disk I/O total · Network total<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Lag features (t-1,t-2,t-3) · Rolling mean/std · Time encodings · VM_ID<br>
      Scaler: <strong style="color:#e8f4fd">StandardScaler</strong><br>
      Model: <strong style="color:#e8f4fd">RandomForestRegressor</strong>
      (best of RandomizedSearchCV, 5-fold TSCV)<br>
      Output: <strong style="color:#ffd166">ŷ_CPU — predicted CPU%</strong>
    </div>
  </div>

  <div style="text-align:center; font-size:2rem; color:#00d4ff; padding:0 8px;">→</div>

  <div style="background:#0a1a26; border:1px solid rgba(127,255,110,0.3); border-radius:12px; padding:20px;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:.7rem; color:#4a7a9b; letter-spacing:.15em;">STAGE 2</div>
    <div style="font-size:1.15rem; font-weight:800; color:#7fff6e; margin:6px 0;">Energy Prediction</div>
    <div style="font-size:.8rem; color:#ffd166; margin-bottom:10px;">Target: Derived Energy (W) at t+5 min</div>
    <div style="font-size:.75rem; color:#4a7a9b; line-height:1.8;">
      Input: All Stage 1 features (scaled) <strong style="color:#00d4ff">+ ŷ_CPU appended</strong><br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i.e. np.hstack([X_scaled, cpu_pred.reshape(-1,1)])<br>
      Scaler: same <strong style="color:#e8f4fd">StandardScaler</strong> as Stage 1<br>
      Model: <strong style="color:#e8f4fd">LGBMRegressor</strong> (oracle variant — trained on true CPU)<br>
      Output: <strong style="color:#ffd166">ŷ_Energy — predicted watts</strong>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Loaded Model Registry")

    reg_rows = []
    for key, label, fname in [
        ("stage1", "Stage 1 — RandomForestRegressor", "stage1_model.pkl"),
        ("stage2", "Stage 2 — LGBMRegressor (oracle)", "stage2_model.pkl"),
        ("scaler", "Feature Scaler — StandardScaler", "scaler.pkl"),
    ]:
        obj = models[key]
        if obj is not None:
            extras = ""
            if hasattr(obj, "n_estimators"):
                extras = f"n_estimators={obj.n_estimators}"
            elif hasattr(obj, "n_features_in_"):
                extras = f"n_features_in={obj.n_features_in_}"
            reg_rows.append({
                "File":   fname,
                "Object": label,
                "Type":   type(obj).__name__,
                "Status": "✅ Loaded",
                "Detail": extras,
            })
        else:
            reg_rows.append({
                "File":   fname,
                "Object": label,
                "Type":   "—",
                "Status": "❌ Not found",
                "Detail": f"Place {fname} in project root",
            })

    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    # feature mismatch warning
    if models["scaler"] is not None and models["stage1"] is not None:
        scaler_n = getattr(models["scaler"], "n_features_in_", None)
        model_n  = getattr(models["stage1"], "n_features_in_", None)
        if scaler_n and model_n and scaler_n != model_n:
            st.error(
                f"⚠️ Feature count mismatch: scaler expects {scaler_n} features, "
                f"Stage 1 model expects {model_n}. Check that scaler.pkl and stage1_model.pkl "
                f"were exported from the same training run."
            )

    st.markdown("---")
    st.markdown("#### Node Roles & Resource Budget")
    role_data = [
        {"Node Role": "Load Balancer", "Count": 1, "vCPUs": 1, "Memory (GB)": 1,  "Disk (GB)": 5,  "Workload Class": "Balanced"},
        {"Node Role": "Compute Node",  "Count": 2, "vCPUs": 2, "Memory (GB)": 4,  "Disk (GB)": 10, "Workload Class": "Compute-intensive"},
        {"Node Role": "Data Node",     "Count": 2, "vCPUs": 1, "Memory (GB)": 2,  "Disk (GB)": 20, "Workload Class": "Data-intensive"},
    ]
    st.dataframe(pd.DataFrame(role_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Export Checklist")
    st.markdown("""
Add this export cell to the end of your training notebook to export everything the dashboard needs:

```python
import pickle

# Stage 1 — best RandomForest from RandomizedSearchCV
with open("stage1_model.pkl", "wb") as f:
    pickle.dump(best_rf_model_stage1, f)

# Stage 2 — LightGBM oracle variant
with open("stage2_model.pkl", "wb") as f:
    pickle.dump(lgbm_model_stage2_oracle, f)

# StandardScaler fitted on X_train — REQUIRED for live inference
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler_standard, f)   # replace with your scaler variable name

# Optional but useful: keep explicit names for both scalers
with open("scaler_standard.pkl", "wb") as f:
    pickle.dump(scaler_standard, f)

with open("scaler_minmax.pkl", "wb") as f:
    pickle.dump(scaler_minmax, f)
```

Then move the exported `.pkl` files into the project root folder alongside `dashboard.py`.
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ACTIVITY LOG
# ══════════════════════════════════════════════════════════════════════════════

with tab_logs:
    st.markdown("#### Activity Log")
    if st.button("🗑 Clear log"):
        st.session_state.logs = []
    log_text = "\n".join(reversed(st.session_state.logs)) if st.session_state.logs else "(empty)"
    st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH
# ══════════════════════════════════════════════════════════════════════════════

if auto_refresh:
    now = time.time()
    if now - st.session_state.last_refresh > 15:
        st.session_state.last_refresh = now
        time.sleep(0.3)
        st.rerun()
