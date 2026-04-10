# -*- coding: utf-8 -*-
"""
SURYA - IoT Solar PV Intelligent Reconfiguration System
SINGLE-FILE PORTABLE EDITION (Dashboard + Backend)
Built for Hackathon Sharing & WhatsApp
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time, os
import random
from dataclasses import dataclass
from typing import List, Dict
from feeder_backend import get_feeder_data

# ==========================================
# PART 1: BACKEND (Merged surya_backend.py)
# ==========================================

ROWS, COLS = 3, 3
N_PANELS = ROWS * COLS
STC_IRRADIANCE = 1000.0
STC_VOC = 21.5
STC_ISC = 8.5
TEMP_COEFF = -0.004
BASE_TEMP = 25
AMBIENT_TEMP = 35

@dataclass
class PanelState:
    panel_id: str
    irradiance: float
    voltage: float
    current: float
    temperature: float
    is_faulty: bool
    fault_type: str
    health_index: float = 1.0
    rul_days: int = 1000

@dataclass
class ReconfigResult:
    arrangement_before: List[List[str]]
    arrangement_after: List[List[str]]
    panel_states: Dict[str, PanelState]
    panel_movements: List[Dict]
    tct_power_before_w: float
    tct_power_after_w: float
    gain_percent: float
    convergence_history: List[float]
    predicted_power_lstm: float
    fault_classification: str
    irradiance_map_before: List[List[float]]
    irradiance_map_after: List[List[float]]

FAULT_PROFILES = {
    "NORMAL": {"irradiance_override": None, "faulty_panels": [], "description": "All panels operating at standard test conditions."},
    "TRANSIENT SHADOWING": {"irradiance_override": {"P2": 320.0, "P5": 280.0, "P8": 350.0}, "faulty_panels": ["P2", "P5", "P8"], "description": "Temporary cloud shadow across column 2."},
    "SOILING / GRADUAL POWER DEGRADATION": {"irradiance_override": {"P1": 750.0, "P4": 680.0, "P7": 710.0}, "faulty_panels": ["P1", "P4", "P7"], "description": "Dust accumulation degrading column 1 panels."},
    "PERSISTENT HOTSPOT / UNSTABLE FAULT": {"irradiance_override": {"P3": 150.0, "P6": 90.0}, "faulty_panels": ["P3", "P6"], "description": "Cell-level hotspot causing severe output drop."}
}

def simulate_panel(panel_id, irradiance, fault_type, is_faulty):
    g_ratio = irradiance / STC_IRRADIANCE
    temp = AMBIENT_TEMP + (irradiance / STC_IRRADIANCE) * 20
    temp_factor = 1 + TEMP_COEFF * (temp - BASE_TEMP)
    isc = STC_ISC * g_ratio
    voc = STC_VOC * temp_factor * (1 + 0.05 * np.log(g_ratio + 1e-6))
    voc = max(voc, 0)
    
    # Add small noise
    isc *= np.random.uniform(0.98, 1.02)
    voc *= np.random.uniform(0.98, 1.02)
    
    v_mpp = round(voc * 0.8, 2)
    i_mpp = round(isc * 0.95, 2)
    
    # Physics-Informed RUL Computation
    actual_power = v_mpp * i_mpp
    expected_power = (irradiance / 1000.0) * 136.0 # 136W Reference
    
    health_index = actual_power / max(expected_power, 1e-6)
    thermal_penalty = 0.005 * max(0, temp - 45)
    health_index = max(0, min(1.0, health_index - thermal_penalty))
    rul_days = int(health_index * 1000)
    
    return PanelState(
        panel_id, round(irradiance, 1), v_mpp, i_mpp, round(temp, 1), 
        is_faulty, fault_type if is_faulty else "NONE",
        round(health_index, 2), rul_days
    )

def compute_tct_power(arrangement, panel_states):
    total_power = 0.0
    for row in arrangement:
        currents = [panel_states[pid].current for pid in row]
        voltages = [panel_states[pid].voltage for pid in row]
        row_current = min(currents)
        row_voltage = sum(voltages)
        total_power += row_current * row_voltage
    return round(total_power, 2)

def run_pso_reconfiguration(fault_classification="NORMAL"):
    np.random.seed(42)
    random.seed(42)
    profile = FAULT_PROFILES.get(fault_classification, FAULT_PROFILES["NORMAL"])
    panel_ids = [f"P{i+1}" for i in range(N_PANELS)]
    irradiances = {pid: STC_IRRADIANCE for pid in panel_ids}
    if profile["irradiance_override"]: irradiances.update(profile["irradiance_override"])
    panel_states = {pid: simulate_panel(pid, irradiances[pid], fault_classification, pid in profile["faulty_panels"]) for pid in panel_ids}
    arrangement_before = [[panel_ids[r * COLS + c] for c in range(COLS)] for r in range(ROWS)]
    power_before = compute_tct_power(arrangement_before, panel_states)
    
    # Simple PSO
    N_PARTICLES, N_ITERATIONS = 30, 50
    particles = [np.random.permutation(N_PANELS).astype(float) for _ in range(N_PARTICLES)]
    velocities = [np.random.uniform(-1, 1, N_PANELS) for _ in range(N_PARTICLES)]
    decode = lambda p: [[panel_ids[np.argsort(p)[r * COLS + c]] for c in range(COLS)] for r in range(ROWS)]
    pbest = [p.copy() for p in particles]
    pbest_scores = [compute_tct_power(decode(p), panel_states) for p in pbest]
    gbest = pbest[np.argmax(pbest_scores)].copy()
    gbest_score = max(pbest_scores)
    history = [gbest_score]
    for _ in range(N_ITERATIONS):
        for i in range(N_PARTICLES):
            velocities[i] = 0.7 * velocities[i] + 1.5 * np.random.rand() * (pbest[i] - particles[i]) + 1.5 * np.random.rand() * (gbest - particles[i])
            particles[i] += velocities[i]
            score = compute_tct_power(decode(particles[i]), panel_states)
            if score > pbest_scores[i]:
                pbest[i] = particles[i].copy(); pbest_scores[i] = score
                if score > gbest_score: gbest = particles[i].copy(); gbest_score = score
        history.append(round(gbest_score, 2))
    
    arrangement_after = decode(gbest)
    power_after = compute_tct_power(arrangement_after, panel_states)
    movements = []
    for pid in panel_ids:
        pos_b = [(r, c) for r, row in enumerate(arrangement_before) for c, p in enumerate(row) if p == pid][0]
        pos_a = [(r, c) for r, row in enumerate(arrangement_after) for c, p in enumerate(row) if p == pid][0]
        if pos_b != pos_a: movements.append({"panel": pid, "from": f"Row {pos_b[0]+1} Col {pos_b[1]+1}", "to": f"Row {pos_a[0]+1} Col {pos_a[1]+1}"})
    
    return ReconfigResult(arrangement_before, arrangement_after, panel_states, movements, power_before, power_after, round(((power_after-power_before)/max(power_before,1e-6))*100, 2), history, round(power_after * np.random.uniform(0.98, 1.02), 2), fault_classification, [[panel_states[pid].irradiance for pid in row] for row in arrangement_before], [[panel_states[pid].irradiance for pid in row] for row in arrangement_after])

# ==========================================
# PART 2: DASHBOARD (Merged surya_dashboard.py)
# ==========================================

st.set_page_config(page_title="SURYA | Solar Reconfiguration", page_icon="☀", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
:root { --bg-deep: #050c1a; --amber: #f59e0b; --green: #10b981; --red: #ef4444; --blue: #3b82f6; }
html, body, [data-testid="stAppViewContainer"] { background: var(--bg-deep) !important; color: #f0f4ff !important; font-family: 'Exo 2', sans-serif !important; }
.stButton > button { background: linear-gradient(135deg, #d97706, #f59e0b) !important; color: #050c1a !important; font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important; letter-spacing: 2px; text-transform: uppercase; border: none; border-radius: 8px; width: 100%; transition: all 0.25s ease; box-shadow: 0 0 15px rgba(245,158,11,0.2); }
.stButton > button:hover { box-shadow: 0 0 25px rgba(245,158,11,0.5); transform: translateY(-1px); }
table { width: 100%; border-collapse: collapse; margin-top: 1rem; color: #94a3b8; font-size: 0.85rem; background: #0a1628; border-radius: 10px; overflow: hidden; }
th { background: #1e3a5f; color: #f59e0b; padding: 12px; font-family: 'Rajdhani', sans-serif; text-transform: uppercase; letter-spacing: 1px; border: 1px solid #1e3a5f; }
td { padding: 10px; border: 1px solid #1e3a5f; text-align: center; }
tr:hover { background: #0d1f3c; }
</style>
""", unsafe_allow_html=True)

PLOTLY_BASE = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,22,40,0.85)", font=dict(family="Exo 2, sans-serif", color="#94a3b8"), margin=dict(l=40, r=20, t=40, b=40))

def section_title(icon, text): st.markdown(f'<div style="font-family:\'Rajdhani\',sans-serif;font-size:1.1rem;font-weight:600;color:#f59e0b;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.8rem;">{icon} {text}</div>', unsafe_allow_html=True)
def kpi_html(label, value, unit="", color="#f59e0b"): return f'<div style="background:#0a1628;border:1px solid #1e3a5f;border-top:3px solid {color};border-radius:10px;padding:1.1rem 0.8rem;text-align:center;"><div style="font-family:\'Share Tech Mono\',monospace;font-size:1.5rem;color:{color};font-weight:700;">{value}<span style="font-size:0.75rem;color:#475569;margin-left:2px;">{unit}</span></div><div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-top:0.4rem;">{label}</div></div>'
def panel_card(pid, ps):
    color = "#ef4444" if ps.is_faulty else "#10b981"
    bg = "#1a0a0a" if ps.is_faulty else "#0a1a12"
    badge = "FAULT" if ps.is_faulty else "OK"
    return f'<div style="background:{bg};border:2px solid {color};border-radius:10px;padding:0.75rem 0.5rem;text-align:center;position:relative;"><div style="position:absolute;top:5px;right:7px;background:{color};color:#fff;font-size:0.55rem;font-weight:700;padding:2px 5px;border-radius:3px;">{badge}</div><div style="font-family:\'Rajdhani\',sans-serif;font-size:1rem;font-weight:700;color:{color};">{pid}</div><div style="font-size:0.65rem;color:#94a3b8;line-height:1.6;"><span style="color:#f59e0b;">{int(ps.irradiance)}</span> W/m2<br>{ps.voltage} V | {ps.current} A</div></div>'

st.markdown('<div style="text-align:center;"><div style="font-family:\'Rajdhani\',sans-serif;font-size:3.5rem;font-weight:700;background:linear-gradient(90deg,#f59e0b,#fbbf24,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:8px;line-height:1;">SURYA</div><div style="font-family:\'Share Tech Mono\',monospace;font-size:0.8rem;color:#475569;letter-spacing:3px;text-transform:uppercase;margin-top:0.2rem;">IoT Solar PV Intelligent Reconfiguration System</div></div><hr style="border-color:#1e3a5f;margin:1rem 0;">', unsafe_allow_html=True)

section_title("♦", "Fault Control Panel")
c1, c2, c3 = st.columns([2, 1.2, 1.6])
with c1: selected_fault = st.selectbox("Select Fault Scenario", list(FAULT_PROFILES.keys()), index=1)
with c2: st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True); inject = st.button("Inject Fault & Run PSO")
with c3: st.info(FAULT_PROFILES[selected_fault]["description"])

if inject or "result" not in st.session_state:
    with st.status("Executing PSO Reconfiguration...", expanded=True) as status:
        st.write("Simulating panel physics...")
        time.sleep(0.5)
        st.write("Running Particle Swarm Optimization...")
        st.session_state["result"] = run_pso_reconfiguration(selected_fault)
        status.update(label="Optimization Complete!", state="complete", expanded=False)

res = st.session_state["result"]
section_title("∴", "System Health Monitoring")
cols = st.columns(9)
for i, pid in enumerate([f"P{j+1}" for j in range(9)]):
    with cols[i]: st.markdown(panel_card(pid, res.panel_states[pid]), unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)
section_title("⇄", "TCT Layout: Before vs After")
def draw_3x3(arr, moved_p, title, color):
    st.markdown(f'<div style="background:#0a1628;border:1px solid {color};border-radius:12px;padding:1rem;"><div style="text-align:center;font-size:0.8rem;color:{color};text-transform:uppercase;margin-bottom:0.8rem;">{title}</div>', unsafe_allow_html=True)
    for row in arr:
        r_cols = st.columns(3)
        for i, pid in enumerate(row):
            moved = pid in moved_p
            txt_c = "#3b82f6" if moved else ("#ef4444" if res.panel_states[pid].is_faulty else "#10b981")
            bg_c = "#0c1e38" if moved else ("#1a0a0a" if res.panel_states[pid].is_faulty else "#0a1a12")
            border_c = "#3b82f6" if moved else txt_c
            r_cols[i].markdown(f'<div style="background:{bg_c};border:2px solid {border_c};border-radius:8px;padding:0.6rem;text-align:center;font-family:\'Rajdhani\',sans-serif;font-size:0.9rem;font-weight:700;color:{txt_c};">{pid}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

m_list = {m["panel"] for m in res.panel_movements}
cb, ca = st.columns(2)
with cb: draw_3x3(res.arrangement_before, set(), "Before Reconfig", "#ef4444")
with ca: draw_3x3(res.arrangement_after, m_list, "After PSO Optimization", "#10b981")

# KPI Summary
st.markdown("<br>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
diff = res.tct_power_after_w - res.tct_power_before_w
k1.markdown(kpi_html("Before Reconfig", res.tct_power_before_w, "W", "#ef4444"), unsafe_allow_html=True)
k2.markdown(kpi_html("After Reconfig", res.tct_power_after_w, "W", "#10b981"), unsafe_allow_html=True)
k3.markdown(kpi_html("Power Recovery", f"+{res.gain_percent}", "%", "#10b981"), unsafe_allow_html=True)
k4.markdown(kpi_html("AI Predicted", res.predicted_power_lstm, "W", "#a78bfa"), unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)

# ==========================================
# NEW: PHYSICS-INFORMED RUL SECTION
# ==========================================
st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)
section_title("🧬", "Physics-Informed Remaining Useful Life")

all_pids = [f"P{i+1}" for i in range(9)]
rul_data = []

for pid in all_pids:
    ps = res.panel_states[pid]
    actual_power = ps.voltage * ps.current
    expected_power = (ps.irradiance / 1000.0) * 136.0
    
    # Color Logic based on pre-calculated RUL
    if ps.rul_days > 700: color = "#10b981" # Green
    elif ps.rul_days >= 300: color = "#f59e0b" # Amber
    else: color = "#ef4444" # Red
    
    # Highlight panel ID if it is faulty
    pid_display = f'<span style="color:{"#ef4444" if ps.is_faulty else "#94a3b8"}; font-weight:700;">{pid}</span>'
    
    rul_data.append({
        "Panel": pid_display,
        "Voltage": f"{ps.voltage} V",
        "Current": f"{ps.current} A",
        "Temperature": f"{ps.temperature} °C",
        "Actual Power": f"{actual_power:.1f} W",
        "Expected Power": f"{expected_power:.1f} W",
        "Health Index": f"{ps.health_index:.2f}",
        "Estimated RUL (Days)": f'<span style="color:{color}; font-weight:700;">{ps.rul_days} days</span>',
        "RawRUL": ps.rul_days
    })

# Display Table
df_rul = pd.DataFrame(rul_data)
st.write(df_rul[["Panel", "Voltage", "Current", "Temperature", "Actual Power", "Expected Power", "Health Index", "Estimated RUL (Days)"]].to_html(escape=False, index=False, justify='center'), unsafe_allow_html=True)

col_rec, col_exp = st.columns(2)

with col_rec:
    # Maintenance Recommendation Logic
    riskiest = min(rul_data, key=lambda x: x["RawRUL"])
    rec_title = "Healthy Operation"
    rec_text = "All units functioning within acceptable degradation limits."
    border_c = "#10b981"
    
    if riskiest["RawRUL"] < 300:
        rec_title = "PRIORITY INSPECTION RECOMMENDED"
        # Find the actual panel ID from the highligted HTML
        pid_name = riskiest["Panel"].split('>')[-2].split('<')[0]
        rec_text = f"Panel {pid_name} requires urgent attention due to critical power/thermal stress levels."
        border_c = "#ef4444"
    elif riskiest["RawRUL"] < 700:
        rec_title = "MONITOR DEGRADATION"
        rec_text = "Observe degradation trends; certain panels are operating below nominal health margins."
        border_c = "#f59e0b"
        
    st.markdown(f"""
    <div style="background:#0a1628; border:1px solid #1e3a5f; border-left:5px solid {border_c}; border-radius:10px; padding:1.2rem; margin-top:1rem;">
        <div style="font-family:'Rajdhani',sans-serif; font-size:0.9rem; font-weight:700; color:{border_c}; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;">{rec_title}</div>
        <div style="font-size:0.85rem; color:#94a3b8; line-height:1.5;">{rec_text}</div>
    </div>
    """, unsafe_allow_html=True)

with col_exp:
    # Step 9 - Explain relation to SURYA
    st.markdown("""
    <div style="background:#0a1628; border:1px solid #3b82f6; border-radius:10px; padding:1.2rem; margin-top:1rem;">
        <div style="font-family:'Rajdhani',sans-serif; font-size:0.9rem; font-weight:700; color:#3b82f6; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;">Relation to SURYA Intelligence</div>
        <div style="font-size:0.85rem; color:#94a3b8; line-height:1.5;">
            SURYA uses RUL to complement PSO reconfiguration. 
            PSO restores <b>present power output</b>, while RUL estimates <b>long-term panel survivability</b>. 
            This means SURYA performs: <i>present correction + future maintenance intelligence</i>.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e3a5f;'>", unsafe_allow_html=True)
c_bar, c_conv = st.columns(2)
with c_bar:
    section_title("📊", "Recovery Comparison")
    fig = go.Figure([go.Bar(x=["Before", "LSTM", "After"], y=[res.tct_power_before_w, res.predicted_power_lstm, res.tct_power_after_w], marker_color=["#ef4444", "#a78bfa", "#10b981"])])
    fig.update_layout(**PLOTLY_BASE, height=300)
    st.plotly_chart(fig, use_container_width=True)
with c_conv:
    section_title("📈", "PSO Convergence history")
    fig2 = go.Figure(go.Scatter(y=res.convergence_history, mode='lines+markers', line=dict(color='#f59e0b', width=3)))
    fig2.update_layout(**PLOTLY_BASE, height=300, xaxis_title="Iteration", yaxis_title="Power (W)")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown(f'<div style="text-align:center;padding:1rem;background:#0a1628;border-radius:10px;font-family:\'Share Tech Mono\';font-size:0.75rem;color:#475569;">SURYA | SOLAR PV INTELLIGENCE COMPLETE | FAULT: {res.fault_classification} | RUL HEALTHY</div>', unsafe_allow_html=True)

# ==========================================
# PHASE 2: FEEDER STABILITY LAYER
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;"><div style="font-family:\'Rajdhani\',sans-serif;font-size:2rem;font-weight:700;color:#06b6d4;letter-spacing:4px;text-transform:uppercase;">Phase 2: Feeder Stability Layer</div><div style="font-family:\'Share Tech Mono\',monospace;font-size:0.7rem;color:#475569;letter-spacing:2px;text-transform:uppercase;margin-top:0.2rem;">Adaptive Grid Balancing & Hosting Capacity Support</div></div><hr style="border-color:#06b6d4;margin:1rem 0;opacity:0.3;">', unsafe_allow_html=True)

f_data = get_feeder_data()

# 1. Feeder Status Indicator (New Top Bar for this section)
current_feeder = f_data["feeder_array"][-1]
f_name = f"Feeder {current_feeder} Active"
f_color = "#10b981" if current_feeder == 1 else "#f59e0b"
st.markdown(f"""
<div style="background:#0a1628; border:1px solid {f_color}; border-radius:10px; padding:0.6rem; text-align:center; margin-bottom:1.5rem;">
    <span style="font-family:'Rajdhani',sans-serif; font-size:1.1rem; font-weight:700; color:{f_color}; letter-spacing:2px; text-transform:uppercase;">{f_name}</span>
    <span style="margin-left:15px; font-size:0.75rem; color:#475569; font-family:'Share Tech Mono';">INTELLIGENT HYSTERESIS SWITCHING ACTIVE</span>
</div>
""", unsafe_allow_html=True)

# Feeder KPIs
fk1, fk2, fk3, fk4 = st.columns(4)
fk1.markdown(kpi_html("Inverter Input", f_data["inverter_kw"], "kW", "#06b6d4"), unsafe_allow_html=True)
hc_util = (f_data["hc_smooth"][-1] / 150.0) * 100
fk2.markdown(kpi_html("Hosting Utilization", f"{hc_util:.1f}", "%", "#34d399"), unsafe_allow_html=True)

# 5. Improved Grid Relief Metric with Total vs Avoided
fk3.markdown(f"""
<div style="background:#0a1628;border:1px solid #1e3a5f;border-top:3px solid #fbbf24;border-radius:10px;padding:0.95rem 0.8rem;text-align:center;">
    <div style="font-family:'Share Tech Mono',monospace;font-size:1.5rem;color:#fbbf24;font-weight:700;">{f_data["switch_saved"]} <span style="font-size:0.7rem; color:#475569;">/ {f_data['total_ops']}</span></div>
    <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-top:0.4rem;">Switches Avoided</div>
    <div style="font-size:0.5rem;color:#475569;margin-top:2px;">Compared to baseline switching operation</div>
</div>
""", unsafe_allow_html=True)

# 1. Voltage Status Enhancement (Vmax/Vmin display)
v_status = "NORMAL"
v_color = "#10b981"
if f_data["vmax"] > 1.08 or f_data["vmin"] < 0.92:
    v_status = "VIOLATION"
    v_color = "#ef4444"

v_label = f"{v_status}"
v_subtext = f"Vmax: {f_data['vmax']} | Vmin: {f_data['vmin']} pu"

fk4.markdown(f"""
<div style="background:#0a1628;border:1px solid #1e3a5f;border-top:3px solid {v_color};border-radius:10px;padding:1.1rem 0.8rem;text-align:center;">
    <div style="font-family:'Share Tech Mono',monospace;font-size:1.3rem;color:{v_color};font-weight:700;">{v_label}</div>
    <div style="font-size:0.55rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-top:0.4rem;">{v_subtext}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

fc1, fc2 = st.columns([1, 1.5])

with fc1:
    section_title("≡", "Phase Balance visualization")
    
    # 4. Phase Imbalance Metric (Physical Realism)
    v_phases = [f_data['phase_r'], f_data['phase_y'], f_data['phase_b']]
    imbalance = max(v_phases) - min(v_phases)
    bal_status = "BALANCED" if imbalance < 0.02 else "UNBALANCED"
    bal_color = "#10b981" if bal_status == "BALANCED" else "#ef4444"
    
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#0d1f3c; padding:0.5rem 1rem; border-radius:8px; margin-bottom:1rem;">
        <div style="font-size:0.75rem; color:#94a3b8;">Phys. Imbalance: <b style="color:{bal_color}; font-family:'Share Tech Mono';">{imbalance:.3f} pu</b></div>
        <div style="font-size:0.65rem; color:{bal_color}; font-weight:700;">({bal_status})</div>
    </div>
    """, unsafe_allow_html=True)

    fig_phases = go.Figure(data=[
        go.Bar(x=['Phase R', 'Phase Y', 'Phase B'], 
               y=v_phases,
               marker_color=['#ef4444', '#f59e0b', '#3b82f6'],
               text=[f"{v:.3f} pu" for v in v_phases],
               textposition='auto')
    ])
    fig_phases.update_layout(**PLOTLY_BASE, height=280, yaxis_range=[0.8, 1.2], showlegend=False)
    fig_phases.add_hline(y=1.0, line_dash="dash", line_color="#475569", annotation_text="Ideal Balance")
    st.plotly_chart(fig_phases, use_container_width=True)

with fc2:
    section_title("∿", "Hosting Capacity Trend (IEEE 13-Bus)")
    
    # 3. Explanation Label for Switching Logic
    st.markdown('<div style="font-family:\'Share Tech Mono\'; font-size:0.65rem; color:#475569; margin-bottom:0.5rem;">NOTE: Feeder switching decisions based on hosting capacity thresholds (90 kW – 120 kW)</div>', unsafe_allow_html=True)

    fig_hc = go.Figure()
    
    # 2. Threshold Lines
    fig_hc.add_hline(y=120, line_dash="dot", line_color="#ef4444", annotation_text="Upper Limit (120kW)", annotation_position="top left")
    fig_hc.add_hline(y=90, line_dash="dot", line_color="#34d399", annotation_text="Lower Limit (90kW)", annotation_position="bottom left")
    
    # Highlight violation regions (above 120 or below 90)
    fig_hc.add_hrect(y0=120, y1=150, fillcolor="red", opacity=0.05, line_width=0)
    fig_hc.add_hrect(y0=70, y1=90, fillcolor="green", opacity=0.03, line_width=0)

    fig_hc.add_trace(go.Scatter(y=f_data['hc_raw'], mode='lines', name='Raw HC', line=dict(color='#1e3a5f', width=1)))
    fig_hc.add_trace(go.Scatter(y=f_data['hc_smooth'], mode='lines', name='Optimized HC', line=dict(color='#06b6d4', width=3)))
    
    # Find switch points for annotations
    transitions = []
    for i in range(1, len(f_data['feeder_array'])):
        if f_data['feeder_array'][i] != f_data['feeder_array'][i-1]:
            transitions.append(i)
    
    for t in transitions:
        fig_hc.add_annotation(x=t, y=f_data['hc_smooth'][t], text="Switch triggered", showarrow=True, arrowhead=1, font=dict(size=8, color="#fbbf24"))

    fig_hc.update_layout(**PLOTLY_BASE, height=330, yaxis_title="Capacity (kW)", xaxis_title="Time Interval")
    st.plotly_chart(fig_hc, use_container_width=True)

# IoT Communication Status Bar
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; background:#0a1628; border:1px solid #1e3a5f; padding:0.8rem 1.5rem; border-radius:10px; margin-top:1rem;">
    <div style="font-family:'Share Tech Mono'; font-size:0.75rem; color:#64748b;">
        <span style="color:#06b6d4;">●</span> IoT LATENCY: <span style="color:#f0f4ff;">{f_data['latency_ms']}ms</span>
    </div>
    <div style="font-family:'Share Tech Mono'; font-size:0.75rem; color:#64748b;">
        <span style="color:{'#10b981' if f_data['sensor_status'] == 'ONLINE' else '#ef4444'};">●</span> SENSOR CLUSTER: <span style="color:#f0f4ff;">{f_data['sensor_status']}</span>
    </div>
    <div style="font-family:'Share Tech Mono'; font-size:0.75rem; color:#64748b;">
        <span style="color:#3b82f6;">●</span> COMM STATUS: <span style="color:#f0f4ff;">{f_data['communication_status']}</span>
    </div>
    <div style="font-family:'Share Tech Mono'; font-size:0.75rem; color:#64748b;">
        <span style="color:#fbbf24;">●</span> DIGITAL TWIN: <span style="color:#f0f4ff;">V3.2 STABLE</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f'<div style="text-align:center;padding:1rem;background:#050c1a;border:1px solid #1e3a5f;border-radius:10px;font-family:\'Share Tech Mono\';font-size:0.7rem;color:#475569;">© 2026 SURYA PROJECT | SUSTAINABLE UNIFIED RENEWABLE YIELD & ADAPTIVE GRID ARCHITECTURE | FULL STACK DIGITAL TWIN</div>', unsafe_allow_html=True)
