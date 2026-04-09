# -*- coding: utf-8 -*-
"""
SURYA - IoT Solar PV Intelligent Reconfiguration Dashboard
Streamlit Digital Twin  |  Competition Presentation Mode
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time, os

# ─── Try real PSO first, fall back to simulation ──────────────────────────────
_USE_REAL_PSO = False
_PSO_ERROR    = ""

try:
    from pso import run_pso_reconfiguration as _real_pso
    _REQUIRED = ["AGNI_PILSTM_BEST.weights.h5", "x_scaler.pkl",
                 "y_scaler.pkl", "PSG_iTech_180W_TrainData.csv"]
    if all(os.path.exists(f) for f in _REQUIRED):
        _USE_REAL_PSO = True
    else:
        _PSO_ERROR = "Model files missing - using built-in simulation."
except Exception as _e:
    _PSO_ERROR = f"pso.py not loaded ({_e}) - using built-in simulation."

if not _USE_REAL_PSO:
    from surya_backend import run_pso_reconfiguration as _sim_pso

FAULT_OPTIONS = [
    "NORMAL",
    "TRANSIENT SHADOWING",
    "SOILING / GRADUAL POWER DEGRADATION",
    "PERSISTENT HOTSPOT / UNSTABLE FAULT",
]

FAULT_DESC = {
    "NORMAL":                                  "All panels at standard test conditions.",
    "TRANSIENT SHADOWING":                     "Temporary cloud shadow across selected panels.",
    "SOILING / GRADUAL POWER DEGRADATION":     "Dust accumulation - mild, gradual output drop.",
    "PERSISTENT HOTSPOT / UNSTABLE FAULT":     "Cell-level hotspot causing severe power loss.",
}

GRID_ROWS = GRID_COLS = 3


# ─── Unified result adapter ────────────────────────────────────────────────────
def run_surya(fault: str):
    if _USE_REAL_PSO:
        r = _real_pso(fault_classification=fault, verbose_pso=False)
        flat_b = r["arrangement_before"]
        flat_a = r["arrangement_after"]
        arr_before = [flat_b[i*GRID_COLS:(i+1)*GRID_COLS] for i in range(GRID_ROWS)]
        arr_after  = [flat_a[i*GRID_COLS:(i+1)*GRID_COLS] for i in range(GRID_ROWS)]

        ps = r["panel_states"]
        sys_irr = r.get("system_irradiance", 1000.0)
        panel_states_norm = {
            pid: {
                "irradiance":  v["irradiance"],
                "voltage":     round(v["V_oc"] * 0.80, 2),
                "current":     round(v["I_sc"] * 0.95, 2),
                "temperature": v["temperature"],
                "is_faulty":   v["irradiance"] < sys_irr * 0.85,
            }
            for pid, v in ps.items()
        }

        movements = [
            {
                "panel": m["panel"],
                "from": "Row {} Col {}".format(m["from_row"], m["from_col"]),
                "to":   "Row {} Col {}".format(m["to_row"],   m["to_col"]),
            }
            for m in r["panel_movements"]
            if (m["from_row"], m["from_col"]) != (m["to_row"], m["to_col"])
        ]

        irr_map_before = [[panel_states_norm[pid]["irradiance"] for pid in row] for row in arr_before]
        irr_map_after  = [[panel_states_norm[pid]["irradiance"] for pid in row] for row in arr_after]

        return {
            "arrangement_before":    arr_before,
            "arrangement_after":     arr_after,
            "panel_states":          panel_states_norm,
            "panel_movements":       movements,
            "tct_power_before_w":    r["tct_power_before_w"],
            "tct_power_after_w":     r["tct_power_after_w"],
            "gain_percent":          r["gain_percent"],
            "convergence_history":   r["convergence_history"],
            "predicted_power_lstm":  r["lstm_predicted_w"],
            "fault_classification":  fault,
            "irradiance_map_before": irr_map_before,
            "irradiance_map_after":  irr_map_after,
        }
    else:
        rc = _sim_pso(fault_classification=fault, verbose_pso=False)
        ps_norm = {
            pid: {
                "irradiance":  ps.irradiance,
                "voltage":     ps.voltage,
                "current":     ps.current,
                "temperature": ps.temperature,
                "is_faulty":   ps.is_faulty,
            }
            for pid, ps in rc.panel_states.items()
        }
        movements = [
            {"panel": m["panel"], "from": m["from"], "to": m["to"]}
            for m in rc.panel_movements
        ]
        return {
            "arrangement_before":    rc.arrangement_before,
            "arrangement_after":     rc.arrangement_after,
            "panel_states":          ps_norm,
            "panel_movements":       movements,
            "tct_power_before_w":    rc.tct_power_before_w,
            "tct_power_after_w":     rc.tct_power_after_w,
            "gain_percent":          rc.gain_percent,
            "convergence_history":   rc.convergence_history,
            "predicted_power_lstm":  rc.predicted_power_lstm,
            "fault_classification":  fault,
            "irradiance_map_before": rc.irradiance_map_before,
            "irradiance_map_after":  rc.irradiance_map_after,
        }


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SURYA | Solar Reconfiguration System",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --bg-deep:  #050c1a;
    --bg-card:  #0a1628;
    --bg-panel: #0d1f3c;
    --amber:    #f59e0b;
    --green:    #10b981;
    --red:      #ef4444;
    --blue:     #3b82f6;
    --border:   #1e3a5f;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    color: #f0f4ff !important;
    font-family: 'Exo 2', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 20% 0%,  rgba(245,158,11,0.06) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 100%, rgba(6,182,212,0.05)  0%, transparent 55%),
        #050c1a !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #1e3a5f !important;
}
.stSelectbox > div > div {
    background: #0d1f3c !important;
    border: 1px solid #1e3a5f !important;
    color: #f0f4ff !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #d97706, #f59e0b) !important;
    color: #050c1a !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 2rem !important;
    box-shadow: 0 0 20px rgba(245,158,11,0.3) !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 0 35px rgba(245,158,11,0.6) !important;
    transform: translateY(-1px) !important;
}
hr { border-color: #1e3a5f !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #050c1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,22,40,0.85)",
    font=dict(family="Exo 2, sans-serif", color="#94a3b8"),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def section_title(icon, text):
    st.markdown(
        '<div style="font-family:\'Rajdhani\',sans-serif;font-size:1.05rem;font-weight:600;'
        'color:#f59e0b;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.7rem;">'
        + icon + " " + text + "</div>",
        unsafe_allow_html=True,
    )


def kpi_html(label, value, unit="", color="#f59e0b", icon=""):
    return (
        '<div style="background:#0a1628;border:1px solid #1e3a5f;border-top:3px solid '
        + color + ';border-radius:10px;padding:1.1rem 0.8rem;text-align:center;">'
        + '<div style="font-size:1.1rem;margin-bottom:0.2rem;">' + icon + "</div>"
        + '<div style="font-family:\'Share Tech Mono\',monospace;font-size:1.55rem;color:'
        + color + ';font-weight:700;line-height:1.1;">' + value
        + '<span style="font-size:0.75rem;color:#475569;margin-left:2px;">' + unit + "</span></div>"
        + '<div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;'
        'letter-spacing:1.5px;margin-top:0.35rem;">' + label + "</div>"
        + "</div>"
    )


def panel_card(pid, irr, volt, curr, is_faulty, is_moved=False):
    if is_moved:
        border, bg, badge, glow = "#3b82f6", "#0c1e38", "MOVED", "0 0 14px rgba(59,130,246,0.45)"
    elif is_faulty:
        border, bg, badge, glow = "#ef4444", "#1a0a0a", "FAULT", "0 0 14px rgba(239,68,68,0.4)"
    else:
        border, bg, badge, glow = "#10b981", "#0a1a12", "OK",    "0 0 10px rgba(16,185,129,0.25)"
    return (
        '<div style="background:' + bg + ';border:2px solid ' + border
        + ';border-radius:10px;padding:0.75rem 0.5rem;text-align:center;'
        'box-shadow:' + glow + ';position:relative;">'
        + '<div style="position:absolute;top:5px;right:7px;background:' + border
        + ';color:#fff;font-size:0.55rem;font-family:\'Rajdhani\',sans-serif;font-weight:700;'
        'letter-spacing:1px;padding:2px 5px;border-radius:3px;">' + badge + "</div>"
        + '<div style="font-family:\'Rajdhani\',sans-serif;font-size:1.05rem;font-weight:700;'
        'color:' + border + ';margin-bottom:0.35rem;">' + pid + "</div>"
        + '<div style="font-size:0.7rem;color:#94a3b8;line-height:1.85;">'
        + '<span style="color:#f59e0b;">' + str(int(irr)) + "</span> W/m2<br>"
        + '<span style="color:#06b6d4;">' + str(round(volt, 1)) + "</span> V<br>"
        + '<span style="color:#a78bfa;">' + str(round(curr, 2)) + "</span> A"
        + "</div></div>"
    )


def grid_cell(pid, is_faulty, is_moved):
    if is_moved:
        bg, border, color = "#0c1e38", "#3b82f6", "#3b82f6"
    elif is_faulty:
        bg, border, color = "#1a0a0a", "#ef4444", "#ef4444"
    else:
        bg, border, color = "#0a1a12", "#10b981", "#10b981"
    return (
        '<div style="background:' + bg + ';border:2px solid ' + border
        + ';border-radius:8px;padding:0.65rem;text-align:center;'
        'font-family:\'Rajdhani\',sans-serif;font-size:0.95rem;font-weight:700;color:'
        + color + ';">' + pid + "</div>"
    )


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.4rem 0 0.4rem;">
    <div style="font-family:'Rajdhani',sans-serif;font-size:3rem;font-weight:700;
         background:linear-gradient(90deg,#f59e0b,#fbbf24,#06b6d4);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
         letter-spacing:6px;line-height:1.1;">SURYA</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;color:#475569;
         letter-spacing:3px;text-transform:uppercase;margin-top:0.25rem;">
        IoT Solar PV Intelligent Reconfiguration System &nbsp;|&nbsp; Digital Twin
    </div>
</div>
<hr style="border-color:#1e3a5f;margin:0.8rem 0;">
""", unsafe_allow_html=True)

# PSO source banner
if _USE_REAL_PSO:
    st.markdown(
        '<div style="background:#0a1a10;border:1px solid #10b981;border-radius:8px;'
        'padding:0.5rem 1rem;font-size:0.8rem;color:#10b981;margin-bottom:0.6rem;'
        'font-family:\'Share Tech Mono\',monospace;">'
        "LIVE | Using your trained AGNI PI-LSTM + real PSO engine (pso.py)"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="background:#1a1200;border:1px solid #f59e0b;border-radius:8px;'
        'padding:0.5rem 1rem;font-size:0.8rem;color:#f59e0b;margin-bottom:0.6rem;'
        'font-family:\'Share Tech Mono\',monospace;">'
        "SIM | " + _PSO_ERROR + " Place model files alongside this script to use real LSTM + PSO."
        "</div>",
        unsafe_allow_html=True,
    )

# ─── SECTION 1 — Fault Injection ──────────────────────────────────────────────
section_title("&diams;", "Fault Injection Control Panel")

c1, c2, c3 = st.columns([2, 1.2, 1.6])
with c1:
    selected_fault = st.selectbox("Select Fault Scenario", FAULT_OPTIONS,
                                  index=1, label_visibility="visible")
with c2:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    inject = st.button("Inject Fault & Run PSO")
with c3:
    st.markdown(
        '<div style="background:#0a1628;border:1px solid #1e3a5f;border-left:3px solid #f59e0b;'
        'border-radius:8px;padding:0.65rem 1rem;margin-top:6px;font-size:0.78rem;'
        'color:#94a3b8;font-family:\'Exo 2\',sans-serif;">'
        + FAULT_DESC.get(selected_fault, "") + "</div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border-color:#1e3a5f;margin:0.9rem 0;'>", unsafe_allow_html=True)

# ─── Run PSO ──────────────────────────────────────────────────────────────────
if inject or "result" not in st.session_state:
    prog = st.progress(0, text="Running PSO reconfiguration engine...")
    for p in range(0, 101, 4):
        time.sleep(0.025)
        prog.progress(p, text="PSO iteration... {}%".format(p))
    result = run_surya(selected_fault)
    st.session_state["result"] = result
    prog.empty()

result        = st.session_state["result"]
moved_panels  = {m["panel"] for m in result["panel_movements"]}
faulty_panels = {pid for pid, ps in result["panel_states"].items() if ps["is_faulty"]}

# ─── SECTION 2 — Panel Health ─────────────────────────────────────────────────
section_title("&there4;", "Panel Health Status Before Reconfiguration")

panel_cols = st.columns(9)
for i, pid in enumerate(["P1","P2","P3","P4","P5","P6","P7","P8","P9"]):
    ps = result["panel_states"][pid]
    with panel_cols[i]:
        st.markdown(
            panel_card(pid, ps["irradiance"], ps["voltage"], ps["current"], ps["is_faulty"]),
            unsafe_allow_html=True,
        )

st.markdown("""
<div style="display:flex;gap:1.5rem;justify-content:center;margin:0.7rem 0;">
    <span style="font-size:0.76rem;color:#10b981;font-family:'Exo 2',sans-serif;">&#11044; Healthy</span>
    <span style="font-size:0.76rem;color:#ef4444;font-family:'Exo 2',sans-serif;">&#11044; Fault Detected</span>
    <span style="font-size:0.76rem;color:#3b82f6;font-family:'Exo 2',sans-serif;">&#11044; Moved by PSO</span>
</div>
<hr style="border-color:#1e3a5f;margin:0.5rem 0;">
""", unsafe_allow_html=True)

# ─── SECTION 3 — Before / After Grid ─────────────────────────────────────────
section_title("&harr;", "TCT Arrangement Before vs After PSO")


def render_grid(arr, faulty, moved, label, border_color):
    st.markdown(
        '<div style="background:#0a1628;border:1px solid ' + border_color
        + ';border-radius:12px;padding:0.9rem;margin-bottom:0.4rem;">'
        + '<div style="font-family:\'Rajdhani\',sans-serif;font-size:0.85rem;font-weight:600;'
        'color:' + border_color + ';letter-spacing:2px;text-align:center;'
        'margin-bottom:0.7rem;text-transform:uppercase;">' + label + "</div>",
        unsafe_allow_html=True,
    )
    for row in arr:
        cols_r = st.columns(3)
        for ci, pid in enumerate(row):
            with cols_r[ci]:
                st.markdown(grid_cell(pid, pid in faulty, pid in moved), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


col_b, col_arrow, col_a = st.columns([5, 1, 5])
with col_b:
    render_grid(result["arrangement_before"], faulty_panels, set(),
                "Before Reconfiguration", "#ef4444")
with col_arrow:
    st.markdown("<div style='height:110px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;font-size:2.3rem;color:#f59e0b;margin-top:1.5rem;">&#9658;</div>
    <div style="text-align:center;font-size:0.62rem;color:#475569;
         font-family:'Share Tech Mono';letter-spacing:1px;margin-top:0.2rem;">PSO</div>
    """, unsafe_allow_html=True)
with col_a:
    render_grid(result["arrangement_after"], faulty_panels, moved_panels,
                "After PSO Optimisation", "#10b981")

st.markdown("<hr style='border-color:#1e3a5f;margin:0.9rem 0;'>", unsafe_allow_html=True)

# ─── SECTION 4 — Panel Movement Map  ─────────────────────────────────────────
# FIX 1: Use components.html() to bypass Streamlit HTML sanitiser
section_title("&map;", "Panel Movement Map")

movements = result["panel_movements"]
if movements:
    rows_html = ""
    for i, m in enumerate(movements):
        bg = "#0a1628" if i % 2 == 0 else "#0d1f3c"
        rows_html += (
            '<tr style="background:' + bg + ';border-bottom:1px solid #1e3a5f;">'
            + '<td style="padding:10px 18px;color:#3b82f6;font-family:Share Tech Mono,monospace;'
            'font-weight:700;font-size:15px;">' + m["panel"] + "</td>"
            + '<td style="padding:10px 18px;color:#ef4444;font-size:14px;">' + m["from"] + "</td>"
            + '<td style="padding:10px 8px;text-align:center;color:#f59e0b;font-size:18px;">&#8594;</td>'
            + '<td style="padding:10px 18px;color:#10b981;font-size:14px;">' + m["to"] + "</td>"
            + "</tr>"
        )

    table_html = """
    <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;overflow:hidden;margin-bottom:1rem;">
    <table style="width:100%;border-collapse:collapse;font-family:'Exo 2',sans-serif;font-size:14px;">
    <thead><tr style="background:#0d1f3c;border-bottom:2px solid #3b82f6;">
        <th style="padding:10px 18px;color:#3b82f6;font-family:'Rajdhani',sans-serif;letter-spacing:2px;font-size:12px;text-transform:uppercase;text-align:left;">Panel</th>
        <th style="padding:10px 18px;color:#3b82f6;font-family:'Rajdhani',sans-serif;letter-spacing:2px;font-size:12px;text-transform:uppercase;text-align:left;">From Position</th>
        <th style="padding:10px 8px;"></th>
        <th style="padding:10px 18px;color:#3b82f6;font-family:'Rajdhani',sans-serif;letter-spacing:2px;font-size:12px;text-transform:uppercase;text-align:left;">To Position</th>
    </tr></thead><tbody>""" + rows_html + """</tbody></table></div>"""

    st.markdown(table_html, unsafe_allow_html=True)
else:
    st.markdown(
        '<div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;'
        'padding:1.4rem;text-align:center;color:#475569;font-family:\'Exo 2\',sans-serif;">'
        "No panel movements &mdash; arrangement already optimal for this fault."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border-color:#1e3a5f;margin:0.9rem 0;'>", unsafe_allow_html=True)

# ─── SECTION 5 — Power Recovery KPIs ─────────────────────────────────────────
section_title("&zap;", "Power Recovery KPIs")

gain_w  = result["tct_power_after_w"] - result["tct_power_before_w"]
g_color = "#10b981" if gain_w >= 0 else "#ef4444"

k1, k2, k3, k4, k5 = st.columns(5)
kpi_rows = [
    (k1, "PI-LSTM Predicted",  "{:.1f}".format(result["predicted_power_lstm"]),   "W",  "#a78bfa", "AI"),
    (k2, "TCT Power Before",   "{:.1f}".format(result["tct_power_before_w"]),     "W",  "#ef4444", "PRE"),
    (k3, "TCT Power After",    "{:.1f}".format(result["tct_power_after_w"]),      "W",  "#10b981", "POST"),
    (k4, "Power Recovery",     "{:+.2f}".format(result["gain_percent"]),           "%",  g_color,   "PCT"),
    (k5, "Gain in Watts",      "{:+.1f}".format(gain_w),                          "W",  g_color,   "GAI"),
]
for col, label, val, unit, color, icon in kpi_rows:
    with col:
        st.markdown(kpi_html(label, val, unit, color, icon), unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#1e3a5f;margin:0.9rem 0;'>", unsafe_allow_html=True)

# ─── SECTIONS 6 & 7 — Charts ──────────────────────────────────────────────────
col_pwr, col_conv = st.columns(2)

with col_pwr:
    section_title("&block;", "Before vs After Power Recovery")
    cats   = ["Before<br>Reconfig", "LSTM<br>Prediction", "After<br>Reconfig"]
    vals   = [result["tct_power_before_w"], result["predicted_power_lstm"], result["tct_power_after_w"]]
    colors = ["#ef4444", "#a78bfa", "#10b981"]

    fig_bar = go.Figure()
    for cat, val, clr in zip(cats, vals, colors):
        fig_bar.add_trace(go.Bar(
            x=[cat], y=[val],
            name=cat.replace("<br>", " "),
            marker=dict(color=clr, opacity=0.88, line=dict(color=clr, width=1)),
            text=["{:.1f} W".format(val)], textposition="outside",
            textfont=dict(family="Share Tech Mono", size=11, color=clr),
            width=0.42,
        ))
    if abs(gain_w) > 0.1:
        label = "+{:.1f}W".format(gain_w) if gain_w > 0 else "{:.1f}W".format(gain_w)
        fig_bar.add_annotation(
            x=cats[2], y=result["tct_power_after_w"],
            text=label, showarrow=False,
            font=dict(color=g_color, size=11, family="Share Tech Mono"),
            yshift=32,
        )
    fig_bar.update_layout(
        **PLOTLY_BASE, showlegend=False, height=320,
        xaxis=dict(color="#64748b"),
        yaxis=dict(title="Power (W)", gridcolor="#1e3a5f", color="#475569", zeroline=False),
        bargap=0.3,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

with col_conv:
    section_title("~", "PSO Convergence History")
    conv  = result["convergence_history"]
    iters = list(range(len(conv)))

    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(
        x=iters, y=conv, fill="tozeroy",
        fillcolor="rgba(245,158,11,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig_c.add_trace(go.Scatter(
        x=iters, y=conv, mode="lines",
        line=dict(color="#f59e0b", width=2.5, shape="spline"),
        name="Best Power (W)",
        hovertemplate="Iter %{x}<br>Power: %{y:.2f} W<extra></extra>",
    ))
    fig_c.add_trace(go.Scatter(
        x=[iters[-1]], y=[conv[-1]], mode="markers",
        marker=dict(color="#10b981", size=10, line=dict(color="#0a1628", width=2)),
        name="Converged: {:.1f} W".format(conv[-1]),
    ))
    fig_c.update_layout(
        **PLOTLY_BASE, height=320,
        legend=dict(font=dict(color="#94a3b8", size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="PSO Iteration", gridcolor="#1e3a5f", color="#475569", zeroline=False),
        yaxis=dict(title="Best Power Found (W)", gridcolor="#1e3a5f", color="#475569", zeroline=False),
    )
    st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})

# ─── SECTION 8 — Irradiance Heatmaps ─────────────────────────────────────────
st.markdown("<hr style='border-color:#1e3a5f;margin:0.9rem 0;'>", unsafe_allow_html=True)
section_title("&curren;", "Irradiance Distribution Heatmap Before vs After")

hmap_base = dict(
    **PLOTLY_BASE, height=255,
    xaxis=dict(title="Column", tickvals=[0,1,2], ticktext=["Col 1","Col 2","Col 3"],
               color="#475569", gridcolor="rgba(0,0,0,0)"),
    yaxis=dict(title="Row",    tickvals=[0,1,2], ticktext=["Row 1","Row 2","Row 3"],
               color="#475569", gridcolor="rgba(0,0,0,0)"),
)


def make_heatmap(irr_map, title):
    fig = go.Figure(go.Heatmap(
        z=irr_map,
        colorscale=[[0,"#1a0505"],[0.3,"#7f1d1d"],[0.6,"#d97706"],[1.0,"#fbbf24"]],
        zmin=0, zmax=1000,
        colorbar=dict(
            title=dict(text="W/m2", font=dict(color="#94a3b8", size=12)),
            tickfont=dict(color="#94a3b8"),
        ),
        text=[["{:.0f}".format(v) for v in row] for row in irr_map],
        texttemplate="%{text}",
        textfont=dict(color="white", size=13, family="Share Tech Mono"),
        hovertemplate="Row %{y}, Col %{x}<br>Irradiance: %{z:.0f} W/m2<extra></extra>",
    ))
    fig.update_layout(**hmap_base, title=dict(text=title, font=dict(color="#94a3b8", size=12)))
    return fig


c_h1, c_h2 = st.columns(2)
with c_h1:
    st.plotly_chart(make_heatmap(result["irradiance_map_before"], "Before PSO"),
                    use_container_width=True, config={"displayModeBar": False})
with c_h2:
    st.plotly_chart(make_heatmap(result["irradiance_map_after"], "After PSO"),
                    use_container_width=True, config={"displayModeBar": False})

# ─── Status Bar ───────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#1e3a5f;margin:0.9rem 0;'>", unsafe_allow_html=True)
n_f = len(faulty_panels)
n_m = len(moved_panels)
mode_label = "LIVE AGNI PI-LSTM" if _USE_REAL_PSO else "SIM Built-in Engine"

st.markdown(
    '<div style="display:flex;justify-content:space-between;align-items:center;'
    'background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;'
    'padding:0.7rem 1.4rem;font-family:\'Share Tech Mono\',monospace;'
    'font-size:0.73rem;color:#475569;">'
    "<span>SURYA Digital Twin v2.0</span>"
    "<span>Mode: <span style='color:#f59e0b;'>" + mode_label + "</span></span>"
    "<span>Fault: <span style='color:#f59e0b;'>" + result["fault_classification"] + "</span></span>"
    "<span>Faulty: <span style='color:#ef4444;'>" + str(n_f) + "</span></span>"
    "<span>Moved: <span style='color:#3b82f6;'>" + str(n_m) + "</span></span>"
    "<span>Gain: <span style='color:#10b981;'>" + "{:+.2f}".format(result["gain_percent"]) + "%</span></span>"
    "<span>PSO Iters: <span style='color:#f59e0b;'>" + str(len(result["convergence_history"])) + "</span></span>"
    "</div>",
    unsafe_allow_html=True,
)
