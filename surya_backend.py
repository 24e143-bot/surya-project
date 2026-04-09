# -*- coding: utf-8 -*-
"""
SURYA - IoT Solar PV Intelligent Reconfiguration System
Backend: Fault Simulation + PSO-based TCT Reconfiguration Engine
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ─── Constants ───────────────────────────────────────────────────────────────
ROWS, COLS = 3, 3
N_PANELS = ROWS * COLS
STC_IRRADIANCE = 1000.0   # W/m²
STC_VOC = 21.5            # V
STC_ISC = 8.5             # A
TEMP_COEFF = -0.004       # /°C
BASE_TEMP = 25            # °C
AMBIENT_TEMP = 35         # °C

# ─── Data Structures ─────────────────────────────────────────────────────────
@dataclass
class PanelState:
    panel_id: str
    irradiance: float
    voltage: float
    current: float
    temperature: float
    is_faulty: bool
    fault_type: str

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

# ─── Fault Scenarios ─────────────────────────────────────────────────────────
FAULT_PROFILES = {
    "NORMAL": {
        "irradiance_override": None,
        "faulty_panels": [],
        "description": "All panels operating at standard test conditions.",
    },
    "TRANSIENT SHADOWING": {
        "irradiance_override": {
            "P2": 320.0,
            "P5": 280.0,
            "P8": 350.0,
        },
        "faulty_panels": ["P2", "P5", "P8"],
        "description": "Temporary cloud shadow across column 2.",
    },
    "SOILING / GRADUAL POWER DEGRADATION": {
        "irradiance_override": {
            "P1": 750.0,
            "P4": 680.0,
            "P7": 710.0,
        },
        "faulty_panels": ["P1", "P4", "P7"],
        "description": "Dust accumulation degrading column 1 panels.",
    },
    "PERSISTENT HOTSPOT / UNSTABLE FAULT": {
        "irradiance_override": {
            "P3": 150.0,
            "P6": 90.0,
        },
        "faulty_panels": ["P3", "P6"],
        "description": "Cell-level hotspot causing severe output drop.",
    },
}

# ─── Panel Physics Simulation ─────────────────────────────────────────────────
def simulate_panel(panel_id: str, irradiance: float, fault_type: str, is_faulty: bool) -> PanelState:
    g_ratio = irradiance / STC_IRRADIANCE
    temp = AMBIENT_TEMP + (irradiance / STC_IRRADIANCE) * 20  # NOCT approx
    temp_factor = 1 + TEMP_COEFF * (temp - BASE_TEMP)

    isc = STC_ISC * g_ratio
    voc = STC_VOC * temp_factor * (1 + 0.05 * np.log(g_ratio + 1e-6))
    voc = max(voc, 0)

    # Add small noise
    isc *= np.random.uniform(0.98, 1.02)
    voc *= np.random.uniform(0.98, 1.02)

    return PanelState(
        panel_id=panel_id,
        irradiance=round(irradiance, 1),
        voltage=round(voc * 0.8, 2),   # MPP voltage ≈ 0.8 Voc
        current=round(isc * 0.95, 2),  # MPP current ≈ 0.95 Isc
        temperature=round(temp, 1),
        is_faulty=is_faulty,
        fault_type=fault_type if is_faulty else "NONE",
    )

# ─── TCT Power Calculation ────────────────────────────────────────────────────
def compute_tct_power(arrangement: List[List[str]], panel_states: Dict[str, PanelState]) -> float:
    """
    Total-Cross-Tied (TCT) power:
    Each row's current = min panel current in that row (series string limit).
    Row voltage = sum of min-current panels' voltages.
    Total power = sum of row powers.
    """
    total_power = 0.0
    for row in arrangement:
        currents = [panel_states[pid].current for pid in row]
        voltages = [panel_states[pid].voltage for pid in row]
        row_current = min(currents)
        row_voltage = sum(voltages)
        total_power += row_current * row_voltage
    return round(total_power, 2)

def arrangement_to_irradiance_map(arrangement, panel_states):
    return [[panel_states[pid].irradiance for pid in row] for row in arrangement]

# ─── PSO Reconfiguration ──────────────────────────────────────────────────────
def run_pso_reconfiguration(fault_classification: str = "NORMAL", verbose_pso: bool = False) -> ReconfigResult:
    np.random.seed(42)
    random.seed(42)

    profile = FAULT_PROFILES.get(fault_classification, FAULT_PROFILES["NORMAL"])
    panel_ids = [f"P{i+1}" for i in range(N_PANELS)]

    # Default irradiance
    irradiances = {pid: STC_IRRADIANCE for pid in panel_ids}
    faulty_panels = profile["faulty_panels"]

    if profile["irradiance_override"]:
        irradiances.update(profile["irradiance_override"])

    # Simulate each panel
    panel_states: Dict[str, PanelState] = {}
    for pid in panel_ids:
        panel_states[pid] = simulate_panel(
            pid,
            irradiances[pid],
            fault_classification,
            pid in faulty_panels,
        )

    # Initial 3×3 arrangement (row-major)
    arrangement_before = [
        [panel_ids[r * COLS + c] for c in range(COLS)]
        for r in range(ROWS)
    ]

    power_before = compute_tct_power(arrangement_before, panel_states)
    irr_map_before = arrangement_to_irradiance_map(arrangement_before, panel_states)

    # ── PSO Core ──────────────────────────────────────────────────────────────
    N_PARTICLES = 30
    N_ITERATIONS = 50
    W = 0.7    # inertia
    C1 = 1.5   # cognitive
    C2 = 1.5   # social

    def decode_particle(particle):
        """Convert continuous particle to permutation arrangement."""
        order = np.argsort(particle)
        return [
            [panel_ids[order[r * COLS + c]] for c in range(COLS)]
            for r in range(ROWS)
        ]

    # Initialize particles
    particles = [np.random.permutation(N_PANELS).astype(float) for _ in range(N_PARTICLES)]
    velocities = [np.random.uniform(-1, 1, N_PANELS) for _ in range(N_PARTICLES)]

    pbest = [p.copy() for p in particles]
    pbest_scores = [compute_tct_power(decode_particle(p), panel_states) for p in pbest]

    gbest_idx = np.argmax(pbest_scores)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]

    convergence_history = [gbest_score]

    for iteration in range(N_ITERATIONS):
        for i in range(N_PARTICLES):
            r1, r2 = np.random.rand(N_PANELS), np.random.rand(N_PANELS)
            velocities[i] = (W * velocities[i]
                             + C1 * r1 * (pbest[i] - particles[i])
                             + C2 * r2 * (gbest - particles[i]))
            particles[i] += velocities[i]

            score = compute_tct_power(decode_particle(particles[i]), panel_states)
            if score > pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score
                if score > gbest_score:
                    gbest = particles[i].copy()
                    gbest_score = score

        convergence_history.append(round(gbest_score, 2))
        if verbose_pso:
            print(f"  Iter {iteration+1:03d} | Best Power: {gbest_score:.2f} W")

    arrangement_after = decode_particle(gbest)
    power_after = compute_tct_power(arrangement_after, panel_states)
    irr_map_after = arrangement_to_irradiance_map(arrangement_after, panel_states)

    # Compute panel movements
    pos_before = {}
    for r, row in enumerate(arrangement_before):
        for c, pid in enumerate(row):
            pos_before[pid] = (r, c)

    pos_after = {}
    for r, row in enumerate(arrangement_after):
        for c, pid in enumerate(row):
            pos_after[pid] = (r, c)

    panel_movements = []
    for pid in panel_ids:
        if pos_before[pid] != pos_after[pid]:
            rb, cb = pos_before[pid]
            ra, ca = pos_after[pid]
            panel_movements.append({
                "panel": pid,
                "from": f"Row {rb+1} Col {cb+1}",
                "to": f"Row {ra+1} Col {ca+1}",
            })

    gain_pct = ((power_after - power_before) / max(power_before, 1e-6)) * 100

    # LSTM-predicted power (simulated as ensemble estimate)
    lstm_noise = np.random.uniform(0.97, 1.03)
    predicted_power_lstm = round(power_after * lstm_noise, 2)

    return ReconfigResult(
        arrangement_before=arrangement_before,
        arrangement_after=arrangement_after,
        panel_states=panel_states,
        panel_movements=panel_movements,
        tct_power_before_w=power_before,
        tct_power_after_w=power_after,
        gain_percent=round(gain_pct, 2),
        convergence_history=convergence_history,
        predicted_power_lstm=predicted_power_lstm,
        fault_classification=fault_classification,
        irradiance_map_before=irr_map_before,
        irradiance_map_after=irr_map_after,
    )
