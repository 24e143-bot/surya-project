"""
SURYA — PSO TCT Reconfiguration (v2)

HOW IT FITS IN YOUR PIPELINE:
  ┌─────────────────────────────────────────────────────────────────┐
  │  SENSOR DATA  →  AGNI PI-LSTM  →  PSO RECONFIGURATION          │
  │  (Irradiance,       (predict        (rearrange 3×3 panels to   │
  │   Temp,              expected        maximise TCT power using   │
  │   Total Power)       power)          per-panel V, I derived     │
  │                                      from LSTM + irradiance)   │
  └─────────────────────────────────────────────────────────────────┘

"""

import os
import json
import random
import logging
import joblib
import pywt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("SURYA_PSO")

# ─────────────────────────────────────────────
# SECTION 1: CONSTANTS  (match your training config)
# ─────────────────────────────────────────────
MODEL_WEIGHTS_PATH  = "AGNI_PILSTM_BEST.weights.h5"
X_SCALER_PATH       = "x_scaler.pkl"
Y_SCALER_PATH       = "y_scaler.pkl"
CONFIG_PATH         = "training_config.json"
DATASET_PATH        = "PSG_iTech_180W_TrainData.csv"

NUM_PANELS   = 9
GRID_ROWS    = 3
GRID_COLS    = 3
V_MP         = 20.0      # rated Vmp per panel (V)
I_MP         = 1.1       # rated Imp per panel (A)
FILL_FACTOR  = 0.80      # typical FF

DEFAULT_LOOKBACK              = 120
DEFAULT_PHYSICS_LOSS_WEIGHT   = 0.5

# PSO hyperparameters
PSO_N_PARTICLES  = 40
PSO_N_ITERATIONS = 200
PSO_W   = 0.7    # inertia
PSO_C1  = 1.5    # cognitive (personal best pull)
PSO_C2  = 2.0    # social    (global best pull)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# SECTION 2: PI-LSTM  (exact copy from your training/testing code)
# ─────────────────────────────────────────────
class PILSTM(Model):
    def __init__(self, y_min, y_max, physics_loss_weight=0.5):
        super().__init__()
        self.lstm1    = LSTM(64, return_sequences=True)
        self.dropout1 = Dropout(0.2)
        self.lstm2    = LSTM(32)
        self.dropout2 = Dropout(0.2)
        self.out      = Dense(1)
        self.y_min    = tf.constant(y_min, dtype=tf.float32)
        self.y_max    = tf.constant(y_max, dtype=tf.float32)
        self.physics_loss_weight = tf.constant(physics_loss_weight, dtype=tf.float32)

    def call(self, inputs, training=False):
        x_dwt, _ = inputs
        x = self.lstm1(x_dwt, training=training)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dropout2(x, training=training)
        return self.out(x)


# ─────────────────────────────────────────────
# SECTION 3: ASSET LOADER
# ─────────────────────────────────────────────
def load_lstm_assets():
    """Load trained LSTM weights + scalers + config."""
    for f in [MODEL_WEIGHTS_PATH, X_SCALER_PATH, Y_SCALER_PATH]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Missing file: {f}\n"
                f"Make sure you have run the training code first."
            )

    cfg = {"lookback": DEFAULT_LOOKBACK, "physics_loss_weight": DEFAULT_PHYSICS_LOSS_WEIGHT}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as fp:
            cfg.update(json.load(fp))
        logger.info("Loaded training_config.json")
    else:
        logger.warning("training_config.json not found — using defaults")

    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    y_min = float(y_scaler.data_min_[0])
    y_max = float(y_scaler.data_max_[0])

    model = PILSTM(y_min=y_min, y_max=y_max,
                   physics_loss_weight=cfg.get("physics_loss_weight", DEFAULT_PHYSICS_LOSS_WEIGHT))

    lookback = int(cfg["lookback"])
    dwt_len  = lookback // 2
    model((np.zeros((1, dwt_len, 2), dtype=np.float32),
           np.zeros((1, 1),          dtype=np.float32)), training=False)
    model.load_weights(MODEL_WEIGHTS_PATH)
    logger.info("LSTM weights loaded successfully")
    return model, x_scaler, y_scaler, lookback


# ─────────────────────────────────────────────
# SECTION 4: DATA HELPERS
# ─────────────────────────────────────────────
def apply_dwt(window_2d: np.ndarray) -> np.ndarray:
    coeffs = pywt.wavedec(window_2d, wavelet="haar", level=1, axis=0)
    return coeffs[0].astype(np.float32)


def select_best_daytime_window(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Pick a lookback-length window with high, non-zero irradiance.
    Avoids night-time rows where irradiance=0 and power=0
    (those are useless for reconfiguration).
    """
    best_start = None
    best_score = -1.0
    for start in range(0, len(df) - lookback):
        window = df.iloc[start : start + lookback]
        irr_mean  = window["Irradiance"].mean()
        pwr_mean  = window["Total_System_Power"].mean()
        pwr_std   = window["Total_System_Power"].std()
        dup_frac  = float(window.duplicated(
                        subset=["Irradiance", "Temp", "Total_System_Power"]).mean())
        zero_frac = float((window["Irradiance"] == 0).mean())
        score = (0.5 * irr_mean) + (1.0 * pwr_mean) + (0.3 * pwr_std) \
                - (50.0 * dup_frac) - (80.0 * zero_frac)
        if score > best_score:
            best_score = score
            best_start = start
    if best_start is None:
        raise ValueError("No valid daytime window found in dataset.")
    return df.iloc[best_start : best_start + lookback].copy().reset_index(drop=True)


def lstm_predict_system_power(model, x_scaler, y_scaler, hist_df: pd.DataFrame) -> float:
    """
    Run LSTM on a history window → return predicted system power in Watts.
    This is exactly what your testing code does inside predict_power().
    """
    x_raw    = hist_df[["Irradiance", "Temp"]].values.astype(np.float32)
    x_scaled = x_scaler.transform(x_raw)
    x_dwt    = apply_dwt(x_scaled)
    x_dwt_b  = np.expand_dims(x_dwt, axis=0)
    irrad_last = np.array([[x_raw[-1, 0]]], dtype=np.float32)

    pred_scaled = model((x_dwt_b, irrad_last), training=False).numpy()
    pred_power  = float(y_scaler.inverse_transform(pred_scaled)[0, 0])

    # Physics cap (same as your testing code)
    phys_max   = float((irrad_last[0, 0] / 1000.0) * (V_MP * I_MP * NUM_PANELS))
    pred_power = min(pred_power, phys_max * 1.08 + 5.0)
    pred_power = max(pred_power, 0.0)
    return round(pred_power, 4)


# ─────────────────────────────────────────────
# SECTION 5: PER-PANEL STATE DERIVATION
# ─────────────────────────────────────────────
"""
KEY CONCEPT — why PSO needs per-panel data:
  Your LSTM sees system-level irradiance + temperature → predicts TOTAL system power.
  The TCT reconfiguration model needs per-PANEL V_oc and I_sc.

  We derive these as follows:
    - V_oc_i  ≈ V_MP × (1 + α_V × (T_i - T_ref)) × (1 + β_V × ln(G_i/G_ref))
    - I_sc_i  ≈ I_MP × (G_i / G_ref)
    - α_V = -0.0035 per °C  (typical silicon: Voc drops ~0.35%/°C)
    - β_V =  0.0500         (irradiance log correction)

  Because your CSV only has one irradiance + temperature column (system-level),
  we SIMULATE per-panel diversity by applying a realistic ±spread around
  the system reading. This represents real-world partial shading conditions.

  In your real deployment:
    → Replace `simulate_panel_states()` with actual per-panel sensor readings.
"""

ALPHA_V   = -0.0035   # Voc temperature coefficient (%/°C)
BETA_V    =  0.05     # Voc irradiance log correction
G_REF     = 1000.0    # reference irradiance (W/m²)
T_REF     = 25.0      # reference temperature (°C)

def derive_voc(irradiance: float, temperature: float) -> float:
    """Estimate open-circuit voltage from irradiance & temperature."""
    if irradiance <= 0:
        return 0.0
    voc = V_MP * (1 + ALPHA_V * (temperature - T_REF)) * \
                 (1 + BETA_V  * np.log(max(irradiance, 1.0) / G_REF))
    return max(float(voc), 0.0)

def derive_isc(irradiance: float) -> float:
    """Estimate short-circuit current from irradiance."""
    return max(float(I_MP * irradiance / G_REF), 0.0)

def simulate_panel_states(
    system_irradiance: float,
    system_temperature: float,
    fault_classification: str = "NORMAL",
    n_panels: int = 9,
    seed: int = RANDOM_SEED
) -> dict:
    """
    Derive per-panel V_oc and I_sc from system-level readings.

    fault_classification: from your LSTM's anomaly output, one of:
        "NORMAL", "SOILING / GRADUAL POWER DEGRADATION",
        "TRANSIENT SHADOWING", "PERSISTENT HOTSPOT / UNSTABLE FAULT",
        "EARLY SOILING / GRADUAL LOSS", "UNCLASSIFIED ANOMALY"

    Returns dict: { "P1": {V_oc, I_sc, irradiance, temperature}, ... }

    ──────────────────────────────────────────────────────────────────
    REPLACE THIS FUNCTION with real per-panel sensor data when available.
    ──────────────────────────────────────────────────────────────────
    """
    rng = np.random.default_rng(seed)
    panel_ids = [f"P{i+1}" for i in range(n_panels)]
    panel_data = {}

    # Determine how many panels are affected based on fault type
    fault = fault_classification.upper()
    if "HOTSPOT" in fault or "PERSISTENT" in fault:
        n_faulty = rng.integers(2, 4)     # 2-3 panels affected
        irr_drop_faulty = rng.uniform(0.35, 0.55)   # drop to 35–55% of normal
    elif "SHADOW" in fault or "TRANSIENT" in fault:
        n_faulty = rng.integers(1, 3)     # 1-2 panels affected
        irr_drop_faulty = rng.uniform(0.40, 0.65)
    elif "SOIL" in fault or "DEGRADATION" in fault:
        n_faulty = rng.integers(2, 5)     # 2-4 panels, mild degradation
        irr_drop_faulty = rng.uniform(0.70, 0.90)   # mild, 70-90% of normal
    else:  # NORMAL
        n_faulty = 0
        irr_drop_faulty = 1.0

    faulty_indices = set(rng.choice(n_panels, size=n_faulty, replace=False).tolist())

    for idx, pid in enumerate(panel_ids):
        # Small natural spread (±3%) among healthy panels
        natural_spread = rng.uniform(0.97, 1.03)

        if idx in faulty_indices:
            irr_i = system_irradiance * irr_drop_faulty * rng.uniform(0.90, 1.05)
            tmp_i = system_temperature + rng.uniform(-2.0, 1.0)
        else:
            irr_i = system_irradiance * natural_spread
            tmp_i = system_temperature + rng.uniform(-1.0, 2.0)

        irr_i = max(irr_i, 0.0)
        panel_data[pid] = {
            "irradiance"  : round(irr_i, 3),
            "temperature" : round(tmp_i, 2),
            "V_oc"        : round(derive_voc(irr_i, tmp_i), 4),
            "I_sc"        : round(derive_isc(irr_i),        4),
        }

    return panel_data


# ─────────────────────────────────────────────
# SECTION 6: TCT POWER MODEL
# ─────────────────────────────────────────────
def tct_power(arrangement: list, panel_data: dict) -> float:
    """
    Compute total TCT power for a given panel arrangement.

    In Total Cross-Tied (TCT) topology:
      • Each ROW has GRID_COLS panels in SERIES
          → row current = min(I_sc) of panels in that row   (weakest limits)
          → row voltage = sum(V_oc) of panels in that row
      • All ROWS are in PARALLEL
          → total power = sum of row powers

    arrangement: list of NUM_PANELS panel IDs in row-major order
    """
    total = 0.0
    for row in range(GRID_ROWS):
        row_panels  = arrangement[row * GRID_COLS : (row + 1) * GRID_COLS]
        row_current = min(panel_data[p]["I_sc"] for p in row_panels)
        row_voltage  = sum(panel_data[p]["V_oc"] for p in row_panels)
        total += FILL_FACTOR * row_current * row_voltage
    return round(total, 4)

def individual_panel_power(pid: str, panel_data: dict) -> float:
    """Max power of a single panel (no mismatch)."""
    d = panel_data[pid]
    return round(FILL_FACTOR * d["V_oc"] * d["I_sc"], 4)

def mismatch_loss(arrangement: list, panel_data: dict) -> float:
    """Power lost due to current mismatch in series connections."""
    ideal  = sum(individual_panel_power(p, panel_data) for p in panel_data)
    actual = tct_power(arrangement, panel_data)
    return round(ideal - actual, 4)


# ─────────────────────────────────────────────
# SECTION 7: PSO ENGINE  (discrete swap-based)
# ─────────────────────────────────────────────
class PSO_TCT:
    """
    Discrete PSO for TCT panel reconfiguration.

    Particle   = permutation of panel IDs (one valid 3×3 arrangement)
    Fitness    = TCT power of that arrangement  (maximise)
    Velocity   = list of (i,j) swap operations to apply to position

    Update rule (discrete adaptation of classic PSO):
      new_velocity = inertia_swaps + cognitive_swaps + social_swaps
      new_position = apply(new_velocity) to current position
    """

    def __init__(self, panel_ids: list, panel_data: dict,
                 n_particles=PSO_N_PARTICLES, n_iterations=PSO_N_ITERATIONS,
                 w=PSO_W, c1=PSO_C1, c2=PSO_C2, seed=RANDOM_SEED):

        self.panel_ids    = panel_ids
        self.panel_data   = panel_data
        self.n_panels     = len(panel_ids)
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.w  = w
        self.c1 = c1
        self.c2 = c2
        random.seed(seed)
        np.random.seed(seed)

    # ── helpers ──
    def _rand_arrangement(self):
        arr = self.panel_ids[:]
        random.shuffle(arr)
        return arr

    def _apply_swaps(self, arrangement, swaps):
        arr = arrangement[:]
        for (i, j) in swaps:
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    def _rand_swaps(self, n):
        return [(random.randint(0, self.n_panels - 1),
                 random.randint(0, self.n_panels - 1)) for _ in range(n)]

    def _drift_toward(self, current, target, prob):
        """
        Generate corrective swaps to move `current` closer to `target`.
        Each mismatched position is fixed with probability `prob`.
        """
        swaps = []
        arr   = current[:]
        for i in range(self.n_panels):
            if arr[i] != target[i] and random.random() < prob:
                j = arr.index(target[i])
                swaps.append((i, j))
                arr[i], arr[j] = arr[j], arr[i]
        return swaps

    def _fitness(self, arrangement):
        return tct_power(arrangement, self.panel_data)

    # ── main loop ──
    def run(self, verbose=True):
        # initialise
        particles  = [self._rand_arrangement() for _ in range(self.n_particles)]
        velocities = [[] for _ in range(self.n_particles)]
        pbest      = [p[:] for p in particles]
        pbest_fit  = [self._fitness(p) for p in pbest]

        gbest_idx = int(np.argmax(pbest_fit))
        gbest     = pbest[gbest_idx][:]
        gbest_fit = pbest_fit[gbest_idx]

        history = []

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  PSO — {self.n_particles} particles × {self.n_iterations} iterations")
            print(f"  Initial best TCT power : {gbest_fit:.4f} W")
            print(f"{'─'*60}")

        for iteration in range(self.n_iterations):
            for k in range(self.n_particles):

                # ── velocity update ──
                inertia   = velocities[k]  if random.random() < self.w  else []
                cognitive = self._drift_toward(particles[k], pbest[k], self.c1 / 3.0)
                social    = self._drift_toward(particles[k], gbest,    self.c2 / 3.0)
                # small random kick to escape local optima
                kick      = self._rand_swaps(1) if random.random() < 0.15 else []

                velocities[k] = inertia + cognitive + social + kick

                # ── position update ──
                particles[k] = self._apply_swaps(particles[k], velocities[k])

                # ── personal best update ──
                fit = self._fitness(particles[k])
                if fit > pbest_fit[k]:
                    pbest[k]     = particles[k][:]
                    pbest_fit[k] = fit

                # ── global best update ──
                if fit > gbest_fit:
                    gbest     = particles[k][:]
                    gbest_fit = fit

            history.append(gbest_fit)

            if verbose and (iteration + 1) % 50 == 0:
                print(f"  Iter {iteration+1:>3}: best = {gbest_fit:.4f} W")

        return gbest, gbest_fit, history


# ─────────────────────────────────────────────
# SECTION 8: REPORTING
# ─────────────────────────────────────────────
def print_grid(arrangement, label, panel_data):
    print(f"\n  {label}")
    print("  ┌──────────┬──────────┬──────────┐")
    for row in range(GRID_ROWS):
        rp   = arrangement[row * GRID_COLS : (row + 1) * GRID_COLS]
        irrs = [panel_data[p]["irradiance"] for p in rp]
        print(f"  │ " + " │ ".join(f"{p:>4}({i:>5.0f})" for p, i in zip(rp, irrs)) + " │")
        if row < GRID_ROWS - 1:
            print("  ├──────────┼──────────┼──────────┤")
    print("  └──────────┴──────────┴──────────┘")
    print("  (panel_id(irradiance W/m²))")

def print_row_breakdown(arrangement, label, panel_data):
    print(f"\n  {label} — per-row breakdown:")
    total = 0.0
    for row in range(GRID_ROWS):
        rp   = arrangement[row * GRID_COLS : (row + 1) * GRID_COLS]
        curr = min(panel_data[p]["I_sc"] for p in rp)
        volt = sum(panel_data[p]["V_oc"] for p in rp)
        pwr  = FILL_FACTOR * curr * volt
        total += pwr
        print(f"    Row {row+1}  [{', '.join(rp)}]  "
              f"I_row={curr:.3f} A   V_row={volt:.3f} V   P_row={pwr:.3f} W")
    print(f"    {'─'*58}")
    print(f"    Total TCT power = {total:.4f} W")
    return total


# ─────────────────────────────────────────────
# SECTION 9: MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pso_reconfiguration(
    fault_classification: str = "TRANSIENT SHADOWING",
    verbose_pso: bool = True
):
    """
    Full pipeline:
      1. Load LSTM assets
      2. Pick best daytime window from your CSV
      3. LSTM predicts system power  (BEFORE reconfiguration baseline)
      4. Derive per-panel states from system irradiance + fault type
      5. Compute TCT power before reconfiguration (TCT model)
      6. Run PSO to find optimal arrangement
      7. Compute TCT power after reconfiguration
      8. Print full comparison report

    Args:
        fault_classification: string from your LSTM anomaly output.
            Controls how many panels are simulated as faulty.
            Options:
              "NORMAL"
              "TRANSIENT SHADOWING"
              "SOILING / GRADUAL POWER DEGRADATION"
              "PERSISTENT HOTSPOT / UNSTABLE FAULT"
              "EARLY SOILING / GRADUAL LOSS"
              "UNCLASSIFIED ANOMALY"

        verbose_pso: print PSO iteration progress

    Returns: dict with all results (ready to push to your dashboard)
    """

    print("\n" + "═"*65)
    print("  SURYA — PSO TCT Reconfiguration")
    print(f"  Fault mode: {fault_classification}")
    print("═"*65)

    # ── Step 1: Load model ──
    model, x_scaler, y_scaler, lookback = load_lstm_assets()

    # ── Step 2: Load data + pick window ──
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    hist_df = select_best_daytime_window(df, lookback)

    sys_irradiance  = float(hist_df["Irradiance"].iloc[-1])
    sys_temperature = float(hist_df["Temp"].iloc[-1])
    actual_power    = float(hist_df["Total_System_Power"].iloc[-1])

    print(f"\n  Window stats:")
    print(f"    Last irradiance  : {sys_irradiance:.2f} W/m²")
    print(f"    Last temperature : {sys_temperature:.2f} °C")
    print(f"    Actual power     : {actual_power:.4f} W  (from sensor)")

    # ── Step 3: LSTM prediction ──
    lstm_predicted = lstm_predict_system_power(model, x_scaler, y_scaler, hist_df)
    print(f"    LSTM predicted   : {lstm_predicted:.4f} W")
    print(f"    Residual         : {lstm_predicted - actual_power:.4f} W")

    # ── Step 4: Derive per-panel states ──
    panel_data = simulate_panel_states(
        system_irradiance   = sys_irradiance,
        system_temperature  = sys_temperature,
        fault_classification = fault_classification
    )

    panel_ids = list(panel_data.keys())

    print(f"\n  Per-panel states derived from system reading:")
    print(f"  {'Panel':<6}  {'Irr (W/m²)':>10}  {'Temp (°C)':>9}  "
          f"{'V_oc (V)':>8}  {'I_sc (A)':>8}  {'Pmax (W)':>8}")
    print(f"  {'─'*60}")
    for pid in panel_ids:
        d = panel_data[pid]
        pmax = FILL_FACTOR * d["V_oc"] * d["I_sc"]
        tag  = "  ← FAULTY" if d["irradiance"] < sys_irradiance * 0.85 else ""
        print(f"  {pid:<6}  {d['irradiance']:>10.2f}  {d['temperature']:>9.2f}  "
              f"{d['V_oc']:>8.4f}  {d['I_sc']:>8.4f}  {pmax:>8.4f}{tag}")

    # ── Step 5: TCT power BEFORE reconfiguration ──
    original_arrangement = panel_ids[:]   # P1→pos0, P2→pos1, ... (as-installed)
    P_before = tct_power(original_arrangement, panel_data)
    ideal    = sum(individual_panel_power(p, panel_data) for p in panel_ids)
    loss_b   = mismatch_loss(original_arrangement, panel_data)

    print_grid(original_arrangement, "BEFORE — original 3×3 layout", panel_data)
    print_row_breakdown(original_arrangement, "BEFORE", panel_data)

    # ── Step 6: PSO ──
    pso = PSO_TCT(
        panel_ids    = panel_ids,
        panel_data   = panel_data,
        n_particles  = PSO_N_PARTICLES,
        n_iterations = PSO_N_ITERATIONS,
        w  = PSO_W,
        c1 = PSO_C1,
        c2 = PSO_C2
    )
    best_arrangement, P_after, history = pso.run(verbose=verbose_pso)
    loss_a = mismatch_loss(best_arrangement, panel_data)

    # ── Step 7: Report ──
    print_grid(best_arrangement, "AFTER — PSO optimal 3×3 layout", panel_data)
    print_row_breakdown(best_arrangement, "AFTER", panel_data)

    gain_w   = P_after - P_before
    gain_pct = (gain_w / P_before * 100) if P_before > 0 else 0.0

    print("\n" + "═"*65)
    print("  RECONFIGURATION SUMMARY")
    print("═"*65)
    print(f"  LSTM predicted power (system)  : {lstm_predicted:.4f} W")
    print(f"  Actual sensor power            : {actual_power:.4f} W")
    print(f"")
    print(f"  TCT power BEFORE reconfig      : {P_before:.4f} W")
    print(f"  TCT power AFTER  reconfig      : {P_after:.4f} W")
    print(f"  Power GAIN                     : +{gain_w:.4f} W  (+{gain_pct:.2f}%)")
    print(f"")
    print(f"  Ideal (zero mismatch) power    : {ideal:.4f} W")
    print(f"  Mismatch loss BEFORE           : {loss_b:.4f} W")
    print(f"  Mismatch loss AFTER            : {loss_a:.4f} W")
    print(f"  Mismatch loss reduced by       : {loss_b - loss_a:.4f} W  "
          f"({((loss_b - loss_a) / loss_b * 100) if loss_b > 0 else 0:.1f}%)")

    print("\n  Panel movement map:")
    print("  ┌──────┬──────────────────────────────────────────────┐")
    print("  │Panel │ Position change                              │")
    print("  ├──────┼──────────────────────────────────────────────┤")
    for orig_idx, pid in enumerate(original_arrangement):
        new_idx  = best_arrangement.index(pid)
        o_row, o_col = divmod(orig_idx, GRID_COLS)
        n_row, n_col = divmod(new_idx,  GRID_COLS)
        moved = "  ← MOVED" if orig_idx != new_idx else ""
        print(f"  │  {pid}  │  Row{o_row+1} Col{o_col+1}  →  "
              f"Row{n_row+1} Col{n_col+1}{moved:<20}│")
    print("  └──────┴──────────────────────────────────────────────┘")

    print("\n  PSO convergence:")
    for i in [0, 49, 99, 149, 199]:
        if i < len(history):
            print(f"    Iter {i+1:>3}: {history[i]:.4f} W")

    print("\n" + "═"*65)
    print("  OUTPUT READY FOR DASHBOARD")
    print("═"*65)

    # ── Return dict (push to your dashboard) ──
    result = {
        "lstm_predicted_w"     : lstm_predicted,
        "actual_sensor_w"      : actual_power,
        "system_irradiance"    : sys_irradiance,
        "system_temperature"   : sys_temperature,
        "fault_classification" : fault_classification,
        "arrangement_before"   : original_arrangement,
        "arrangement_after"    : best_arrangement,
        "tct_power_before_w"   : P_before,
        "tct_power_after_w"    : P_after,
        "gain_watts"           : round(gain_w,   4),
        "gain_percent"         : round(gain_pct, 4),
        "ideal_power_w"        : round(ideal,    4),
        "mismatch_loss_before" : round(loss_b,   4),
        "mismatch_loss_after"  : round(loss_a,   4),
        "convergence_history"  : [round(h, 4) for h in history],
        "panel_states"         : panel_data,
        "panel_movements"      : [
            {
                "panel"    : pid,
                "from_row" : original_arrangement.index(pid) // GRID_COLS + 1,
                "from_col" : original_arrangement.index(pid) %  GRID_COLS + 1,
                "to_row"   : best_arrangement.index(pid) // GRID_COLS + 1,
                "to_col"   : best_arrangement.index(pid) %  GRID_COLS + 1,
            }
            for pid in panel_ids
        ]
    }
    return result


# ─────────────────────────────────────────────
# SECTION 10: RUN ALL FAULT SCENARIOS
# ─────────────────────────────────────────────
def run_all_scenarios():
    """
    Demo: run PSO reconfiguration for all 4 fault scenarios
    and print a comparison table.
    Mirrors the scenario structure in your testing code's run_demo().
    """
    scenarios = [
        "NORMAL",
        "SOILING / GRADUAL POWER DEGRADATION",
        "TRANSIENT SHADOWING",
        "PERSISTENT HOTSPOT / UNSTABLE FAULT",
    ]

    results = {}
    for s in scenarios:
        print(f"\n{'#'*65}")
        print(f"# SCENARIO: {s}")
        print(f"{'#'*65}")
        results[s] = run_pso_reconfiguration(fault_classification=s, verbose_pso=False)

    # Summary table
    print("\n" + "═"*80)
    print("  SCENARIO COMPARISON TABLE")
    print("═"*80)
    print(f"  {'Fault Scenario':<42} {'Before (W)':>10} {'After (W)':>10} {'Gain (W)':>9} {'Gain (%)':>8}")
    print(f"  {'─'*79}")
    for s, r in results.items():
        label = s[:42]
        print(f"  {label:<42} {r['tct_power_before_w']:>10.4f} "
              f"{r['tct_power_after_w']:>10.4f} "
              f"{r['gain_watts']:>9.4f} {r['gain_percent']:>7.2f}%")
    print("═"*80)
    return results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Option A: Run one specific fault scenario ──
    # Change fault_classification to whatever your LSTM outputs
    result = run_pso_reconfiguration(
        fault_classification = "TRANSIENT SHADOWING",
        verbose_pso          = True
    )

    # ── Option B: Run all 4 scenarios ──
    # Uncomment the line below instead:
    # results = run_all_scenarios()
