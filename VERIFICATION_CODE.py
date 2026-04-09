import os
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import pywt
import joblib
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================================================
# 1. CONFIGURATION
# =========================================================
MODEL_WEIGHTS_PATH = "AGNI_PILSTM_BEST.weights.h5"
X_SCALER_PATH = "x_scaler.pkl"
Y_SCALER_PATH = "y_scaler.pkl"
TRAINING_CONFIG_PATH = "training_config.json"
DATASET_PATH = "PSG_iTech_180W_TrainData.csv"

# Physics constants
NUM_PANELS = 9
V_MP = 20.0
I_MP = 1.1
THEORETICAL_MAX_W = NUM_PANELS * V_MP * I_MP   # 198 W

# Sampling / DWT
SHORT_BUFFER_POINTS = 50
SHORT_BUFFER_MIN_REQUIRED = 32

# Thresholds
ABS_RESIDUAL_THRESHOLD_W = 20.0
REL_RESIDUAL_THRESHOLD = 0.12
HIGH_SEVERITY_REL_THRESHOLD = 0.25
VERY_HIGH_SEVERITY_REL_THRESHOLD = 0.40

DWT_SHADOW_ENERGY_THRESHOLD = 40.0
DWT_HOTSPOT_ENERGY_THRESHOLD = 25.0
HOTSPOT_PERSISTENCE_COUNT = 4
PERSISTENCE_WINDOW_COUNT = 5

# Plausibility bounds
IRRADIANCE_MIN = 0.0
IRRADIANCE_MAX = 1300.0

TEMP_MIN = -20.0
TEMP_MAX = 90.0

POWER_MIN = 0.0
POWER_MAX = 300.0

EPS = 1e-6
RANDOM_SEED = 42
DEFAULT_LOOKBACK = 120
DEFAULT_PHYSICS_LOSS_WEIGHT = 0.5

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =========================================================
# 2. LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("AGNI_VERIFY")

# =========================================================
# 3. RESULT STRUCTURES
# =========================================================
@dataclass
class DataQualityReport:
    valid: bool
    missing_fraction: float
    duplicate_fraction: float
    out_of_range_fraction: float
    clipped_fraction: float
    notes: List[str]

@dataclass
class PredictionReport:
    predicted_power_w: float
    actual_power_w: float
    physical_max_w: float
    residual_w: float
    relative_residual: float

@dataclass
class AnomalyReport:
    anomaly_flag: bool
    anomaly_score: float
    severity: str
    confidence_score: float
    classification: str
    recommended_action: str

@dataclass
class VerificationReport:
    timestamp_utc: float
    status: str
    data_quality: Dict[str, Any]
    prediction: Dict[str, Any]
    anomaly: Dict[str, Any]
    diagnostics: Dict[str, Any]

# =========================================================
# 4. PI-LSTM MODEL
# =========================================================
class PILSTM(Model):
    def __init__(self, y_min, y_max, physics_loss_weight=0.5):
        super().__init__()
        self.lstm1 = LSTM(64, return_sequences=True)
        self.dropout1 = Dropout(0.2)
        self.lstm2 = LSTM(32)
        self.dropout2 = Dropout(0.2)
        self.out = Dense(1)

        self.y_min = tf.constant(y_min, dtype=tf.float32)
        self.y_max = tf.constant(y_max, dtype=tf.float32)
        self.physics_loss_weight = tf.constant(physics_loss_weight, dtype=tf.float32)

    def call(self, inputs, training=False):
        x_dwt, _ = inputs
        x = self.lstm1(x_dwt, training=training)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dropout2(x, training=training)
        return self.out(x)

# =========================================================
# 5. UTILITIES
# =========================================================
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def safe_mean(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return float(np.mean(arr)) if arr.size else 0.0

def safe_std(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return float(np.std(arr)) if arr.size else 0.0

def normalize_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(clamp((value - low) / (high - low), 0.0, 1.0))

# =========================================================
# 6. DWT
# =========================================================
def apply_dwt(window_2d: np.ndarray) -> np.ndarray:
    coeffs = pywt.wavedec(window_2d, wavelet="haar", level=1, axis=0)
    approx = coeffs[0]
    return approx.astype(np.float32)

def dwt_detail_energy(signal_1d: np.ndarray, wavelet="haar", level=1) -> float:
    signal_1d = np.asarray(signal_1d, dtype=np.float32).flatten()
    if len(signal_1d) < 4:
        return 0.0

    coeffs = pywt.wavedec(signal_1d, wavelet=wavelet, level=level)
    detail_energy = 0.0
    for detail in coeffs[1:]:
        detail_energy += float(np.sum(np.square(detail)))
    return detail_energy

# =========================================================
# 7. ASSET LOADING
# =========================================================
def load_assets():
    required_files = [MODEL_WEIGHTS_PATH, X_SCALER_PATH, Y_SCALER_PATH]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file missing: {f}")

    cfg = {
        "lookback": DEFAULT_LOOKBACK,
        "physics_loss_weight": DEFAULT_PHYSICS_LOSS_WEIGHT
    }

    if os.path.exists(TRAINING_CONFIG_PATH):
        with open(TRAINING_CONFIG_PATH, "r") as fp:
            loaded_cfg = json.load(fp)
        cfg.update(loaded_cfg)
        logger.info("Loaded training_config.json successfully.")
    else:
        logger.warning("training_config.json not found. Using fallback defaults.")

    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    y_min = float(y_scaler.data_min_[0])
    y_max = float(y_scaler.data_max_[0])

    model = PILSTM(
        y_min=y_min,
        y_max=y_max,
        physics_loss_weight=cfg.get("physics_loss_weight", DEFAULT_PHYSICS_LOSS_WEIGHT)
    )

    lookback = int(cfg["lookback"])
    dwt_len = lookback // 2

    dummy_x = np.zeros((1, dwt_len, 2), dtype=np.float32)
    dummy_irrad = np.zeros((1, 1), dtype=np.float32)
    _ = model((dummy_x, dummy_irrad), training=False)

    model.load_weights(MODEL_WEIGHTS_PATH)
    logger.info("Assets loaded successfully.")

    return model, x_scaler, y_scaler, cfg

# =========================================================
# 8. DATA QUALITY
# =========================================================
def clean_and_validate_history(hist_df: pd.DataFrame, lookback: int) -> Tuple[pd.DataFrame, DataQualityReport]:
    required_cols = ["Irradiance", "Temp", "Total_System_Power"]
    for col in required_cols:
        if col not in hist_df.columns:
            raise ValueError(f"Missing required column: {col}")

    if len(hist_df) < lookback:
        raise ValueError(f"Need at least {lookback} rows, got {len(hist_df)}")

    df = hist_df.copy().tail(lookback).reset_index(drop=True)
    notes = []

    missing_fraction = float(df[required_cols].isnull().sum().sum()) / float(df[required_cols].size)
    duplicate_fraction = float(df.duplicated(subset=required_cols).sum()) / float(len(df))

    out_of_range_mask = (
        (df["Irradiance"] < IRRADIANCE_MIN) | (df["Irradiance"] > IRRADIANCE_MAX) |
        (df["Temp"] < TEMP_MIN) | (df["Temp"] > TEMP_MAX) |
        (df["Total_System_Power"] < POWER_MIN) | (df["Total_System_Power"] > POWER_MAX)
    )
    out_of_range_fraction = float(out_of_range_mask.sum()) / float(len(df))

    if df[required_cols].isnull().any().any():
        notes.append("Missing values found. Applied interpolation + forward/back fill.")
        df[required_cols] = df[required_cols].interpolate(method="linear", limit_direction="both")
        df[required_cols] = df[required_cols].ffill().bfill()

    clipped_count = 0

    irr_before = df["Irradiance"].copy()
    tmp_before = df["Temp"].copy()
    pwr_before = df["Total_System_Power"].copy()

    df["Irradiance"] = df["Irradiance"].clip(IRRADIANCE_MIN, IRRADIANCE_MAX)
    df["Temp"] = df["Temp"].clip(TEMP_MIN, TEMP_MAX)
    df["Total_System_Power"] = df["Total_System_Power"].clip(POWER_MIN, POWER_MAX)

    clipped_count += int((irr_before != df["Irradiance"]).sum())
    clipped_count += int((tmp_before != df["Temp"]).sum())
    clipped_count += int((pwr_before != df["Total_System_Power"]).sum())

    clipped_fraction = float(clipped_count) / float(len(df) * 3)

    if duplicate_fraction > 0.20:
        notes.append("High duplicate fraction in history window.")
    if out_of_range_fraction > 0.05:
        notes.append("Out-of-range values detected.")
    if clipped_fraction > 0.0:
        notes.append("Sensor values were clipped to plausibility bounds.")

    valid = True
    if missing_fraction > 0.20:
        valid = False
        notes.append("Too many missing values.")
    if out_of_range_fraction > 0.25:
        valid = False
        notes.append("Too many implausible values.")

    report = DataQualityReport(
        valid=valid,
        missing_fraction=round(missing_fraction, 5),
        duplicate_fraction=round(duplicate_fraction, 5),
        out_of_range_fraction=round(out_of_range_fraction, 5),
        clipped_fraction=round(clipped_fraction, 5),
        notes=notes
    )

    return df, report

# =========================================================
# 9. PREDICTION ENGINE
# =========================================================
def predict_power(model, x_scaler, y_scaler, hist_df: pd.DataFrame, lookback: int) -> PredictionReport:
    x_raw = hist_df[["Irradiance", "Temp"]].values.astype(np.float32)
    y_actual = float(hist_df["Total_System_Power"].iloc[-1])

    x_scaled = x_scaler.transform(x_raw)
    x_dwt = apply_dwt(x_scaled)

    x_dwt_batch = np.expand_dims(x_dwt, axis=0)
    irrad_last_actual = np.array([[x_raw[-1, 0]]], dtype=np.float32)

    pred_scaled = model((x_dwt_batch, irrad_last_actual), training=False).numpy()
    pred_power_unclamped = float(y_scaler.inverse_transform(pred_scaled)[0, 0])

    physical_max = float((irrad_last_actual[0, 0] / 1000.0) * THEORETICAL_MAX_W)

    pred_power = min(pred_power_unclamped, physical_max * 1.08 + 5.0)
    pred_power = max(pred_power, 0.0)

    residual = pred_power - y_actual
    relative_residual = residual / max(pred_power, EPS)

    return PredictionReport(
        predicted_power_w=round(pred_power, 4),
        actual_power_w=round(y_actual, 4),
        physical_max_w=round(physical_max, 4),
        residual_w=round(residual, 4),
        relative_residual=round(relative_residual, 6)
    )

# =========================================================
# 10. SCORING
# =========================================================
def compute_anomaly_score(pred: PredictionReport, dq: DataQualityReport, dwt_energy_now: float, energy_history: List[float]) -> Dict[str, float]:
    residual_component = normalize_score(pred.residual_w, ABS_RESIDUAL_THRESHOLD_W, 80.0)
    rel_component = normalize_score(pred.relative_residual, REL_RESIDUAL_THRESHOLD, 0.50)
    max_dwt_energy = max(energy_history) if energy_history else 0.0
    dwt_component = normalize_score(max_dwt_energy, 5.0, 100.0)

    persistence_hits = sum(e > DWT_HOTSPOT_ENERGY_THRESHOLD for e in energy_history[-PERSISTENCE_WINDOW_COUNT:])
    persistence_component = normalize_score(persistence_hits, 1.0, float(PERSISTENCE_WINDOW_COUNT))

    data_penalty = 0.0
    data_penalty += 0.25 * clamp(dq.missing_fraction / 0.20, 0.0, 1.0)
    data_penalty += 0.20 * clamp(dq.out_of_range_fraction / 0.25, 0.0, 1.0)
    data_penalty += 0.10 * clamp(dq.duplicate_fraction / 0.30, 0.0, 1.0)

    anomaly_score = (
        0.40 * residual_component +
        0.25 * rel_component +
        0.20 * dwt_component +
        0.15 * persistence_component
    )

    confidence_score = clamp(1.0 - data_penalty, 0.0, 1.0)

    return {
        "residual_component": round(residual_component, 6),
        "relative_component": round(rel_component, 6),
        "dwt_component": round(dwt_component, 6),
        "persistence_component": round(persistence_component, 6),
        "data_penalty": round(data_penalty, 6),
        "anomaly_score": round(clamp(anomaly_score, 0.0, 1.0), 6),
        "confidence_score": round(confidence_score, 6)
    }

# =========================================================
# 11. CLASSIFICATION
# =========================================================
def classify_fault(pred: PredictionReport, dwt_energy_now: float, energy_history: List[float], confidence_score: float) -> Dict[str, str]:
    residual = pred.residual_w
    rel = pred.relative_residual

    if residual <= ABS_RESIDUAL_THRESHOLD_W or rel <= REL_RESIDUAL_THRESHOLD:
        return {
            "severity": "NORMAL",
            "classification": "SYSTEM HEALTHY",
            "recommended_action": "Continue monitoring"
        }

    if len(energy_history) == 0:
        max_dwt_energy = 0.0
    else:
        max_dwt_energy = max(energy_history)

    hotspot_hits = sum(e > DWT_HOTSPOT_ENERGY_THRESHOLD for e in energy_history[-PERSISTENCE_WINDOW_COUNT:])

    # Transient shadowing:
    # one or a few strong high-frequency bursts, but not persistent
    if max_dwt_energy >= DWT_SHADOW_ENERGY_THRESHOLD and hotspot_hits < HOTSPOT_PERSISTENCE_COUNT:
        if rel >= HIGH_SEVERITY_REL_THRESHOLD:
            return {
                "severity": "MEDIUM",
                "classification": "TRANSIENT SHADOWING",
                "recommended_action": "Trigger optimization / tracking / PSO response"
            }
        return {
            "severity": "LOW",
            "classification": "TRANSIENT SHADOWING",
            "recommended_action": "Monitor persistence before dispatching maintenance"
        }

    # Persistent repeated unstable behavior
    if hotspot_hits >= HOTSPOT_PERSISTENCE_COUNT and confidence_score >= 0.55:
        if rel >= VERY_HIGH_SEVERITY_REL_THRESHOLD:
            return {
                "severity": "CRITICAL",
                "classification": "PERSISTENT HOTSPOT / UNSTABLE FAULT",
                "recommended_action": "Isolate affected string or trigger protective shutdown"
            }
        return {
            "severity": "HIGH",
            "classification": "PERSISTENT HOTSPOT / UNSTABLE FAULT",
            "recommended_action": "Immediate inspection and derating recommended"
        }

    # Low-frequency degradation
    if max_dwt_energy < DWT_SHADOW_ENERGY_THRESHOLD and rel >= REL_RESIDUAL_THRESHOLD:
        if rel >= HIGH_SEVERITY_REL_THRESHOLD:
            return {
                "severity": "MEDIUM",
                "classification": "SOILING / GRADUAL POWER DEGRADATION",
                "recommended_action": "Schedule cleaning and inspect for static obstruction"
            }
        return {
            "severity": "LOW",
            "classification": "EARLY SOILING / GRADUAL LOSS",
            "recommended_action": "Track trend over next maintenance interval"
        }

    return {
        "severity": "MEDIUM",
        "classification": "UNCLASSIFIED ANOMALY",
        "recommended_action": "Perform operator review and retain logs"
    }

# =========================================================
# 12. SHORT BUFFER ANALYSIS
# =========================================================
def analyze_short_buffer(power_buffer: np.ndarray) -> Dict[str, Any]:
    power_buffer = np.asarray(power_buffer, dtype=np.float32).flatten()

    if len(power_buffer) < SHORT_BUFFER_MIN_REQUIRED:
        return {
            "valid": False,
            "dwt_energy": 0.0,
            "mean": safe_mean(power_buffer),
            "std": safe_std(power_buffer),
            "notes": ["Short buffer too small for reliable DWT analysis."]
        }

    dwt_energy = dwt_detail_energy(power_buffer, wavelet="haar", level=1)

    return {
        "valid": True,
        "dwt_energy": round(float(dwt_energy), 6),
        "mean": round(safe_mean(power_buffer), 6),
        "std": round(safe_std(power_buffer), 6),
        "notes": []
    }

# =========================================================
# 13. VERIFY WINDOW
# =========================================================
def verify_window(hist_df: pd.DataFrame, model, x_scaler, y_scaler, lookback: int,
                  short_power_buffers: Optional[List[np.ndarray]] = None) -> VerificationReport:
    cleaned_df, dq = clean_and_validate_history(hist_df, lookback=lookback)

    if not dq.valid:
        return VerificationReport(
            timestamp_utc=time.time(),
            status="DATA_INVALID",
            data_quality=asdict(dq),
            prediction={},
            anomaly={},
            diagnostics={"notes": ["Verification aborted due to invalid data quality."]}
        )

    pred = predict_power(model, x_scaler, y_scaler, cleaned_df, lookback=lookback)

    short_power_buffers = short_power_buffers or []
    energy_history = []
    valid_short_windows = 0

    for buf in short_power_buffers[-PERSISTENCE_WINDOW_COUNT:]:
        d = analyze_short_buffer(buf)
        if d["valid"]:
            valid_short_windows += 1
            energy_history.append(d["dwt_energy"])

    dwt_energy_now = energy_history[-1] if energy_history else 0.0

    score_parts = compute_anomaly_score(
        pred=pred,
        dq=dq,
        dwt_energy_now=dwt_energy_now,
        energy_history=energy_history
    )

    fault = classify_fault(
        pred=pred,
        dwt_energy_now=dwt_energy_now,
        energy_history=energy_history,
        confidence_score=score_parts["confidence_score"]
    )

    anomaly_flag = (
        pred.residual_w > ABS_RESIDUAL_THRESHOLD_W and
        pred.relative_residual > REL_RESIDUAL_THRESHOLD
    )

    anomaly = AnomalyReport(
        anomaly_flag=bool(anomaly_flag),
        anomaly_score=round(score_parts["anomaly_score"], 6),
        severity=fault["severity"],
        confidence_score=round(score_parts["confidence_score"], 6),
        classification=fault["classification"],
        recommended_action=fault["recommended_action"]
    )

    diagnostics = {
        "valid_short_windows": valid_short_windows,
        "latest_dwt_energy": round(float(dwt_energy_now), 6),
        "energy_history": energy_history,
        "score_breakdown": score_parts,
        "lookback_used": lookback,
        "short_buffer_points_expected": SHORT_BUFFER_POINTS
    }

    return VerificationReport(
        timestamp_utc=time.time(),
        status="OK",
        data_quality=asdict(dq),
        prediction=asdict(pred),
        anomaly=asdict(anomaly),
        diagnostics=diagnostics
    )

# =========================================================
# 14. SYNTHETIC SHORT BUFFERS
# =========================================================
def generate_synthetic_short_buffer(base_power: float, drop: float, mode: str, n_points: int = 50) -> np.ndarray:
    if n_points < 8:
        raise ValueError("n_points too small")

    if mode == "healthy":
        return np.full(n_points, base_power, dtype=np.float32)

    if mode == "soiling":
        return np.linspace(base_power, max(base_power - drop, 0), n_points).astype(np.float32)

    if mode == "shadow":
        buf = np.full(n_points, base_power, dtype=np.float32)
        half = n_points // 2
        buf[half:] = max(base_power - drop, 0)
        return buf

    if mode == "hotspot":
        t = np.arange(n_points)
        signal = np.full(n_points, max(base_power - drop, 0), dtype=np.float32)
        signal += 0.20 * max(drop, 1.0) * np.sin(2 * np.pi * 8 * t / n_points)
        signal += np.random.normal(0, 0.05 * max(drop, 1.0), size=n_points)
        return signal.astype(np.float32)

    raise ValueError(f"Unknown mode: {mode}")

# =========================================================
# 15. SELECT A GOOD DEMO WINDOW
# =========================================================
def select_best_demo_window(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    best_start = None
    best_score = -1.0

    for start in range(0, len(df) - lookback):
        window = df.iloc[start:start + lookback]
        irr_mean = window["Irradiance"].mean()
        pwr_mean = window["Total_System_Power"].mean()
        pwr_std = window["Total_System_Power"].std()
        dup_frac = float(window.duplicated(subset=["Irradiance", "Temp", "Total_System_Power"]).mean())

        score = (0.5 * irr_mean) + (1.0 * pwr_mean) + (0.5 * pwr_std) - (50.0 * dup_frac)

        if score > best_score:
            best_score = score
            best_start = start

    if best_start is None:
        raise ValueError("Could not find a valid demo window.")

    return df.iloc[best_start:best_start + lookback].copy().reset_index(drop=True)

# =========================================================
# 16. DEMO RUNNER
# =========================================================
def run_demo():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Demo dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    required_cols = ["Irradiance", "Temp", "Total_System_Power"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Demo dataset missing column: {c}")

    model, x_scaler, y_scaler, cfg = load_assets()
    lookback = int(cfg["lookback"])

    hist_df = select_best_demo_window(df, lookback=lookback)
    base_power = float(hist_df["Total_System_Power"].iloc[-1])

    scenarios = [
        {
            "name": "HEALTHY",
            "drop": 0.0,
            "buffers": [
                generate_synthetic_short_buffer(base_power, 0.0, "healthy", SHORT_BUFFER_POINTS)
                for _ in range(5)
            ]
        },
        {
            "name": "SOILING",
            "drop": 30.0,
            "buffers": [
                generate_synthetic_short_buffer(base_power, 30.0, "soiling", SHORT_BUFFER_POINTS)
                for _ in range(5)
            ]
        },
        {
            "name": "SHADOW",
            "drop": 35.0,
            "buffers": [
                generate_synthetic_short_buffer(base_power, 0.0, "healthy", SHORT_BUFFER_POINTS),
                generate_synthetic_short_buffer(base_power, 0.0, "healthy", SHORT_BUFFER_POINTS),
                generate_synthetic_short_buffer(base_power, 35.0, "shadow", SHORT_BUFFER_POINTS),
                generate_synthetic_short_buffer(base_power, 0.0, "healthy", SHORT_BUFFER_POINTS),
                generate_synthetic_short_buffer(base_power, 0.0, "healthy", SHORT_BUFFER_POINTS),
            ]
        },
        {
            "name": "HOTSPOT",
            "drop": 35.0,
            "buffers": [
                generate_synthetic_short_buffer(base_power, 35.0, "hotspot", SHORT_BUFFER_POINTS)
                for _ in range(5)
            ]
        }
    ]

    print("\n" + "=" * 90)
    print("AGNI INDUSTRIAL VERIFICATION DEMO")
    print("=" * 90)

    for s in scenarios:
        hist_mod = hist_df.copy()

        if s["drop"] > 0:
            hist_mod.loc[len(hist_mod) - 1, "Total_System_Power"] = max(base_power - s["drop"], 0.0)

        report = verify_window(
            hist_df=hist_mod,
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            lookback=lookback,
            short_power_buffers=s["buffers"]
        )

        print(f"\nSCENARIO: {s['name']}")
        print(f"STATUS              : {report.status}")
        print(f"PREDICTED POWER     : {report.prediction.get('predicted_power_w', 'N/A')} W")
        print(f"ACTUAL POWER        : {report.prediction.get('actual_power_w', 'N/A')} W")
        print(f"PHYSICAL MAX        : {report.prediction.get('physical_max_w', 'N/A')} W")
        print(f"RESIDUAL            : {report.prediction.get('residual_w', 'N/A')} W")
        rel = report.prediction.get("relative_residual", None)
        print(f"RELATIVE RESIDUAL   : {rel * 100:.2f} %" if rel is not None else "RELATIVE RESIDUAL   : N/A")
        print(f"ANOMALY FLAG        : {report.anomaly.get('anomaly_flag', 'N/A')}")
        print(f"ANOMALY SCORE       : {report.anomaly.get('anomaly_score', 'N/A')}")
        print(f"CONFIDENCE SCORE    : {report.anomaly.get('confidence_score', 'N/A')}")
        print(f"SEVERITY            : {report.anomaly.get('severity', 'N/A')}")
        print(f"CLASSIFICATION      : {report.anomaly.get('classification', 'N/A')}")
        print(f"RECOMMENDED ACTION  : {report.anomaly.get('recommended_action', 'N/A')}")

    print("\n" + "=" * 90)
    print("DEMO COMPLETED")
    print("=" * 90)

# =========================================================
# 17. ENTRY POINT
# =========================================================
if __name__ == "__main__":
    run_demo()