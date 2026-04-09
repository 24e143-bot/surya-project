
import os
import json
import joblib
import pywt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# =========================================================
# 1. CONFIG
# =========================================================
DATASET_PATH = "PSG_iTech_180W_TrainData.csv"

LOOKBACK = 120
EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

NUM_PANELS = 9
V_MP = 20.0
I_MP = 1.1
PHYSICS_LOSS_WEIGHT = 0.5

RANDOM_SEED = 42

MODEL_WEIGHTS_PATH = "AGNI_PILSTM_BEST.weights.h5"
X_SCALER_PATH = "x_scaler.pkl"
Y_SCALER_PATH = "y_scaler.pkl"
CONFIG_PATH = "training_config.json"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =========================================================
# 2. CUSTOM CALLBACK FOR CLEAN LOSS PRINTING
# =========================================================
class LossPrinter(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", None)
        val_loss = logs.get("val_loss", None)
        mse = logs.get("mse", None)
        val_mse = logs.get("val_mse", None)
        physics = logs.get("physics", None)
        val_physics = logs.get("val_physics", None)

        print(f"\nEpoch {epoch + 1} Summary")
        print(f"loss       : {loss:.6f}" if loss is not None else "loss       : N/A")
        print(f"val_loss   : {val_loss:.6f}" if val_loss is not None else "val_loss   : N/A")
        print(f"mse        : {mse:.6f}" if mse is not None else "mse        : N/A")
        print(f"val_mse    : {val_mse:.6f}" if val_mse is not None else "val_mse    : N/A")
        print(f"physics    : {physics:.6f}" if physics is not None else "physics    : N/A")
        print(f"val_physics: {val_physics:.6f}" if val_physics is not None else "val_physics: N/A")


# =========================================================
# 3. PI-LSTM MODEL
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
        x_dwt, _ = inputs  # second input is last irradiance in actual units for physics loss
        x = self.lstm1(x_dwt, training=training)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dropout2(x, training=training)
        return self.out(x)

    def train_step(self, data):
        (x_dwt, irrad_last_actual), y_true = data

        with tf.GradientTape() as tape:
            y_pred = self((x_dwt, irrad_last_actual), training=True)

            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

            # Physical max power in ACTUAL watts
            p_theo_actual = (irrad_last_actual / 1000.0) * (I_MP * V_MP * NUM_PANELS)

            # Convert theoretical power to scaled target domain
            p_theo_scaled = (p_theo_actual - self.y_min) / (self.y_max - self.y_min + 1e-8)
            p_theo_scaled = tf.reshape(p_theo_scaled, (-1, 1))

            physics_violation = tf.maximum(0.0, y_pred - p_theo_scaled)
            physics_loss = tf.reduce_mean(tf.square(physics_violation))

            total_loss = mse_loss + self.physics_loss_weight * physics_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": total_loss,
            "mse": mse_loss,
            "physics": physics_loss
        }

    def test_step(self, data):
        (x_dwt, irrad_last_actual), y_true = data

        y_pred = self((x_dwt, irrad_last_actual), training=False)

        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        p_theo_actual = (irrad_last_actual / 1000.0) * (I_MP * V_MP * NUM_PANELS)
        p_theo_scaled = (p_theo_actual - self.y_min) / (self.y_max - self.y_min + 1e-8)
        p_theo_scaled = tf.reshape(p_theo_scaled, (-1, 1))

        physics_violation = tf.maximum(0.0, y_pred - p_theo_scaled)
        physics_loss = tf.reduce_mean(tf.square(physics_violation))

        total_loss = mse_loss + self.physics_loss_weight * physics_loss

        return {
            "loss": total_loss,
            "mse": mse_loss,
            "physics": physics_loss
        }


# =========================================================
# 4. DWT PREPROCESSING
# =========================================================
def apply_dwt(window_2d: np.ndarray) -> np.ndarray:
    """
    Input shape:  (LOOKBACK, 2)
    Output shape: (~LOOKBACK/2, 2) for level=1 Haar approximation
    """
    coeffs = pywt.wavedec(window_2d, wavelet="haar", level=1, axis=0)
    approx = coeffs[0]
    return approx.astype(np.float32)


# =========================================================
# 5. DATA PREPARATION
# =========================================================
def load_and_prepare_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    required_cols = ["Irradiance", "Temp", "Total_System_Power"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required_cols].isnull().sum().sum() > 0:
        raise ValueError("Dataset contains NaN values. Clean it before training.")

    print(f"Dataset shape: {df.shape}")
    print("Columns found:", df.columns.tolist())

    # Separate feature and target scaling
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_raw = df[["Irradiance", "Temp"]].values.astype(np.float32)
    y_raw = df[["Total_System_Power"]].values.astype(np.float32)

    x_scaled = x_scaler.fit_transform(x_raw).astype(np.float32)
    y_scaled = y_scaler.fit_transform(y_raw).astype(np.float32)

    X_dwt = []
    X_irrad_last_actual = []
    y = []

    print("Creating windows and applying DWT...")

    for i in range(LOOKBACK, len(df)):
        window_x_scaled = x_scaled[i - LOOKBACK:i]         # shape: (120, 2)
        x_dwt = apply_dwt(window_x_scaled)                 # shape: (60, 2)

        # Last actual irradiance before prediction point
        irrad_last_actual = x_raw[i - 1, 0]

        target_y = y_scaled[i, 0]

        X_dwt.append(x_dwt)
        X_irrad_last_actual.append(irrad_last_actual)
        y.append(target_y)

    X_dwt = np.array(X_dwt, dtype=np.float32)
    X_irrad_last_actual = np.array(X_irrad_last_actual, dtype=np.float32).reshape(-1, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    print(f"Final X_dwt shape              : {X_dwt.shape}")
    print(f"Final X_irrad_last_actual shape: {X_irrad_last_actual.shape}")
    print(f"Final y shape                  : {y.shape}")

    return X_dwt, X_irrad_last_actual, y, x_scaler, y_scaler


# =========================================================
# 6. TRAIN / VALIDATION SPLIT (TIME-SERIES SAFE)
# =========================================================
def train_val_split_time_series(X_dwt, X_irrad, y, validation_split=0.2):
    n = len(X_dwt)
    split_idx = int(n * (1 - validation_split))

    X_train_dwt = X_dwt[:split_idx]
    X_val_dwt = X_dwt[split_idx:]

    X_train_irrad = X_irrad[:split_idx]
    X_val_irrad = X_irrad[split_idx:]

    y_train = y[:split_idx]
    y_val = y[split_idx:]

    print(f"Training samples  : {len(X_train_dwt)}")
    print(f"Validation samples: {len(X_val_dwt)}")

    return (X_train_dwt, X_train_irrad, y_train), (X_val_dwt, X_val_irrad, y_val)


# =========================================================
# 7. TRAINING
# =========================================================
def train_model():
    X_dwt, X_irrad, y, x_scaler, y_scaler = load_and_prepare_data(DATASET_PATH)

    (X_train_dwt, X_train_irrad, y_train), (X_val_dwt, X_val_irrad, y_val) = train_val_split_time_series(
        X_dwt, X_irrad, y, validation_split=VALIDATION_SPLIT
    )

    y_min = float(y_scaler.data_min_[0])
    y_max = float(y_scaler.data_max_[0])

    model = PILSTM(
        y_min=y_min,
        y_max=y_max,
        physics_loss_weight=PHYSICS_LOSS_WEIGHT
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    )

    # Build model once so summary works
    _ = model((X_train_dwt[:1], X_train_irrad[:1]), training=False)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_WEIGHTS_PATH,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        LossPrinter()
    ]

    print("\nStarting training...\n")

    history = model.fit(
        x=(X_train_dwt, X_train_irrad),
        y=y_train,
        validation_data=((X_val_dwt, X_val_irrad), y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,   # important for time-series
        callbacks=callbacks,
        verbose=1
    )

    # Save scalers
    joblib.dump(x_scaler, X_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)

    # Save config
    config = {
        "dataset_path": DATASET_PATH,
        "lookback": LOOKBACK,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "num_panels": NUM_PANELS,
        "v_mp": V_MP,
        "i_mp": I_MP,
        "physics_loss_weight": PHYSICS_LOSS_WEIGHT,
        "model_weights_path": MODEL_WEIGHTS_PATH,
        "x_scaler_path": X_SCALER_PATH,
        "y_scaler_path": Y_SCALER_PATH
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    print("\nTraining completed successfully.")
    print(f"Best model weights saved to: {MODEL_WEIGHTS_PATH}")
    print(f"X scaler saved to          : {X_SCALER_PATH}")
    print(f"Y scaler saved to          : {Y_SCALER_PATH}")
    print(f"Config saved to            : {CONFIG_PATH}")

    # Final best validation metrics from history
    best_epoch = int(np.argmin(history.history["val_loss"]))
    print("\nBest Epoch Summary")
    print(f"Epoch      : {best_epoch + 1}")
    print(f"loss       : {history.history['loss'][best_epoch]:.6f}")
    print(f"val_loss   : {history.history['val_loss'][best_epoch]:.6f}")
    print(f"mse        : {history.history['mse'][best_epoch]:.6f}")
    print(f"val_mse    : {history.history['val_mse'][best_epoch]:.6f}")
    print(f"physics    : {history.history['physics'][best_epoch]:.6f}")
    print(f"val_physics: {history.history['val_physics'][best_epoch]:.6f}")

    # =========================================================
    # 9. FINAL MODEL ACCURACY (REGRESSION-STYLE)
    # =========================================================
    print("\nCalculating final model accuracy on validation set...")

    # Predict on validation data
    y_pred_scaled = model.predict((X_val_dwt, X_val_irrad), verbose=0)

    # Inverse transform to actual power values
    y_pred_actual = y_scaler.inverse_transform(y_pred_scaled)
    y_true_actual = y_scaler.inverse_transform(y_val)

    # Avoid division by zero
    epsilon = 1e-6
    relative_error = np.abs(y_pred_actual - y_true_actual) / (np.abs(y_true_actual) + epsilon)

    accuracy = (1 - np.mean(relative_error)) * 100

    print(f"\nFinal Model Accuracy: {accuracy:.2f}%")


# =========================================================
# 8. ENTRY POINT
# =========================================================
if __name__ == "__main__":
    train_model()