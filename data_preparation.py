import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib


# ==============================
# CONFIGURATION
# ==============================

BASE_DIR = Path("Saharsh_13_Feb")

NUM_TRIALS = 26

WINDOW_SIZE = 200

EMG_CHANNELS = [
    "emg1","emg2","emg3","emg4",
    "emg5","emg6","emg7","emg8"
]

IMU_CHANNELS = [
    "acc_x","acc_y","acc_z",
    "gyro_x","gyro_y","gyro_z"
]


# ==============================
# TIMESTAMP → SECONDS
# ==============================

def timestamp_to_seconds(ts_series):
    ts = pd.to_datetime(ts_series, format="mixed")
    return (ts - ts.iloc[0]).dt.total_seconds()


# ==============================
# LOAD SINGLE TRIAL
# ==============================

def load_trial(trial_path):

    emg = pd.read_csv(trial_path / "emg_envelope.csv")
    imu = pd.read_csv(trial_path / "imu_wrist_filtered.csv")

    emg["time"] = timestamp_to_seconds(emg["timestamp"])
    imu["time"] = timestamp_to_seconds(imu["timestamp"])

    emg = emg.drop(columns=["timestamp"])
    imu = imu.drop(columns=["timestamp"])

    return emg, imu


# ==============================
# SYNCHRONIZE IMU TO EMG
# ==============================

def synchronize_signals(emg, imu):

    emg_time = emg["time"].values
    imu_time = imu["time"].values

    # Only keep overlapping region
    start = max(emg_time.min(), imu_time.min())
    end   = min(emg_time.max(), imu_time.max())

    mask = (emg_time >= start) & (emg_time <= end)

    emg = emg[mask]
    emg_time = emg["time"].values

    imu_interp = {}

    for col in IMU_CHANNELS:

        imu_interp[col] = np.interp(
            emg_time,
            imu_time,
            imu[col].values
        )

    imu_aligned = pd.DataFrame(imu_interp)

    return emg, imu_aligned


# ==============================
# WINDOW CREATION
# ==============================

def create_windows(emg, imu):

    X = []
    Y = []

    emg_values = emg[EMG_CHANNELS].values
    imu_values = imu.values

    for i in range(WINDOW_SIZE, len(emg_values)):

        emg_window = emg_values[i-WINDOW_SIZE:i]
        imu_target = imu_values[i]

        X.append(emg_window)
        Y.append(imu_target)
    
    print("Samples after sync:", len(emg))

    return np.array(X), np.array(Y)


# ==============================
# BUILD DATASET FROM ALL TRIALS
# ==============================

def build_dataset():

    X_all = []
    Y_all = []

    for trial_no in range(1, NUM_TRIALS + 1):

        trial_path = BASE_DIR / f"trial_{trial_no:02d}"

        try:

            emg, imu = load_trial(trial_path)

            emg, imu = synchronize_signals(emg, imu)

            X, Y = create_windows(emg, imu)

            if len(X) == 0:
                print(f"Skipping trial {trial_no}: not enough data")
                continue

            X_all.append(X)
            Y_all.append(Y)

            print(f"Trial {trial_no} processed, windows: {len(X)}")

        except Exception as e:

            print(f"Skipping trial {trial_no}: {e}")


    # AFTER LOOP FINISHES

    if len(X_all) == 0:
        raise ValueError("No valid trials found")

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    print("\nFinal dataset shape:")
    print("X:", X_all.shape)
    print("Y:", Y_all.shape)

    return X_all, Y_all


# ==============================
# NORMALIZATION
# ==============================

def normalize_dataset(X, Y):

    N, T, C = X.shape

    X_flat = X.reshape(-1, C)

    emg_scaler = StandardScaler()
    imu_scaler = StandardScaler()

    X_scaled = emg_scaler.fit_transform(X_flat)
    X_scaled = X_scaled.reshape(N, T, C)

    Y_scaled = imu_scaler.fit_transform(Y)

    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(emg_scaler, MODEL_DIR / "emg_scaler.pkl")
    joblib.dump(imu_scaler, MODEL_DIR / "imu_scaler.pkl")

    X_scaled = X_scaled.astype(np.float32)
    Y_scaled = Y_scaled.astype(np.float32)

    return X_scaled, Y_scaled


# ==============================
# MAIN PIPELINE
# ==============================

def prepare_dataset():

    X, Y = build_dataset()
    X, Y = normalize_dataset(X, Y)
    return X, Y


# ==============================
# TEST RUN
# ==============================

if __name__ == "__main__":

    X, Y = prepare_dataset()
    print("\nDataset ready")