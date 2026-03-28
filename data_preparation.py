import pandas as pd
import numpy as np
from pathlib import Path

# ==============================
# CONFIGURATION (UPDATED)
# ==============================

DATA_ROOT = Path("23_March_Trials")

SUBJECT_DIRS = [d for d in DATA_ROOT.iterdir() if d.is_dir()]

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

    return np.array(X), np.array(Y)


# ==============================
# BUILD DATASET FOR SUBJECT
# ==============================

def build_subject_dataset(subject_dir):

    X_trials = []
    Y_trials = []

    trial_dirs = sorted([
        d for d in subject_dir.iterdir()
        if d.is_dir() and "trial_" in d.name
    ])

    for trial_path in trial_dirs:

        try:
            emg, imu = load_trial(trial_path)
            emg, imu = synchronize_signals(emg, imu)

            X, Y = create_windows(emg, imu)

            if len(X) == 0:
                print(f"{subject_dir.name} {trial_path.name} skipped (no windows)")
                continue

            X_trials.append(X)
            Y_trials.append(Y)

            print(f"{subject_dir.name} {trial_path.name} windows: {len(X)}")

        except Exception as e:
            print(f"{subject_dir.name} {trial_path.name} error:", e)

    if len(X_trials) == 0:
        return None, None

    return X_trials, Y_trials


# ==============================
# MAIN PIPELINE
# ==============================

def prepare_dataset():

    X_trials_all = []
    Y_trials_all = []

    for subject_dir in SUBJECT_DIRS:

        print("\nProcessing subject:", subject_dir.name)

        X_trials, Y_trials = build_subject_dataset(subject_dir)

        if X_trials is None:
            continue

        X_trials_all.extend(X_trials)
        Y_trials_all.extend(Y_trials)

    if len(X_trials_all) == 0:
        raise ValueError("No valid data found.")

    print("\nTotal trials:", len(X_trials_all))

    return X_trials_all, Y_trials_all


# ==============================
# TEST RUN
# ==============================

if __name__ == "__main__":

    X_trials, Y_trials = prepare_dataset()

    print("\nDataset ready")
    print("Number of trials:", len(X_trials))
    print("Example trial shape:", X_trials[0].shape)