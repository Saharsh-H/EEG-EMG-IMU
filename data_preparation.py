import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================
# CONFIGURATION
# ==============================

SUBJECT_DIRS = [
    Path("Saharsh_13_Feb"),
    Path("Aabha_13_Feb")
]

NUM_TRIALS = 26
TRAIN_RATIO = 0.8

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

    # BEFORE sync (diagnostic)
    print("Before sync:")
    print("EMG samples:", len(emg_time))
    print("IMU samples:", len(imu_time))

    start = max(emg_time.min(), imu_time.min())
    end   = min(emg_time.max(), imu_time.max())

    print("Overlap window:", end - start, "seconds")

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

    # AFTER sync (diagnostic)
    print("After sync:")
    print("Aligned EMG samples:", len(emg))
    print("Aligned IMU samples:", len(imu_aligned))
    print("-----------------------------")

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

def build_subject_dataset(subject_dir, trial_numbers):

    X_all = []
    Y_all = []

    for trial_no in trial_numbers:

        trial_path = subject_dir / f"trial_{trial_no:02d}"

        try:

            emg, imu = load_trial(trial_path)

            emg, imu = synchronize_signals(emg, imu)
            
            print(f"{subject_dir.name} trial {trial_no} samples after sync:", len(emg))

            X, Y = create_windows(emg, imu)

            if len(X) == 0:
                print(f"{subject_dir.name} trial {trial_no} skipped")
                continue

            X_all.append(X)
            Y_all.append(Y)

            print(f"{subject_dir.name} trial {trial_no} windows: {len(X)}")

        except Exception as e:

            print(f"{subject_dir.name} trial {trial_no} error:", e)

    if len(X_all) == 0:
        return None, None

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    return X_all, Y_all


# ==============================
# NORMALIZATION
# ==============================

def normalize_dataset(X_train, Y_train, X_test, Y_test):

    N, T, C = X_train.shape

    X_train_flat = X_train.reshape(-1, C)

    emg_scaler = StandardScaler()
    imu_scaler = StandardScaler()

    X_train_scaled = emg_scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled.reshape(N, T, C)

    Y_train_scaled = imu_scaler.fit_transform(Y_train)

    # transform test set
    X_test_flat = X_test.reshape(-1, C)
    X_test_scaled = emg_scaler.transform(X_test_flat)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    Y_test_scaled = imu_scaler.transform(Y_test)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    joblib.dump(emg_scaler, model_dir / "emg_scaler.pkl")
    joblib.dump(imu_scaler, model_dir / "imu_scaler.pkl")

    return (
        X_train_scaled.astype(np.float32),
        Y_train_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        Y_test_scaled.astype(np.float32),
    )


# ==============================
# MAIN PIPELINE
# ==============================

def prepare_dataset():

    split_index = int(NUM_TRIALS * TRAIN_RATIO)

    train_trials = list(range(1, split_index + 1))
    test_trials  = list(range(split_index + 1, NUM_TRIALS + 1))

    print("Train trials:", train_trials)
    print("Test trials:", test_trials)

    X_train_list = []
    Y_train_list = []

    X_test_list = []
    Y_test_list = []

    for subject_dir in SUBJECT_DIRS:

        print("\nProcessing subject:", subject_dir.name)

        X_train, Y_train = build_subject_dataset(subject_dir, train_trials)
        X_test, Y_test   = build_subject_dataset(subject_dir, test_trials)

        X_train_list.append(X_train)
        Y_train_list.append(Y_train)

        X_test_list.append(X_test)
        Y_test_list.append(Y_test)

    X_train = np.concatenate(X_train_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)

    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)

    X_train, Y_train, X_test, Y_test = normalize_dataset(
        X_train, Y_train, X_test, Y_test
    )

    return X_train, Y_train, X_test, Y_test


# ==============================
# TEST RUN
# ==============================

if __name__ == "__main__":

    X_train, Y_train, X_test, Y_test = prepare_dataset()

    print("\nDataset ready")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)