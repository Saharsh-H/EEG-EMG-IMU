# ==============================
# FILE 1: pre_processing_fixed.py
# ==============================

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\rajes\Desktop\EMG_IMU_Dataset"

DATASET_DIR = os.path.join(BASE_DIR, "new_trials")

# ============================================================
# PARAMETERS
# ============================================================

TARGET_FS = 100
TRIAL_DURATION = 6
TARGET_SAMPLES = TARGET_FS * TRIAL_DURATION

TIME_COL = "timestamp"

BP_LOW = 1.0
BP_HIGH = 40.0
ENV_LP = 3.0

FILTER_ORDER = 4

# ============================================================
# FILTERS
# ============================================================

def bandpass_filter(signal, fs, lowcut, highcut):

    nyq = 0.5 * fs

    b, a = butter(
        FILTER_ORDER,
        [lowcut / nyq, highcut / nyq],
        btype='band'
    )

    return filtfilt(b, a, signal)


def lowpass_filter(signal, fs, cutoff):

    nyq = 0.5 * fs

    b, a = butter(
        FILTER_ORDER,
        cutoff / nyq,
        btype='low'
    )

    return filtfilt(b, a, signal)

# ============================================================
# EMG ENVELOPE
# ============================================================

def compute_emg_envelope(raw, fs):

    raw = raw - np.mean(raw)

    filtered = bandpass_filter(
        raw,
        fs,
        BP_LOW,
        BP_HIGH
    )

    env = np.abs(hilbert(filtered))

    env = lowpass_filter(env, fs, ENV_LP)

    return env

# ============================================================
# TIMESTAMP
# ============================================================

def timestamp_to_seconds(ts):

    ts = pd.to_datetime(ts)

    return (
        ts - ts.iloc[0]
    ).dt.total_seconds().values

# ============================================================
# RESAMPLING
# ============================================================

def resample_signal(t_old, signal, t_new):

    f = interp1d(
        t_old,
        signal,
        axis=0,
        fill_value="extrapolate"
    )

    return f(t_new)

# ============================================================
# MAIN
# ============================================================

def build_dataset():

    trials = sorted(os.listdir(DATASET_DIR))

    X_emg = []
    Y_gyro = []

    for trial in trials:

        trial_path = Path(DATASET_DIR) / trial

        emg_file = trial_path / "emg_data.csv"
        imu_file = trial_path / "imu_wrist_data.csv"

        if not emg_file.exists() or not imu_file.exists():
            continue

        print(f"\nProcessing: {trial}")

        emg_df = pd.read_csv(emg_file)
        imu_df = pd.read_csv(imu_file)

        if len(emg_df) < 50 or len(imu_df) < 50:
            print("Skipping bad trial")
            continue

        # ---------------------------------
        # TIMESTAMPS
        # ---------------------------------

        t_emg = timestamp_to_seconds(emg_df[TIME_COL])
        t_imu = timestamp_to_seconds(imu_df[TIME_COL])

        fs_emg = 1 / np.mean(np.diff(t_emg))

        # ---------------------------------
        # COMMON TIMELINE
        # ---------------------------------

        t_end = min(t_emg[-1], t_imu[-1])

        t_new = np.linspace(
            0,
            t_end,
            TARGET_SAMPLES
        )

        # ---------------------------------
        # PROCESS EMG
        # ---------------------------------

        emg_channels = emg_df.columns.drop(TIME_COL)

        emg_processed = []

        for ch in emg_channels:

            raw = emg_df[ch].values

            env = compute_emg_envelope(
                raw,
                fs_emg
            )

            env_resampled = resample_signal(
                t_emg,
                env,
                t_new
            )

            emg_processed.append(env_resampled)

        emg_processed = np.stack(
            emg_processed,
            axis=1
        )

        # ---------------------------------
        # TARGET = gyro_y
        # ---------------------------------

        gyro_y = imu_df["gyro_y"].values

        gyro_resampled = resample_signal(
            t_imu,
            gyro_y,
            t_new
        )

        X_emg.append(emg_processed)
        Y_gyro.append(gyro_resampled)

    # ========================================================
    # FIX LENGTHS
    # ========================================================

    min_len = min(x.shape[0] for x in X_emg)

    X_emg = np.array([
        x[:min_len] for x in X_emg
    ])

    Y_gyro = np.array([
        y[:min_len] for y in Y_gyro
    ])

    # ========================================================
    # SHUFFLE
    # ========================================================

    idx = np.random.permutation(len(X_emg))

    X_emg = X_emg[idx]
    Y_gyro = Y_gyro[idx]

    print("\nFinal Dataset Shapes")
    print("X:", X_emg.shape)
    print("Y:", Y_gyro.shape)

    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        os.path.join(BASE_DIR, "X_emg.npy"),
        X_emg
    )

    np.save(
        os.path.join(BASE_DIR, "Y_gyro.npy"),
        Y_gyro
    )

    print("\n✅ Dataset saved successfully")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    build_dataset()