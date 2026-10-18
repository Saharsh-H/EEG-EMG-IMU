# ============================================================
# SCRIPT 1 : step1_preprocessing.py
#
# PURPOSE   : Load raw EMG + IMU data, compute EMG envelopes,
#             resample to a common timeline, and save arrays.
#
# RUN ORDER : Run this FIRST, once only.
#
# OUTPUT    : X_emg.npy  → shape (trials, samples, 8)
#             Y_gyro.npy → shape (trials, samples)
# ============================================================

import os
import random
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import interp1d
from pathlib import Path

# ============================================================
# SEED  ← fixed so shuffle is reproducible across all scripts
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# PATHS  ← update BASE_DIR to match your folder
# ============================================================

BASE_DIR    = r"C:\Users\rajes\Desktop\EMG_IMU_Dataset"
DATASET_DIR = os.path.join(BASE_DIR, "new_trials")

# ============================================================
# PARAMETERS
# ============================================================

TARGET_FS      = 100
TRIAL_DURATION = 6
TARGET_SAMPLES = TARGET_FS * TRIAL_DURATION   # 600 samples per trial

TIME_COL     = "timestamp"

BP_LOW       = 1.0
BP_HIGH      = 40.0
ENV_LP       = 3.0
FILTER_ORDER = 4

# ============================================================
# FILTERS
# ============================================================

def bandpass_filter(signal, fs, lowcut, highcut):
    nyq  = 0.5 * fs
    b, a = butter(FILTER_ORDER, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def lowpass_filter(signal, fs, cutoff):
    nyq  = 0.5 * fs
    b, a = butter(FILTER_ORDER, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

# ============================================================
# EMG ENVELOPE
# ============================================================

def compute_emg_envelope(raw, fs):
    raw      = raw - np.mean(raw)
    filtered = bandpass_filter(raw, fs, BP_LOW, BP_HIGH)
    env      = np.abs(hilbert(filtered))
    env      = lowpass_filter(env, fs, ENV_LP)
    return env

# ============================================================
# HELPERS
# ============================================================

def timestamp_to_seconds(ts):
    ts = pd.to_datetime(ts)
    return (ts - ts.iloc[0]).dt.total_seconds().values


def resample_signal(t_old, signal, t_new):
    f = interp1d(t_old, signal, axis=0, fill_value="extrapolate")
    return f(t_new)

# ============================================================
# MAIN
# ============================================================

def build_dataset():

    trials = sorted(os.listdir(DATASET_DIR))
    X_emg  = []
    Y_gyro = []

    for trial in trials:

        trial_path = Path(DATASET_DIR) / trial
        emg_file   = trial_path / "emg_data.csv"
        imu_file   = trial_path / "imu_wrist_data.csv"

        if not emg_file.exists() or not imu_file.exists():
            print(f"  [SKIP] {trial} — missing file")
            continue

        print(f"\nProcessing: {trial}")

        emg_df = pd.read_csv(emg_file)
        imu_df = pd.read_csv(imu_file)

        if len(emg_df) < 50 or len(imu_df) < 50:
            print(f"  [SKIP] {trial} — too short")
            continue

        # ── timestamps ──────────────────────────────────────
        t_emg  = timestamp_to_seconds(emg_df[TIME_COL])
        t_imu  = timestamp_to_seconds(imu_df[TIME_COL])
        fs_emg = 1.0 / np.mean(np.diff(t_emg))
        t_end  = min(t_emg[-1], t_imu[-1])
        t_new  = np.linspace(0, t_end, TARGET_SAMPLES)

        # ── all 8 EMG channels (no removal here) ────────────
        emg_channels  = emg_df.columns.drop(TIME_COL)
        emg_processed = []

        for ch in emg_channels:
            raw           = emg_df[ch].values
            env           = compute_emg_envelope(raw, fs_emg)
            env_resampled = resample_signal(t_emg, env, t_new)
            emg_processed.append(env_resampled)

        emg_processed = np.stack(emg_processed, axis=1)   # (samples, 8)

        # ── gyro_y ──────────────────────────────────────────
        gyro_y         = imu_df["gyro_y"].values
        gyro_resampled = resample_signal(t_imu, gyro_y, t_new)

        X_emg.append(emg_processed)
        Y_gyro.append(gyro_resampled)

    if len(X_emg) == 0:
        print("\n[ERROR] No valid trials found. Check your DATASET_DIR path.")
        return

    # ── align lengths ────────────────────────────────────────
    min_len = min(x.shape[0] for x in X_emg)
    X_emg   = np.array([x[:min_len] for x in X_emg])
    Y_gyro  = np.array([y[:min_len] for y in Y_gyro])

    # ── shuffle with fixed seed ──────────────────────────────
    idx    = np.random.permutation(len(X_emg))
    X_emg  = X_emg[idx]
    Y_gyro = Y_gyro[idx]

    # ── save ─────────────────────────────────────────────────
    np.save(os.path.join(BASE_DIR, "X_emg.npy"),  X_emg)
    np.save(os.path.join(BASE_DIR, "Y_gyro.npy"), Y_gyro)

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print(f"  X_emg  shape : {X_emg.shape}")
    print(f"  Y_gyro shape : {Y_gyro.shape}")
    print(f"  Saved to     : {BASE_DIR}")
    print("=" * 50)
    print("\nNext step → run  step2_ablation_search.py")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    build_dataset()