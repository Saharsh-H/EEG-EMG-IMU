import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from pathlib import Path

# ============================================================
# CONFIGURATION (LOCKED)
# ============================================================

FS_EMG = 100
BP_LOW = 1.0
BP_HIGH = 40.0
ENV_LP = 1.5
FILTER_ORDER = 4

TIME_COL = "timestamp"
N_TRIALS = 26
base_dir = Path("Saharsh_13_Feb")

# ============================================================
# FILTER FUNCTIONS
# ============================================================

def bandpass_filter(signal, fs, lowcut, highcut, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def lowpass_filter(signal, fs, cutoff, order):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype="low")
    return filtfilt(b, a, signal)

# ============================================================
# CORE EMG ENVELOPE PIPELINE
# ============================================================

def compute_emg_envelope(raw_signal, fs=FS_EMG):

    dc_removed = raw_signal - np.mean(raw_signal)

    bp_signal = bandpass_filter(
        dc_removed,
        fs=fs,
        lowcut=BP_LOW,
        highcut=BP_HIGH,
        order=FILTER_ORDER
    )

    envelope = np.abs(hilbert(bp_signal))

    envelope_smooth = lowpass_filter(
        envelope,
        fs=fs,
        cutoff=ENV_LP,
        order=FILTER_ORDER
    )

    return envelope_smooth

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

for trial_no in range(1, N_TRIALS + 1):
    trial_path = base_dir / f"trial_{trial_no:02d}"
    input_csv = trial_path / "emg_data.csv"
    output_csv = trial_path / "emg_envelope.csv"

    print(f"Processing {input_csv} ...")

    df = pd.read_csv(input_csv)

    emg_cols = df.columns.drop(TIME_COL)

    env_df = pd.DataFrame()
    env_df[TIME_COL] = df[TIME_COL]

    for ch in emg_cols:
        raw_signal = df[ch].values
        env_df[ch] = compute_emg_envelope(raw_signal)

    env_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

print("All trials processed successfully.")