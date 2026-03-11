import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from pathlib import Path

# ============================================================
# CONFIGURATION (LOCKED)
# ============================================================

FS_EMG = 100        # Hz
BP_LOW = 1.0        # Hz
BP_HIGH = 40.0      # Hz
ENV_LP = 1.5        # Hz
FILTER_ORDER = 4

TIME_COL = "timestamp"
N_TRIALS = 9

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
    """
    Full EMG envelope pipeline:
    Raw EMG
      → DC removal
      → 1-40 Hz band-pass
      → Hilbert transform (magnitude)
      → 1.5 Hz low-pass smoothing
    """

    # Stage 1: DC removal
    dc_removed = raw_signal - np.mean(raw_signal)

    # Stage 2: Band-pass filtering
    bp_signal = bandpass_filter(
        dc_removed,
        fs=fs,
        lowcut=BP_LOW,
        highcut=BP_HIGH,
        order=FILTER_ORDER
    )

    # Stage 3: Hilbert envelope
    envelope = np.abs(hilbert(bp_signal))

    # Stage 4: Envelope smoothing
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
    trial_path = Path(f"trial_0{trial_no}")
    input_csv = trial_path / "emg_data.csv"
    output_csv = trial_path / "emg_envelope.csv"

    print(f"Processing {input_csv} ...")

    # Load raw EMG
    df = pd.read_csv(input_csv)

    # Identify EMG channels
    emg_cols = df.columns.drop(TIME_COL)

    # Output dataframe
    env_df = pd.DataFrame()
    env_df[TIME_COL] = df[TIME_COL]

    # Process each EMG channel
    for ch in emg_cols:
        raw_signal = df[ch].values
        env_df[ch] = compute_emg_envelope(raw_signal)

    # Save envelope CSV
    env_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

print("All trials processed successfully.")