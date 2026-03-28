import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from pathlib import Path

# ============================================================
# CONFIGURATION (UPDATED)
# ============================================================

FS_EMG = 100
FS_IMU = 100

BP_LOW = 1.0
BP_HIGH = 40.0
ENV_LP = 1.5

ACC_LP = 15.0
GYRO_LP = 6.0

FILTER_ORDER = 4

TIME_COL = "timestamp"

# Root folder containing all subject folders
DATA_ROOT = Path("23_March_Trials")

# Automatically detect subjects
SUBJECT_DIRS = [d for d in DATA_ROOT.iterdir() if d.is_dir()]

ACC_CHANNELS = ["acc_x","acc_y","acc_z"]
GYRO_CHANNELS = ["gyro_x","gyro_y","gyro_z"]

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
# IMU PREPROCESSING
# ============================================================

def preprocess_imu(df):

    imu_df = df.copy()

    # Accelerometer smoothing
    for ch in ACC_CHANNELS:
        imu_df[ch] = lowpass_filter(
            imu_df[ch].values,
            fs=FS_IMU,
            cutoff=ACC_LP,
            order=FILTER_ORDER
        )

    # Gyroscope smoothing
    for ch in GYRO_CHANNELS:
        imu_df[ch] = lowpass_filter(
            imu_df[ch].values,
            fs=FS_IMU,
            cutoff=GYRO_LP,
            order=FILTER_ORDER
        )

    return imu_df


# ============================================================
# MAIN PROCESSING LOOP (FIXED)
# ============================================================

for subject_dir in SUBJECT_DIRS:

    print("\n===================================")
    print("Processing subject:", subject_dir.name)
    print("===================================\n")

    # Automatically detect trials
    trial_dirs = sorted([
        d for d in subject_dir.iterdir()
        if d.is_dir() and "trial_" in d.name
    ])

    for trial_path in trial_dirs:

        # =========================
        # EMG ENVELOPE GENERATION
        # =========================

        input_csv = trial_path / "emg_data.csv"
        output_csv = trial_path / "emg_envelope.csv"

        print(f"Processing {input_csv} ...")

        try:
            if not input_csv.exists():
                raise FileNotFoundError("emg_data.csv missing")

            if output_csv.exists():
                print(f"Skipping (already exists): {output_csv}")
            else:
                df = pd.read_csv(input_csv)

                emg_cols = df.columns.drop(TIME_COL)

                env_df = pd.DataFrame()
                env_df[TIME_COL] = df[TIME_COL]

                for ch in emg_cols:
                    raw_signal = df[ch].values
                    env_df[ch] = compute_emg_envelope(raw_signal)

                env_df.to_csv(output_csv, index=False)
                print(f"Saved {output_csv}")

        except Exception as e:
            print(f"Skipping EMG for {trial_path.name}: {e}")


        # =========================
        # IMU PREPROCESSING
        # =========================

        imu_input = trial_path / "imu_wrist_data.csv"
        imu_output = trial_path / "imu_wrist_filtered.csv"

        print(f"Processing {imu_input} ...")

        try:
            if not imu_input.exists():
                raise FileNotFoundError("imu_wrist_data.csv missing")

            if imu_output.exists():
                print(f"Skipping (already exists): {imu_output}")
            else:
                imu_df = pd.read_csv(imu_input)

                imu_filtered = preprocess_imu(imu_df)

                imu_filtered.to_csv(imu_output, index=False)
                print(f"Saved {imu_output}")

        except Exception as e:
            print(f"Skipping IMU for {trial_path.name}: {e}")


print("\nAll subjects processed successfully.")