# predict.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from model import EMG2IMU_CNN_TCN
from data_preparation import (
    load_trial,
    synchronize_signals,
    create_windows,
    EMG_CHANNELS,
    IMU_CHANNELS,
    BASE_DIR,
    NUM_TRIALS
)

# ==============================
# LOAD MODEL + SCALERS
# ==============================

MODEL_DIR = BASE_DIR / "Models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EMG2IMU_CNN_TCN().to(device)

model.load_state_dict(
    torch.load(MODEL_DIR / "cnn_tcn_model.pt", map_location=device)
)

model.eval()

emg_scaler = joblib.load(MODEL_DIR / "emg_scaler.pkl")
imu_scaler = joblib.load(MODEL_DIR / "imu_scaler.pkl")


# ==============================
# OUTPUT DIRECTORY
# ==============================

out_root = BASE_DIR / "Plots/Reconstructed_IMU_Plots"
out_root.mkdir(exist_ok=True)

# ==============================
# LOOP THROUGH TRIALS
# ==============================

for trial_no in range(1, NUM_TRIALS + 1):

    print(f"Processing trial {trial_no}")

    trial_path = BASE_DIR / f"trial_{trial_no:02d}"

    try:

        emg, imu = load_trial(trial_path)

        emg, imu = synchronize_signals(emg, imu)

        X, Y = create_windows(emg, imu)

        if len(X) == 0:
            print("Skipping trial: not enough windows")
            continue

        # ---------- SCALE INPUT ----------
        N, T, C = X.shape

        X_flat = X.reshape(-1, C)
        X_scaled = emg_scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(N, T, C)

        X_tensor = torch.tensor(X_scaled).float().to(device)

        # ---------- PREDICT ----------
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()

        pred = imu_scaler.inverse_transform(pred_scaled)

        # ---------- PLOT ----------
        out_dir = out_root / f"Trial {trial_no}"
        out_dir.mkdir(parents=True, exist_ok=True)

        time = np.arange(len(pred))

        for i, ch in enumerate(IMU_CHANNELS):

            fig, ax = plt.subplots(figsize=(10,4))

            ax.plot(time, Y[:, i], label="True", linewidth=1)
            ax.plot(time, pred[:, i], label="Predicted", linewidth=1)

            ax.set_title(f"Trial {trial_no} {ch} Reconstruction")
            ax.set_xlabel("Window Index")
            ax.set_ylabel(ch)

            ax.legend()

            fig.tight_layout()

            fig.savefig(out_dir / f"{ch}.png", dpi=300)

            plt.close(fig)

    except Exception as e:
        print(f"Skipping trial {trial_no}: {e}")