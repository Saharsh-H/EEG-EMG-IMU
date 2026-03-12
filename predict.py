import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import r2_score

from model import EMG2IMU_CNN_TCN
from data_preparation import (
    load_trial,
    synchronize_signals,
    create_windows,
    EMG_CHANNELS,
    IMU_CHANNELS
)

# ======================================
# CONFIG
# ======================================

SUBJECT_DIRS = [
    Path("Saharsh_13_Feb"),
    Path("Aabha_13_Feb")
]

NUM_TRIALS = 26
TRAIN_SPLIT = 20

MODEL_DIR = Path("models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================
# LOAD SCALERS
# ======================================

emg_scaler = joblib.load(MODEL_DIR / "emg_scaler.pkl")
imu_scaler = joblib.load(MODEL_DIR / "imu_scaler.pkl")

# ======================================
# LOAD TRAINED MODELS
# ======================================

model_files = list(MODEL_DIR.glob("*_model.pt"))

if len(model_files) == 0:
    raise RuntimeError("No trained models found")

models = {}

for file in model_files:

    channel = file.stem.replace("_model", "")

    print("Loading model:", channel)

    model = EMG2IMU_CNN_TCN(output_channels=1).to(device)

    model.load_state_dict(
        torch.load(file, map_location=device)
    )

    model.eval()

    models[channel] = model

print("Channels loaded:", list(models.keys()))

# ======================================
# OUTPUT DIRECTORIES
# ======================================

recon_root = Path("Plots/Reconstructed_IMU_Plots")
perf_root = Path("Plots/Performance")

recon_root.mkdir(parents=True, exist_ok=True)
perf_root.mkdir(parents=True, exist_ok=True)

# ======================================
# STORE PERFORMANCE RESULTS
# ======================================

results = []

# ======================================
# PROCESS DATA
# ======================================

for subject_dir in SUBJECT_DIRS:

    print("\nProcessing subject:", subject_dir.name)

    subject_out = recon_root / subject_dir.name
    subject_out.mkdir(exist_ok=True)

    for trial_no in range(1, NUM_TRIALS + 1):

        print("Trial", trial_no)

        trial_path = subject_dir / f"trial_{trial_no:02d}"

        try:

            emg, imu = load_trial(trial_path)

            emg, imu = synchronize_signals(emg, imu)

            X, Y = create_windows(emg, imu)

            if len(X) == 0:
                continue

            # ======================================
            # SCALE INPUT
            # ======================================

            N, T, C = X.shape

            X_flat = X.reshape(-1, C)
            X_scaled = emg_scaler.transform(X_flat)
            X_scaled = X_scaled.reshape(N, T, C)

            X_tensor = torch.tensor(X_scaled).float().to(device)

            trial_out = subject_out / f"trial_{trial_no:02d}"
            trial_out.mkdir(parents=True, exist_ok=True)

            # ======================================
            # PREDICT PER CHANNEL
            # ======================================

            for channel, model in models.items():

                idx = IMU_CHANNELS.index(channel)

                with torch.no_grad():
                    pred_scaled = model(X_tensor).cpu().numpy()

                # reconstruct full IMU vector for inverse scaling
                pred_full = np.zeros((len(pred_scaled), len(IMU_CHANNELS)))
                pred_full[:, idx] = pred_scaled[:,0]

                pred = imu_scaler.inverse_transform(pred_full)[:, idx]

                true = Y[:, idx]

                r2 = r2_score(true, pred)

                results.append({
                    "subject": subject_dir.name,
                    "trial": trial_no,
                    "channel": channel,
                    "r2": r2
                })

                # ======================================
                # RECONSTRUCTION PLOT
                # ======================================

                time = np.arange(len(pred))

                fig, ax = plt.subplots(figsize=(10,4))

                ax.plot(time, true, label="True", linewidth=1)
                ax.plot(time, pred, label="Predicted", linewidth=1)

                ax.set_title(
                    f"{subject_dir.name} Trial {trial_no} {channel} (R²={r2:.3f})"
                )

                ax.set_xlabel("Window Index")
                ax.set_ylabel(channel)

                ax.legend()

                fig.tight_layout()

                fig.savefig(
                    trial_out / f"{channel}.png",
                    dpi=300
                )

                plt.close(fig)

        except Exception as e:
            print("Skipping trial:", e)

# ======================================
# SAVE PERFORMANCE SUMMARY
# ======================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    perf_root / "overall_results.csv",
    index=False
)

print("Saved overall results.")

# ======================================
# R² VS TRIAL PLOTS
# ======================================

for channel in results_df["channel"].unique():

    ch_data = results_df[results_df["channel"] == channel]

    plt.figure(figsize=(8,5))

    for subject in ch_data["subject"].unique():

        sub = ch_data[ch_data["subject"] == subject]

        plt.plot(
            sub["trial"],
            sub["r2"],
            marker="o",
            label=subject
        )

    plt.axvline(
        x=TRAIN_SPLIT + 0.5,
        linestyle="--",
        color="red",
        label="Train/Test Split"
    )

    plt.title(f"{channel} R² vs Trial")

    plt.xlabel("Trial")
    plt.ylabel("R²")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        perf_root / f"{channel}_r2_vs_trial.png",
        dpi=300
    )

    plt.close()

# ======================================
# TRAIN VS TEST DISTRIBUTION
# ======================================

for channel in results_df["channel"].unique():

    ch_data = results_df[results_df["channel"] == channel]

    train = ch_data[ch_data["trial"] <= TRAIN_SPLIT]["r2"]
    test = ch_data[ch_data["trial"] > TRAIN_SPLIT]["r2"]

    plt.figure(figsize=(6,5))

    plt.boxplot(
        [train, test],
        tick_labels=["Train Trials", "Test Trials"]
    )

    plt.title(f"{channel} R² Distribution")

    plt.ylabel("R²")
    plt.grid(True)

    plt.savefig(
        perf_root / f"{channel}_r2_distribution.png",
        dpi=300
    )

    plt.close()

# ======================================
# SUBJECT COMPARISON
# ======================================

for channel in results_df["channel"].unique():

    ch_data = results_df[results_df["channel"] == channel]

    subjects = ch_data["subject"].unique()

    data = [
        ch_data[ch_data["subject"] == s]["r2"]
        for s in subjects
    ]

    plt.figure(figsize=(6,5))

    plt.boxplot(data, tick_labels=subjects)

    plt.title(f"{channel} R² by Subject")

    plt.ylabel("R²")
    plt.grid(True)

    plt.savefig(
        perf_root / f"{channel}_subject_comparison.png",
        dpi=300
    )

    plt.close()

print("Saved performance plots.")