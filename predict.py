import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import r2_score

from model import EMG2IMU_CNN_TCN
from data_preparation import prepare_dataset, IMU_CHANNELS


# ==============================
# CONFIG
# ==============================

MODEL_DIR = Path("models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# LOAD
# ==============================

emg_scaler = joblib.load(MODEL_DIR / "emg_scaler.pkl")
imu_scaler = joblib.load(MODEL_DIR / "imu_scaler.pkl")

X_trials, Y_trials = prepare_dataset()
test_indices = np.load(MODEL_DIR / "test_trial_indices.npy")


# ==============================
# LOAD MODELS
# ==============================

models = {}

for file in MODEL_DIR.glob("*_model.pt"):
    channel = file.stem.replace("_model", "")

    model = EMG2IMU_CNN_TCN(output_channels=1).to(device)
    model.load_state_dict(torch.load(file, map_location=device))
    model.eval()

    models[channel] = model


# ==============================
# OUTPUT DIRS
# ==============================

recon_root = Path("Plots/Reconstructed_IMU_Plots")
perf_root = Path("Plots/Performance")

recon_root.mkdir(parents=True, exist_ok=True)
perf_root.mkdir(parents=True, exist_ok=True)


# ==============================
# PROCESS TEST TRIALS
# ==============================

results = []

for count, trial_id in enumerate(test_indices):

    X = X_trials[trial_id]
    Y = Y_trials[trial_id]

    N, T, C = X.shape

    # ---- Scaling ----
    X_scaled = emg_scaler.transform(X.reshape(-1, C)).reshape(N, T, C)

    X_tensor = torch.tensor(X_scaled).float().to(device)

    trial_out = recon_root / f"trial_{count:02d}"   # cleaner numbering
    trial_out.mkdir(parents=True, exist_ok=True)

    for channel, model in models.items():

        idx = IMU_CHANNELS.index(channel)

        # ---- Inference ----
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()

        # ---- Inverse scaling ----
        pred_full = np.zeros((len(pred_scaled), len(IMU_CHANNELS)))
        pred_full[:, idx] = pred_scaled[:, 0]

        pred = imu_scaler.inverse_transform(pred_full)[:, idx]
        true = Y[:, idx]

        # ---- Metrics ----
        r2 = r2_score(true, pred)

        results.append({
            "trial_id": int(trial_id),
            "channel": channel,
            "r2": r2
        })

        # ---- Plot ----
        plt.figure(figsize=(10,4))
        plt.plot(true, label="True")
        plt.plot(pred, label="Pred")
        plt.title(f"Trial {trial_id} | {channel} | R2={r2:.3f}")
        plt.legend()

        plt.savefig(trial_out / f"{channel}.png")
        plt.close()


# ==============================
# SAVE RESULTS
# ==============================


# ==============================
# PERFORMANCE PLOTS
# ==============================

df = pd.DataFrame(results)
df.to_csv(perf_root / "test_results.csv", index=False)

for channel in df["channel"].unique():

    subset = df[df["channel"] == channel]

    plt.figure(figsize=(8,4))
    plt.hist(subset["r2"], bins=15)
    plt.title(f"R2 Distribution - {channel}")
    plt.xlabel("R2")
    plt.ylabel("Count")

    plt.savefig(perf_root / f"{channel}_r2_distribution.png")
    plt.close()


# Mean R2 bar plot
mean_r2 = df.groupby("channel")["r2"].mean()

plt.figure(figsize=(6,4))
mean_r2.plot(kind="bar")
plt.title("Mean R2 per Channel")
plt.ylabel("R2")

plt.savefig(perf_root / "mean_r2.png")
plt.close()

print("Done.")