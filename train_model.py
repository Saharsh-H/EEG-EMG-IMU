import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import random

from data_preparation import prepare_dataset, IMU_CHANNELS
from model import EMG2IMU_CNN_TCN


# ==============================
# CONFIG
# ==============================

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 20
TRAIN_RATIO = 0.8
SEED = 42

TARGET_CHANNELS = ["acc_x", "gyro_y"]

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ==============================
# REPRODUCIBILITY
# ==============================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ==============================
# LOAD DATA
# ==============================

print("Preparing dataset...")
X_trials, Y_trials = prepare_dataset()

print("Total trials:", len(X_trials))


# ==============================
# TRIAL SPLIT (CORRECT)
# ==============================

indices = np.arange(len(X_trials))
np.random.shuffle(indices)

split_idx = int(TRAIN_RATIO * len(indices))

train_idx = indices[:split_idx]
test_idx  = indices[split_idx:]

np.save(MODEL_DIR / "test_trial_indices.npy", test_idx)

X_train_trials = [X_trials[i] for i in train_idx]
Y_train_trials = [Y_trials[i] for i in train_idx]

X_test_trials = [X_trials[i] for i in test_idx]
Y_test_trials = [Y_trials[i] for i in test_idx]

print("Train trials:", len(X_train_trials))
print("Test trials:", len(X_test_trials))


# ==============================
# FLATTEN
# ==============================

X_train = np.concatenate(X_train_trials, axis=0)
Y_train_full = np.concatenate(Y_train_trials, axis=0)

X_test = np.concatenate(X_test_trials, axis=0)
Y_test_full = np.concatenate(Y_test_trials, axis=0)


# ==============================
# NORMALIZATION (FIXED CLEANLY)
# ==============================

N, T, C = X_train.shape

emg_scaler = StandardScaler()
imu_scaler = StandardScaler()

X_train_reshaped = X_train.reshape(-1, C)
X_test_reshaped  = X_test.reshape(-1, C)

X_train = emg_scaler.fit_transform(X_train_reshaped).reshape(N, T, C)
X_test  = emg_scaler.transform(X_test_reshaped).reshape(X_test.shape)

Y_train_full = imu_scaler.fit_transform(Y_train_full)
Y_test_full  = imu_scaler.transform(Y_test_full)

joblib.dump(emg_scaler, MODEL_DIR / "emg_scaler.pkl")
joblib.dump(imu_scaler, MODEL_DIR / "imu_scaler.pkl")


# ==============================
# DEVICE
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# TRAIN LOOP
# ==============================

for TARGET_CHANNEL in TARGET_CHANNELS:

    print("\n=================================")
    print("Training model for:", TARGET_CHANNEL)
    print("=================================")

    idx = IMU_CHANNELS.index(TARGET_CHANNEL)

    Y_train = Y_train_full[:, idx:idx+1]
    Y_test  = Y_test_full[:, idx:idx+1]

    train_dataset = TensorDataset(
        torch.tensor(X_train).float(),
        torch.tensor(Y_train).float()
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test).float(),
        torch.tensor(Y_test).float()
    )

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True
    )

    model = EMG2IMU_CNN_TCN(output_channels=1).to(device)

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    early_stop_counter = 0

    history = []

    for epoch in range(EPOCHS):

        # ------------------
        # TRAIN
        # ------------------
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # ------------------
        # VALIDATION (test used)
        # ------------------
        model.eval()

        val_losses = []
        preds_all = []
        targets_all = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                preds = model(xb)
                loss = criterion(preds, yb)

                val_losses.append(loss.item())

                preds_all.append(preds.cpu().numpy())
                targets_all.append(yb.cpu().numpy())

        val_loss = np.mean(val_losses)

        preds_all = np.concatenate(preds_all).flatten()
        targets_all = np.concatenate(targets_all).flatten()

        r2 = r2_score(targets_all, preds_all)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:03d} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | R2: {r2:.4f}")

        history.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "r2": r2
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / f"{TARGET_CHANNEL}_model.pt")
            early_stop_counter = 0
            print("Model saved")
        else:
            early_stop_counter += 1

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping")
            break

    pd.DataFrame(history).to_csv(
        MODEL_DIR / f"{TARGET_CHANNEL}_training_history.csv",
        index=False
    )