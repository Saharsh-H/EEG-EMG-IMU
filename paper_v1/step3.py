# ============================================================
# SCRIPT 3 : step3_final_training.py
#
# PURPOSE:
#   - Train the FULL heavy model on the winning channels
#     identified by Script 2
#   - Winning channels : EMG_2, EMG_3, EMG_5, EMG_7
#     (indices 1, 2, 4, 6)
#   - These 4 channels gave the best R2 (0.8798) during search
#   - Saves model, metrics CSV, and per-trial plots
#
# RUN AFTER step1_preprocessing.py and step2_ablation_search.py
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib as mpl

mpl.rcParams.update({
    'font.weight': 'bold',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# ============================================================
# SEED
# ============================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\rajes\Desktop\EMG_IMU_Dataset"
SAVE_DIR = r"C:\Users\rajes\Desktop\final_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# WINNING CHANNELS FROM STEP 2
#
# EMG_2 -> index 1
# EMG_3 -> index 2
# EMG_5 -> index 4
# EMG_7 -> index 6
#
# These 4 gave the best R2 = 0.8798 during ablation
# Change these indices if your Step 2 gave different results
# ============================================================

WINNING_CHANNEL_INDICES = [1, 2, 4, 6]

ALL_CHANNEL_NAMES = [
    "EMG_1", "EMG_2", "EMG_3", "EMG_4",
    "EMG_5", "EMG_6", "EMG_7", "EMG_8"
]

WINNING_CHANNEL_NAMES = [
    ALL_CHANNEL_NAMES[i] for i in WINNING_CHANNEL_INDICES
]

print("\n" + "="*55)
print("STEP 3 : FINAL TRAINING")
print("="*55)
print(f"Winning channels : {WINNING_CHANNEL_NAMES}")
print(f"Channel indices  : {WINNING_CHANNEL_INDICES}")

# ============================================================
# FULL MODEL CONFIG
# ============================================================

EPOCHS     = 180
BATCH_SIZE = 8
LR         = 0.001

# ============================================================
# LOAD DATA
# ============================================================

X_full = np.load(os.path.join(BASE_DIR, "X_emg.npy"))
Y_full = np.load(os.path.join(BASE_DIR, "Y_gyro.npy"))

print(f"\nData Loaded")
print(f"X shape : {X_full.shape}")
print(f"Y shape : {Y_full.shape}")

# ============================================================
# SELECT WINNING CHANNELS
# ============================================================

X_full = X_full[:, :, WINNING_CHANNEL_INDICES]
print(f"X after channel selection : {X_full.shape}")

# ============================================================
# DOWNSAMPLE
# ============================================================

X_full = X_full[:, ::2, :]
Y_full = Y_full[:, ::2]

# ============================================================
# NORMALIZE
# ============================================================

for i in range(X_full.shape[0]):
    for ch in range(X_full.shape[2]):
        mean = X_full[i, :, ch].mean()
        std  = X_full[i, :, ch].std() + 1e-8
        X_full[i, :, ch] = (X_full[i, :, ch] - mean) / std

for i in range(Y_full.shape[0]):
    mean = Y_full[i].mean()
    std  = Y_full[i].std() + 1e-8
    Y_full[i] = (Y_full[i] - mean) / std

# ============================================================
# SPLIT
# ============================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X_full, Y_full,
    test_size=0.2,
    random_state=SEED
)

print(f"\nTrain trials : {len(X_train)}")
print(f"Test  trials : {len(X_test)}")

# ============================================================
# DATASET
# ============================================================

class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(
    EMGDataset(X_train, Y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    EMGDataset(X_test, Y_test),
    batch_size=BATCH_SIZE
)

# ============================================================
# FULL MODEL  -- same as your original model_final.py
# ============================================================

class CNN_LSTM(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32,
                      kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64,
                      kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )

        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x.squeeze(-1)

# ============================================================
# DEVICE + MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device : {device}")

model     = CNN_LSTM(input_channels=len(WINNING_CHANNEL_INDICES)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================================================
# LOSS FUNCTIONS  -- identical to your original
# ============================================================

def derivative_loss(y_true, y_pred):
    dy_true = y_true[:, 1:] - y_true[:, :-1]
    dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
    return nn.MSELoss()(dy_true, dy_pred)


def amplitude_loss(y_true, y_pred):
    return (y_true.std(dim=1) - y_pred.std(dim=1)).pow(2).mean()


def pearson_corr_loss(y_true, y_pred):
    vx   = y_true - torch.mean(y_true, dim=1, keepdim=True)
    vy   = y_pred - torch.mean(y_pred, dim=1, keepdim=True)
    corr = torch.sum(vx * vy, dim=1) / (
        torch.sqrt(torch.sum(vx**2, dim=1) + 1e-8) *
        torch.sqrt(torch.sum(vy**2, dim=1) + 1e-8)
    )
    return 1 - corr.mean()


def combined_loss(y_true, y_pred):
    mse       = nn.MSELoss()(y_true, y_pred)
    d_loss    = derivative_loss(y_true, y_pred)
    a_loss    = amplitude_loss(y_true, y_pred)
    corr_loss = pearson_corr_loss(y_true, y_pred)
    return (
        0.4 * mse +
        1.0 * d_loss +
        0.2 * a_loss +
        0.4 * corr_loss
    )

# ============================================================
# TRAINING
# ============================================================

print("\nTraining Started\n")

train_losses = []

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred   = model(xb)
        loss   = combined_loss(yb, pred)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d} | Loss {avg_loss:.4f}")

# ============================================================
# EVALUATION
# ============================================================

model.eval()
preds, trues = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb   = xb.to(device)
        pred = model(xb)
        preds.append(pred.cpu().numpy())
        trues.append(yb.numpy())

preds = np.concatenate(preds)
trues = np.concatenate(trues)

# ============================================================
# METRICS
# ============================================================

mse     = mean_squared_error(trues.flatten(), preds.flatten())
r2      = r2_score(trues.flatten(), preds.flatten())
pearson = pearsonr(trues.flatten(), preds.flatten())[0]

print("\n" + "="*55)
print("FINAL RESULTS")
print("="*55)
print(f"Channels Used : {WINNING_CHANNEL_NAMES}")
print(f"MSE           : {mse:.4f}")
print(f"R2 Score      : {r2:.4f}")
print(f"Pearson Corr  : {pearson:.4f}")
print("="*55)

# ============================================================
# SANITY CHECK vs ABLATION SEARCH
# ============================================================

ABLATION_R2 = 0.8798

print("\nSANITY CHECK")
print(f"  Ablation search R2 (lightweight, 30 epochs) : {ABLATION_R2:.4f}")
print(f"  Final model R2     (full model,  180 epochs) : {r2:.4f}")

if r2 >= ABLATION_R2:
    print("  PASS -- Full model matches or beats the search result")
else:
    diff = ABLATION_R2 - r2
    print(f"  NOTE -- R2 is {diff:.4f} lower than search result")
    print("         This can happen due to model stochasticity.")
    print("         If gap > 0.05, consider rerunning or checking data.")

# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(SAVE_DIR, "cnn_lstm_optimized_channels.pth")
torch.save(model.state_dict(), model_path)
print(f"\nModel saved    : {model_path}")

# ============================================================
# SAVE METRICS CSV
# ============================================================

metrics_df = pd.DataFrame([{
    "channels"      : str(WINNING_CHANNEL_NAMES),
    "num_channels"  : len(WINNING_CHANNEL_NAMES),
    "mse"           : round(mse, 4),
    "r2"            : round(r2, 4),
    "pearson"       : round(pearson, 4),
    "epochs"        : EPOCHS
}])
metrics_path = os.path.join(SAVE_DIR, "final_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"Metrics saved  : {metrics_path}")

# ============================================================
# SAVE TRAINING LOSS CURVE
# ============================================================

plt.figure(figsize=(10, 4))
plt.plot(train_losses, linewidth=1.5, color="steelblue")
plt.title(f"Training Loss Curve | Channels: {WINNING_CHANNEL_NAMES}")
plt.xlabel("Epoch")
plt.ylabel("Combined Loss")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_loss_curve.png"), dpi=100)
plt.close()
print(f"Loss curve     : {os.path.join(SAVE_DIR, 'training_loss_curve.png')}")

# ============================================================
# SAVE PER-TRIAL PLOTS
# ============================================================

print("\nSaving per-trial prediction plots...")

for i in range(len(trues)):
    trial_r2      = r2_score(trues[i], preds[i])
    trial_pearson = pearsonr(trues[i], preds[i])[0]

    plt.figure(figsize=(10, 4))
    plt.plot(trues[i], label="True Gyro Y",  linewidth=1.5)
    plt.plot(preds[i], label="Predicted",    linewidth=1.5, linestyle="--")
    plt.title(
        f"Test Trial {i+1} | R2={trial_r2:.3f} | "
        f"Pearson={trial_pearson:.3f} | "
        f"Channels: {WINNING_CHANNEL_NAMES}"
    )
    plt.xlabel("Time Steps")
    plt.ylabel("Gyro Y (normalised)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_DIR, f"trial_{i+1}.png"),
        dpi=100
    )
    plt.close()

print(f"Plots saved    : {SAVE_DIR}")

# ============================================================
# DONE
# ============================================================

print("\n" + "="*55)
print("ALL DONE")
print(f"Results in : {SAVE_DIR}")
print("="*55)