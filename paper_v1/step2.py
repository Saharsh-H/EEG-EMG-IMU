# ============================================================
# SCRIPT 2 : step2_ablation_search.py
#
# PURPOSE   : Greedy backward channel elimination using a
#             lightweight CNN-LSTM model (fast to train).
#             Each stage removes the least important channel
#             and continues until only 1 channel remains.
#             Results are printed and saved to a CSV.
#
# RUN ORDER : Run AFTER step1_preprocessing.py
#
# OUTPUT    : ablation_results.csv   → full results table
#             ablation_summary.txt   → which channels to keep
# ============================================================

import os
import gc
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# ============================================================
# SEED
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# PATHS  ← update BASE_DIR to match your folder
# ============================================================

BASE_DIR = r"C:\Users\rajes\Desktop\EMG_IMU_Dataset"
SAVE_DIR = r"C:\Users\rajes\Desktop\ablation_results"

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LIGHTWEIGHT MODEL SETTINGS
# (fast for search — full model used in script 3)
# ============================================================

LSTM_HIDDEN  = 64     # vs 256 in full model
LSTM_LAYERS  = 1      # vs 2 in full model
SEARCH_EPOCHS = 30    # vs 180 in full model

# ============================================================
# LOAD DATA  (full 8-channel arrays from script 1)
# ============================================================

X_full = np.load(os.path.join(BASE_DIR, "X_emg.npy"))
Y_full = np.load(os.path.join(BASE_DIR, "Y_gyro.npy"))

print("\nData loaded")
print(f"  X shape : {X_full.shape}")
print(f"  Y shape : {Y_full.shape}")

# ============================================================
# DOWNSAMPLE  (keep every 2nd sample — same as your originals)
# ============================================================

X_full = X_full[:, ::2, :]
Y_full = Y_full[:, ::2]

# ============================================================
# PER-TRIAL NORMALISATION
# ============================================================

for i in range(X_full.shape[0]):
    for ch in range(X_full.shape[2]):
        mu  = X_full[i, :, ch].mean()
        std = X_full[i, :, ch].std() + 1e-8
        X_full[i, :, ch] = (X_full[i, :, ch] - mu) / std

for i in range(Y_full.shape[0]):
    mu  = Y_full[i].mean()
    std = Y_full[i].std() + 1e-8
    Y_full[i] = (Y_full[i] - mu) / std

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

# ============================================================
# LIGHTWEIGHT MODEL
# ============================================================

class LightCNN_LSTM(nn.Module):

    def __init__(self, input_channels):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size  = 32,
            hidden_size = LSTM_HIDDEN,
            num_layers  = LSTM_LAYERS,
            dropout     = 0.0,
            batch_first = True
        )

        self.fc = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x.squeeze(-1)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device : {device}")

# ============================================================
# LOSS FUNCTIONS
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
        torch.sqrt(torch.sum(vx ** 2, dim=1) + 1e-8) *
        torch.sqrt(torch.sum(vy ** 2, dim=1) + 1e-8)
    )
    return 1 - corr.mean()


def combined_loss(y_true, y_pred):
    mse       = nn.MSELoss()(y_true, y_pred)
    d_loss    = derivative_loss(y_true, y_pred)
    a_loss    = amplitude_loss(y_true, y_pred)
    corr_loss = pearson_corr_loss(y_true, y_pred)
    return 0.4 * mse + 1.0 * d_loss + 0.2 * a_loss + 0.4 * corr_loss

# ============================================================
# TRAIN AND EVALUATE FUNCTION
# ============================================================

def train_and_evaluate(X_data, Y_data, channel_indices, channel_names):

    X = X_data[:, :, channel_indices]
    Y = Y_data.copy()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=SEED
    )

    train_loader = DataLoader(
        EMGDataset(X_train, Y_train), batch_size=8, shuffle=True
    )
    test_loader = DataLoader(
        EMGDataset(X_test, Y_test), batch_size=8
    )

    model     = LightCNN_LSTM(input_channels=len(channel_indices)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ── training ─────────────────────────────────────────────
    for epoch in range(SEARCH_EPOCHS):

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

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d} | Loss {total_loss / len(train_loader):.4f}")

    # ── evaluation ───────────────────────────────────────────
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

    mse  = mean_squared_error(trues.flatten(), preds.flatten())
    r2   = r2_score(trues.flatten(), preds.flatten())
    corr = pearsonr(trues.flatten(), preds.flatten())[0]

    # ── memory cleanup ───────────────────────────────────────
    del model, optimizer, train_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mse, r2, corr

# ============================================================
# ABLATION LOOP
# ============================================================

ALL_CHANNEL_NAMES = ["EMG_1", "EMG_2", "EMG_3", "EMG_4",
                     "EMG_5", "EMG_6", "EMG_7", "EMG_8"]

active_indices = list(range(8))   # start with all 8
removed_order  = []               # track removal order
results_log    = []               # for CSV

print("\n" + "=" * 60)
print("ABLATION SEARCH STARTED")
print("=" * 60)

stage = 1

while len(active_indices) > 1:

    active_names = [ALL_CHANNEL_NAMES[i] for i in active_indices]

    print(f"\n{'=' * 60}")
    print(f"STAGE {stage}  |  Active channels: {active_names}")
    print(f"{'=' * 60}")

    stage_results = []

    for remove_pos, remove_idx in enumerate(active_indices):

        test_indices = [i for i in active_indices if i != remove_idx]
        test_names   = [ALL_CHANNEL_NAMES[i] for i in test_indices]

        print(f"\n  Removing {ALL_CHANNEL_NAMES[remove_idx]} "
              f"→ testing {test_names}")

        mse, r2, corr = train_and_evaluate(
            X_full, Y_full, test_indices, test_names
        )

        print(f"  MSE={mse:.4f}  R2={r2:.4f}  Pearson={corr:.4f}")

        stage_results.append({
            "stage"           : stage,
            "removed_channel" : ALL_CHANNEL_NAMES[remove_idx],
            "channels_used"   : str(test_names),
            "n_channels"      : len(test_indices),
            "mse"             : mse,
            "r2"              : r2,
            "pearson"         : corr
        })

        results_log.append(stage_results[-1])

    # ── find which removal gave best R2 ──────────────────────
    best = max(stage_results, key=lambda x: x["r2"])

    worst_channel = best["removed_channel"]
    worst_idx     = ALL_CHANNEL_NAMES.index(worst_channel)

    removed_order.append(worst_channel)
    active_indices = [i for i in active_indices if i != worst_idx]

    print(f"\n  → REMOVING {worst_channel} PERMANENTLY")
    print(f"    Remaining : {[ALL_CHANNEL_NAMES[i] for i in active_indices]}")

    stage += 1

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results_log)
csv_path   = os.path.join(SAVE_DIR, "ablation_results.csv")
results_df.to_csv(csv_path, index=False)

# ── final summary ─────────────────────────────────────────
final_channels = [ALL_CHANNEL_NAMES[i] for i in active_indices]

summary_lines = [
    "=" * 60,
    "ABLATION SEARCH COMPLETE",
    "=" * 60,
    f"Removal order (least → most important):",
    "  " + " → ".join(removed_order),
    "",
    f"Recommended channels to KEEP : {final_channels}",
    f"Channel indices (0-based)     : {active_indices}",
    "",
    "Now run step3_final_training.py with these channels.",
    "=" * 60
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

summary_path = os.path.join(SAVE_DIR, "ablation_summary.txt")
with open(summary_path, "w") as f:
    f.write(summary_text)

print(f"\nResults saved to : {SAVE_DIR}")
print("Next step → run  step3_final_training.py")