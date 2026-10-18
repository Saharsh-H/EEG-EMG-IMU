# ============================================================
# FINAL MODEL : EMG -> ACC_X
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

from scipy.stats import pearsonr
import matplotlib as mpl

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

SAVE_DIR = r"C:\Users\rajes\Desktop\final_results_acc"

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

X = np.load(
    os.path.join(BASE_DIR, "X_emg.npy")
)

Y = np.load(
    os.path.join(BASE_DIR, "Y_acc_x.npy")
)

print("\nLoaded Dataset")
print("EMG shape:", X.shape)
print("ACC_X shape:", Y.shape)

# ============================================================
# DOWNSAMPLING
# ============================================================

X = X[:, ::2, :]
Y = Y[:, ::2]

# ============================================================
# TIME FEATURE
# ============================================================

time = np.linspace(0, 1, X.shape[1])

time = np.tile(
    time,
    (X.shape[0], 1)
)[:, :, None]

X = np.concatenate(
    [X, time],
    axis=2
)

# ============================================================
# NORMALIZATION
# ============================================================

for i in range(X.shape[0]):

    for ch in range(X.shape[2]):

        mean = X[i, :, ch].mean()
        std = X[i, :, ch].std() + 1e-8

        X[i, :, ch] = (
            X[i, :, ch] - mean
        ) / std

for i in range(Y.shape[0]):

    mean = Y[i].mean()
    std = Y[i].std() + 1e-8

    Y[i] = (
        Y[i] - mean
    ) / std

# ============================================================
# SPLIT
# ============================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=SEED
)

print("\nTrain Trials:", len(X_train))
print("Test Trials :", len(X_test))

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

        return (
            self.X[idx],
            self.y[idx]
        )

train_loader = DataLoader(
    EMGDataset(X_train, Y_train),
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    EMGDataset(X_test, Y_test),
    batch_size=8
)

# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=15,
            padding=7
        )

        self.bn1 = nn.BatchNorm1d(out_ch)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size=15,
            padding=7
        )

        self.bn2 = nn.BatchNorm1d(out_ch)

        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else None
        )

    def forward(self, x):

        identity = x

        out = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        out = self.bn2(
            self.conv2(out)
        )

        if self.skip:
            identity = self.skip(identity)

        out += identity

        return self.relu(out)

# ============================================================
# MODEL
# ============================================================

class CNN_LSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            ResidualBlock(9, 32),

            nn.Dropout(0.2),

            ResidualBlock(32, 64),

            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )

        self.norm = nn.LayerNorm(256)

        self.fc = nn.Linear(
            256,
            1
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.conv(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        x = self.norm(x)

        x = self.fc(x)

        return x.squeeze(-1)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNN_LSTM().to(device)

# ============================================================
# LOSSES
# ============================================================

def derivative_loss(y_true, y_pred):

    dy_true = y_true[:, 1:] - y_true[:, :-1]

    dy_pred = y_pred[:, 1:] - y_pred[:, :-1]

    return nn.MSELoss()(dy_true, dy_pred)

def amplitude_loss(y_true, y_pred):

    return (
        y_true.std(dim=1) -
        y_pred.std(dim=1)
    ).pow(2).mean()

def pearson_corr_loss(y_true, y_pred):

    vx = y_true - torch.mean(
        y_true,
        dim=1,
        keepdim=True
    )

    vy = y_pred - torch.mean(
        y_pred,
        dim=1,
        keepdim=True
    )

    corr = torch.sum(vx * vy, dim=1) / (

        torch.sqrt(
            torch.sum(vx ** 2, dim=1) + 1e-8
        )

        *

        torch.sqrt(
            torch.sum(vy ** 2, dim=1) + 1e-8
        )
    )

    return 1 - corr.mean()

def combined_loss(y_true, y_pred):

    mse = nn.MSELoss()(y_true, y_pred)

    d_loss = derivative_loss(
        y_true,
        y_pred
    )

    a_loss = amplitude_loss(
        y_true,
        y_pred
    )

    corr_loss = pearson_corr_loss(
        y_true,
        y_pred
    )

    total = (
        0.25 * mse +
        1.8 * d_loss +
        0.35 * a_loss +
        0.4 * corr_loss
    )

    return total

# ============================================================
# TRAINING
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=10
)

EPOCHS = 120

print("\nTraining Started\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        # Noise injection
        xb = xb + 0.01 * torch.randn_like(xb)

        pred = model(xb)

        loss = combined_loss(
            yb,
            pred
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    old_lr = optimizer.param_groups[0]['lr']

    scheduler.step(avg_loss)

    new_lr = optimizer.param_groups[0]['lr']

    if new_lr != old_lr:

        print(
            f"LR reduced: "
            f"{old_lr:.6f} → {new_lr:.6f}"
        )

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch} "
            f"| Loss {avg_loss:.4f}"
        )

# ============================================================
# EVALUATION
# ============================================================

model.eval()

preds = []
trues = []

with torch.no_grad():

    for xb, yb in test_loader:

        xb = xb.to(device)

        pred = model(xb)

        preds.append(
            pred.cpu().numpy()
        )

        trues.append(
            yb.numpy()
        )

preds = np.concatenate(preds)
trues = np.concatenate(trues)

# ============================================================
# METRICS
# ============================================================

mse = mean_squared_error(
    trues.flatten(),
    preds.flatten()
)

r2 = r2_score(
    trues.flatten(),
    preds.flatten()
)

pearson_corr = pearsonr(
    trues.flatten(),
    preds.flatten()
)[0]

print("\n======================")
print("FINAL RESULTS")
print("======================")

print("MSE               :", mse)
print("R2 Score          :", r2)
print("Pearson Corr      :", pearson_corr)

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    os.path.join(
        SAVE_DIR,
        "cnn_lstm_acc_x_final.pth"
    )
)

# ============================================================
# SAVE PLOTS
# ============================================================

for i in range(len(trues)):

    plt.figure(figsize=(10, 5))

    plt.plot(
        trues[i],
        label="True acc_x"
    )

    plt.plot(
        preds[i],
        label="Predicted acc_x"
    )

    plt.title(f"Test Trial {i+1}")

    plt.legend()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            f"trial_{i+1}.png"
        )
    )

    plt.close()

print("\n✅ All plots saved")

# ============================================================
# DONE
# ============================================================

print("\n======================")
print("ALL RESULTS SAVED IN:")
print(SAVE_DIR)
print("======================")