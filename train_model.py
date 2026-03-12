# train_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

from data_preparation import prepare_dataset, BASE_DIR
from model import EMG2IMU_CNN_TCN


# ==============================
# CONFIG
# ==============================

BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 15

MODEL_DIR = BASE_DIR / "Models"
MODEL_DIR.mkdir(exist_ok=True)


# ==============================
# LOAD DATA
# ==============================

print("Preparing dataset...")

X, Y = prepare_dataset()

print("Dataset loaded:", X.shape, Y.shape)
X = X.astype(np.float32)
Y = Y.astype(np.float32)

# ==============================
# TRAIN / VALIDATION SPLIT
# ==============================

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(Y_train)
)

val_dataset = TensorDataset(
    torch.tensor(X_val),
    torch.tensor(Y_val)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)


# ==============================
# MODEL
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EMG2IMU_CNN_TCN().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=5,
    factor=0.5
)


# ==============================
# TRAINING LOOP
# ==============================

best_val_loss = float("inf")
early_stop_counter = 0

for epoch in range(EPOCHS):

    # ---------- TRAIN ----------
    model.train()

    train_loss = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        preds = model(xb)

        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(train_loader)


    # ---------- VALIDATION ----------
    model.eval()

    val_loss = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)

            loss = criterion(preds, yb)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)


    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Val Loss: {val_loss:.5f}"
    )


    # ---------- SAVE BEST MODEL ----------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_DIR / "cnn_tcn_model.pt"
        )

        early_stop_counter = 0

        print("Model saved")

    else:

        early_stop_counter += 1


    # ---------- EARLY STOPPING ----------

    if early_stop_counter >= EARLY_STOPPING_PATIENCE:

        print("Early stopping triggered")
        break


print("\nTraining complete")
print("Best validation loss:", best_val_loss)