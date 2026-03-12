import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score

from data_preparation import prepare_dataset, IMU_CHANNELS
from model import EMG2IMU_CNN_TCN


# ==============================
# CONFIG
# ==============================

BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 15

# channels to train
TARGET_CHANNELS = ["acc_x", "gyro_y"]

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ==============================
# LOAD DATA
# ==============================

print("Preparing dataset...")

X_train, Y_train_full, X_test, Y_test_full = prepare_dataset()

print("Train:", X_train.shape)
print("Test:", X_test.shape)


# ==============================
# TRAIN ONE MODEL PER CHANNEL
# ==============================

for TARGET_CHANNEL in TARGET_CHANNELS:

    print("\n=================================")
    print("Training model for:", TARGET_CHANNEL)
    print("=================================")

    target_index = IMU_CHANNELS.index(TARGET_CHANNEL)

    Y_train = Y_train_full[:, target_index:target_index+1]
    Y_test  = Y_test_full[:, target_index:target_index+1]

    # ==============================
    # DATASET
    # ==============================

    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(Y_train)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(Y_test)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    # ==============================
    # MODEL
    # ==============================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EMG2IMU_CNN_TCN(output_channels=1).to(device)

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

    best_val_loss = float("inf")
    early_stop_counter = 0

    history = []

    # ==============================
    # TRAINING LOOP
    # ==============================

    for epoch in range(EPOCHS):

        model.train()

        train_losses = []

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)

            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)


        # ---------- VALIDATION ----------

        model.eval()

        val_losses = []
        preds_all = []
        targets_all = []

        with torch.no_grad():

            for xb, yb in test_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                preds = model(xb)

                loss = criterion(preds, yb)

                val_losses.append(loss.item())

                preds_all.append(preds.cpu().numpy())
                targets_all.append(yb.cpu().numpy())

        val_loss = np.mean(val_losses)

        preds_all = np.concatenate(preds_all)
        targets_all = np.concatenate(targets_all)

        r2 = r2_score(targets_all, preds_all)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"R2: {r2:.4f}"
        )

        history.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "r2": r2
        })


        # ---------- SAVE BEST MODEL ----------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_DIR / f"{TARGET_CHANNEL}_model.pt"
            )

            early_stop_counter = 0

            print("Model saved")

        else:

            early_stop_counter += 1


        # ---------- EARLY STOPPING ----------

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:

            print("Early stopping triggered")
            break


    # ==============================
    # SAVE TRAINING HISTORY
    # ==============================

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        MODEL_DIR / f"{TARGET_CHANNEL}_training_history.csv",
        index=False
    )

    print("\nFinished training for", TARGET_CHANNEL)