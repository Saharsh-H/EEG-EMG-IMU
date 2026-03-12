import torch
import torch.nn as nn


# ==============================
# TCN Residual Block
# ==============================

class TemporalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation):

        super().__init__()

        padding1 = (kernel_size - 1) * dilation // 2
        padding2 = (kernel_size - 1) * (dilation * 2) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding1,
            dilation=dilation
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding2,
            dilation=dilation * 2
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


# ==============================
# CNN + TCN Network
# ==============================

class EMG2IMU_CNN_TCN(nn.Module):

    def __init__(self, output_channels=6):

        super().__init__()

        # ------------------------------
        # CNN Feature Extraction
        # ------------------------------

        self.cnn = nn.Sequential(

            nn.Conv1d(8, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Dropout(0.2)
        )

        # ------------------------------
        # TCN Blocks
        # ------------------------------

        self.tcn = nn.Sequential(

            TemporalBlock(64, 64, kernel_size=3, dilation=1),
            TemporalBlock(64, 64, kernel_size=3, dilation=2),
            TemporalBlock(64, 64, kernel_size=3, dilation=4),
            TemporalBlock(64, 64, kernel_size=3, dilation=8)

        )

        # ------------------------------
        # Decoder
        # ------------------------------

        self.fc = nn.Sequential(

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_channels)
        )


    def forward(self, x):

        # x shape
        # (batch, time, channels)

        x = x.permute(0, 2, 1)      # (B, 8, 200)

        x = self.cnn(x)             # (B, 64, 200)

        x = self.tcn(x)             # (B, 64, 200)

        x = x[:, :, -1]             # last timestep

        out = self.fc(x)            # (B, output_channels)

        return out