"""
models.py — sequence-to-sequence models (one output per time step) and losses.

  cnn_lstm : the paper's model (Purva)       Conv(k15)x2 -> 2-layer LSTM(256) -> Linear
  cnn      : ablation, conv block only        Conv(k15)x2 -> Linear
  lstm     : ablation, LSTM only              2-layer LSTM(256) -> Linear
  cnn_tcn  : Saharsh's CNN-TCN, adapted to output every time step
             (original paper_v1/cnn_tcn predicted only the last sample of a window)
"""
import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, in_ch=8, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 15, padding=7), nn.BatchNorm1d(64), nn.ReLU())
        self.lstm = nn.LSTM(64, hidden, layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (B, T, C)
        h = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        h, _ = self.lstm(h)
        return self.fc(h).squeeze(-1)           # (B, T)


class CNNOnly(nn.Module):
    def __init__(self, in_ch=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 15, padding=7), nn.BatchNorm1d(64), nn.ReLU())
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        h = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.fc(h).squeeze(-1)


class LSTMOnly(nn.Module):
    def __init__(self, in_ch=8, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h).squeeze(-1)


class TemporalBlock(nn.Module):
    """Same block as paper_v1/cnn_tcn/model.py."""
    def __init__(self, cin, cout, k, dil):
        super().__init__()
        pad = (k - 1) * dil // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=pad, dilation=dil), nn.BatchNorm1d(cout), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(cout, cout, k, padding=pad, dilation=dil), nn.BatchNorm1d(cout), nn.ReLU(), nn.Dropout(0.2))
        self.down = None if cin == cout else nn.Sequential(nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout))
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        return self.relu(self.net(x) + res)


class CNNTCN(nn.Module):
    def __init__(self, in_ch=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2))
        self.tcn = nn.Sequential(*[TemporalBlock(64, 64, 3, d) for d in (1, 2, 4, 8)])
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.tcn(self.cnn(x.permute(0, 2, 1))).permute(0, 2, 1)
        return self.fc(h).squeeze(-1)


def make_model(name, in_ch):
    return {"cnn_lstm": CNNLSTM, "cnn": CNNOnly, "lstm": LSTMOnly, "cnn_tcn": CNNTCN}[name](in_ch=in_ch)


# ---------------------------------------------------------------- losses
def _pearson_loss(y, p):
    vy, vp = y - y.mean(1, keepdim=True), p - p.mean(1, keepdim=True)
    r = (vy * vp).sum(1) / (torch.sqrt((vy ** 2).sum(1) + 1e-8) * torch.sqrt((vp ** 2).sum(1) + 1e-8))
    return 1 - r.mean()


def composite_loss(y, p):
    """Paper loss: 0.4*MSE + 1.0*derivative + 0.2*amplitude(std) + 0.4*(1 - Pearson)."""
    mse = nn.functional.mse_loss(p, y)
    deriv = nn.functional.mse_loss(p[:, 1:] - p[:, :-1], y[:, 1:] - y[:, :-1])
    amp = ((y.std(1) - p.std(1)) ** 2).mean()
    return 0.4 * mse + 1.0 * deriv + 0.2 * amp + 0.4 * _pearson_loss(y, p)


def mse_loss(y, p):
    return nn.functional.mse_loss(p, y)


LOSSES = {"composite": composite_loss, "mse": mse_loss}
