"""
diagnose.py — why does the CNN-LSTM fail on some subjects?

  python src/diagnose.py --preds results/main
  (writes results/diagnose_channels.png and results/diagnose_preds.png, prints tables)

1. Armband rotation check: for each subject, the share of total EMG envelope on each of the
   8 channels (raw envelopes, before any z-scoring). If the most active channel moves around
   between subjects, the Myo was worn at different rotations -> channel k means a different
   muscle in different people, which breaks subject-independent models.
   Also prints the circular shift that best matches each subject's profile to S01's.
2. Per subject: mean true gyro_y vs mean CNN-LSTM prediction vs template (seed 0), and
   timing lag of the prediction (cross-correlation peak), amplitude ratio.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import build_dataset, FS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/March_trials")
ap.add_argument("--cache", default="results/cache")
ap.add_argument("--preds", default="results/main")
ap.add_argument("--model", default="cnn_lstm")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

D = build_dataset(a.data, "absolute", 0.0, cache_dir=a.cache)
X, subj = D["X"], D["subj"]
S = sorted(set(subj))

# ---------------------------------------------------------------- 1. channel profiles
prof = np.array([X[subj == s].mean((0, 1)) for s in S])
prof = prof / prof.sum(1, keepdims=True)
print("Share of EMG envelope per channel (rows = subjects, cols = ch1..ch8):")
print("      " + " ".join(f"ch{c + 1:<4}" for c in range(8)) + "  top  best_shift_vs_S01  corr")
for i, s in enumerate(S):
    cors = [np.corrcoef(np.roll(prof[i], -k), prof[0])[0, 1] for k in range(8)]
    k = int(np.argmax(cors))
    print(f"{s:>5} " + " ".join(f"{v:.3f}" for v in prof[i]) +
          f"  ch{np.argmax(prof[i]) + 1}   {k:>3}  ({cors[k]:.2f}; no shift {cors[0]:.2f})")

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(prof, aspect="auto", cmap="viridis")
ax.set_xticks(range(8)); ax.set_xticklabels([f"ch{c + 1}" for c in range(8)])
ax.set_yticks(range(len(S))); ax.set_yticklabels(S)
ax.set_title("Share of EMG envelope per channel"); fig.colorbar(im)
fig.tight_layout(); fig.savefig("results/diagnose_channels.png", dpi=120)

# ---------------------------------------------------------------- 2. predictions
pdir = Path(a.preds)
fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharex=True)
t = np.arange(X.shape[1]) / FS
print(f"\nPer subject ({a.model}, seed {a.seed}): lag of prediction vs truth, amplitude ratio")
for ax, s in zip(axes.ravel(), S):
    f = pdir / f"pred_{a.model}_seed{a.seed}_{s}.npz"
    g = pdir / f"pred_template_seed{a.seed}_{s}.npz"
    if not f.exists():
        ax.set_title(f"{s}: no preds"); continue
    P = np.load(f); y, p = P["y"].mean(0), P["p"].mean(0)
    ax.plot(t, y, "k", lw=2, label="true (mean)")
    ax.plot(t, p, "C0", label=a.model)
    if g.exists():
        ax.plot(t, np.load(g)["p"].mean(0), "C1--", label="template")
    # per-trial lag and amplitude
    lags, amps = [], []
    for yi, pi in zip(P["y"], P["p"]):
        yc, pc = yi - yi.mean(), pi - pi.mean()
        cc = np.correlate(pc, yc, "full")
        lags.append((np.argmax(cc) - (len(yc) - 1)) / FS)
        amps.append(pi.std() / (yi.std() + 1e-8))
    print(f"{s}: lag {np.median(lags):+.2f} s (IQR {np.percentile(lags, 25):+.2f}..{np.percentile(lags, 75):+.2f}),"
          f" amplitude ratio {np.median(amps):.2f}")
    ax.set_title(s)
axes[0, 0].legend(fontsize=7)
fig.supxlabel("time (s)"); fig.supylabel("gyro_y (deg/s)")
fig.tight_layout(); fig.savefig("results/diagnose_preds.png", dpi=120)
print("\nsaved results/diagnose_channels.png, results/diagnose_preds.png")
