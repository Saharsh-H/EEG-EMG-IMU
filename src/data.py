"""
data.py — load the March dataset with CORRECT EMG/IMU time alignment.

Layout expected:  data/March_trials/S01/trial_01/{emg_data.csv, imu_wrist_data.csv}

Two alignment modes:
  absolute  (fixed)   : both streams placed on the shared host-PC clock. The window starts
                        when BOTH streams are running (the wrist stream starts ~0.9 s late).
  paper_v1  (old bug) : each stream re-zeroed at its own first sample, as in paper_v1/step1.py.
                        Kept only to show the effect of the bug.

Output per trial: X = 8 EMG envelopes, Y = 6 wrist-IMU channels (raw units: g, deg/s),
both on a uniform FS = 50 Hz grid, N = 240 samples (4.8 s; the shortest overlap is ~4.85 s).

Run directly for a sanity check:
    python src/data.py --root data/March_trials
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

FS = 50.0                      # analysis rate (paper also ended at ~50 Hz after ::2)
WIN_S = 4.8                    # window length in seconds
N = int(round(FS * WIN_S))     # 240 samples
TARGETS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
EMG_COLS = [f"emg{i}" for i in range(1, 9)]


# ---------------------------------------------------------------- time helpers
def _secs(col):
    """'HH:MM:SS.ffffff' -> seconds since midnight, unwrapping a midnight crossing."""
    t = pd.to_datetime(col, format="%H:%M:%S.%f")
    s = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
         + t.dt.microsecond / 1e6).to_numpy(dtype=float)
    for j in np.where(np.diff(s) < -43200)[0]:
        s[j + 1:] += 86400.0
    return s


# ---------------------------------------------------------------- EMG envelope
def emg_envelope(x, fs, bp=(1.0, 40.0), lp=3.0, order=4):
    """Same envelope as the paper: DC removal -> band-pass -> |Hilbert| -> low-pass."""
    x = x - x.mean()
    b, a = butter(order, [bp[0] / (fs / 2), bp[1] / (fs / 2)], btype="band")
    y = np.abs(hilbert(filtfilt(b, a, x)))
    b, a = butter(order, lp / (fs / 2), btype="low")
    return filtfilt(b, a, y)


# ---------------------------------------------------------------- one trial
def load_trial(trial_dir, alignment="absolute", shift=0.0):
    """Returns X (N,8) EMG envelopes, Y (N,6) wrist IMU, info dict.

    shift (s, absolute mode only): pairs EMG(t) with IMU(t + shift). Use for lag scans.
    """
    trial_dir = Path(trial_dir)
    e = pd.read_csv(trial_dir / "emg_data.csv")
    w = pd.read_csv(trial_dir / "imu_wrist_data.csv")
    te, tw = _secs(e["timestamp"]), _secs(w["timestamp"])
    if tw[0] - te[0] < -43200:          # trial crossed midnight between streams
        tw = tw + 86400.0
    elif tw[0] - te[0] > 43200:
        te = te + 86400.0

    # EMG host timestamps arrive in BLE bursts; the device itself samples uniformly,
    # so use a uniform time base between first and last arrival.
    te_u = np.linspace(te[0], te[-1], len(te))
    fs_e = (len(te) - 1) / (te[-1] - te[0])
    env = np.stack([emg_envelope(e[c].to_numpy(float), fs_e) for c in EMG_COLS], axis=1)
    W = w[TARGETS].to_numpy(float)

    if alignment == "absolute":
        tws = tw - shift
        t0 = max(te_u[0], tws[0])
        grid = t0 + np.arange(N) / FS
        if grid[-1] > min(te_u[-1], tws[-1]) + 1e-6:
            raise ValueError(f"overlap too short ({min(te_u[-1], tws[-1]) - t0:.2f}s)")
        X = np.stack([np.interp(grid, te_u, env[:, c]) for c in range(8)], 1)
        Y = np.stack([np.interp(grid, tws, W[:, k]) for k in range(6)], 1)
    elif alignment == "paper_v1":
        grid = np.arange(N) / FS
        X = np.stack([np.interp(grid, te_u - te_u[0], env[:, c]) for c in range(8)], 1)
        Y = np.stack([np.interp(grid, tw - tw[0], W[:, k]) for k in range(6)], 1)
    else:
        raise ValueError(alignment)

    info = dict(offset_s=tw[0] - te[0], emg_dur=te[-1] - te[0], imu_dur=tw[-1] - tw[0])
    return X.astype(np.float32), Y.astype(np.float32), info


# ---------------------------------------------------------------- whole dataset
def build_dataset(root="data/March_trials", alignment="absolute", shift=0.0,
                  cache_dir=None, verbose=True):
    """Returns dict: X (n,N,8), Y (n,N,6), subj (n,), trial (n,), offset (n,)."""
    if cache_dir is not None:
        cache = Path(cache_dir) / f"ds_{alignment}_shift{shift:+.3f}.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            return {k: d[k] for k in d.files}

    Xs, Ys, subj, trial, off, skipped = [], [], [], [], [], []
    for sdir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for tdir in sorted(p for p in sdir.iterdir() if p.is_dir() and p.name.startswith("trial_")):
            try:
                X, Y, info = load_trial(tdir, alignment, shift)
            except Exception as ex:          # noqa: BLE001
                skipped.append(f"{sdir.name}/{tdir.name}: {ex}")
                continue
            Xs.append(X); Ys.append(Y); subj.append(sdir.name)
            trial.append(tdir.name); off.append(info["offset_s"])
    D = dict(X=np.stack(Xs), Y=np.stack(Ys), subj=np.array(subj),
             trial=np.array(trial), offset=np.array(off))
    if verbose:
        print(f"[data] {len(subj)} trials, {len(set(subj))} subjects, alignment={alignment}, "
              f"shift={shift:+.2f}s, skipped={len(skipped)}")
        for s in skipped:
            print("   skipped:", s)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez(cache, **D)
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/March_trials")
    args = ap.parse_args()
    D = build_dataset(args.root, "absolute")
    print("X", D["X"].shape, "Y", D["Y"].shape)
    print("wrist-stream start delay (s): min %.2f  mean %.2f  max %.2f"
          % (D["offset"].min(), D["offset"].mean(), D["offset"].max()))
    for s in sorted(set(D["subj"])):
        m = D["subj"] == s
        print(f"  {s}: {m.sum():2d} trials   gyro_y range per trial (deg/s) ~ "
              f"{np.median(np.ptp(D['Y'][m][:, :, 4], axis=1)):.0f}")
