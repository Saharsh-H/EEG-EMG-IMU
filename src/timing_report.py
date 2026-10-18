"""
timing_report.py — where does the timing error of the EMG-timed template come from?

Reads the per-trial shifts.csv written by run_loso.py for emg_* models:
  k_true_s   best time shift of the template for that trial (from the TRUE wrist signal), s
  k_pred_s   shift predicted from EMG, s
  emg_lag_s  raw EMG lag feature (shift of the mean EMG envelope that best matches the trial), s

  python src/timing_report.py results/align3/shifts.csv
  python src/timing_report.py results/align3/shifts.csv --model emg_time

Per subject: spread of the true shift, corr(EMG lag, true shift), corr(pred, true),
mean error (= a subject-specific delay a one-off calibration could remove) and the
remaining within-subject error SD.
Summary: total error = between-subject part (calibratable) + within-subject part.
The within-subject SD of (true shift - EMG lag) is an upper bound on per-trial EMG/IMU
clock jitter plus trial-to-trial variation of the electromechanical delay.
"""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--model", default=None, help="default: every emg_* model in the file")
a = ap.parse_args()

df = pd.read_csv(a.csv)
models = [a.model] if a.model else sorted(df["model"].unique())
for m in models:
    d = df[df["model"] == m].copy()
    d = d.groupby(["test_subject", "trial"])[["k_true_s", "k_pred_s", "emg_lag_s"]].mean().reset_index()
    d["err"] = d["k_pred_s"] - d["k_true_s"]
    d["gap"] = d["k_true_s"] - d["emg_lag_s"]          # true movement shift minus EMG lag
    rows = []
    for s, g in d.groupby("test_subject"):
        rows.append(dict(subject=s, n=len(g),
                         sd_true_ms=1000 * g["k_true_s"].std(),
                         r_lag_true=g[["emg_lag_s", "k_true_s"]].corr().iloc[0, 1],
                         r_pred_true=g[["k_pred_s", "k_true_s"]].corr().iloc[0, 1],
                         mean_err_ms=1000 * g["err"].mean(),
                         sd_err_ms=1000 * g["err"].std(),
                         mean_gap_ms=1000 * g["gap"].mean(),
                         sd_gap_ms=1000 * g["gap"].std()))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n=== {m} ===")
    print(R.round(2).to_string(index=False))

    tot = np.sqrt(np.mean(d["err"] ** 2))
    between = np.sqrt(np.mean(d.groupby("test_subject")["err"].transform("mean") ** 2))
    within = np.sqrt(np.mean((d["err"] - d.groupby("test_subject")["err"].transform("mean")) ** 2))
    print(f"\nshift error RMS {1000*tot:.0f} ms = between-subject {1000*between:.0f} ms (fixed per-person "
          f"offset, removable by calibration) + within-subject {1000*within:.0f} ms  (in quadrature)")
    print(f"true shift SD (what a perfect timer must explain): {1000*d['k_true_s'].std():.0f} ms overall; "
          f"pooled corr(pred, true) {d[['k_pred_s','k_true_s']].corr().iloc[0,1]:.2f}")
    print(f"within-subject SD of (true shift - EMG lag): median {R['sd_gap_ms'].median():.0f} ms "
          f"[{R['sd_gap_ms'].min():.0f}-{R['sd_gap_ms'].max():.0f}]  "
          f"(upper bound on per-trial sync jitter + delay variability)")
