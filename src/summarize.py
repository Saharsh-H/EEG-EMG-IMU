"""
summarize.py — turn one or more results.csv files into the paper table.

  python src/summarize.py results/main/results.csv results/mse/results.csv
  python src/summarize.py results/*/results.csv --ref "cnn_lstm|composite"

Per configuration: seeds are averaged per test subject first, then mean ± std ACROSS SUBJECTS.
Wilcoxon signed-rank (paired over subjects) of every config vs the reference config, on R2.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--ref", default="cnn_lstm|composite", help="reference config 'model|loss'")
ap.add_argument("--per_subject", action="store_true", help="also print skill per test subject")
ap.add_argument("--out", default=None, help="optional path for summary csv")
a = ap.parse_args()

df = pd.concat([pd.read_csv(f) for f in a.csv], ignore_index=True)
for c, d in [("residual", 0), ("wd", 0.0), ("base", ""), ("aug", 0), ("rel_input", 0)]:
    if c not in df:
        df[c] = d
df["base"] = df["base"].fillna("").astype(str)
df["config"] = df["model"] + "|" + df["loss"].astype(str) \
    + np.where(df["base"] != "", "|on:" + df["base"], np.where(df["residual"] > 0, "|res", "")) \
    + np.where(df["wd"] > 0, "|wd" + df["wd"].astype(str), "") \
    + np.where(df["aug"].fillna(0) > 0, "|aug", "") + np.where(df["rel_input"].fillna(0) > 0, "|rel", "")
keys = ["config", "alignment", "split", "norm", "target", "channels"]

if a.per_subject:
    ps = df.groupby(["config", "test_subject"])["skill"].mean().unstack(0).round(3)
    print(ps.to_string(), "\n")
mets = ["r2", "rmse", "pearson", "skill"] + (["shift_r"] if "shift_r" in df and df["shift_r"].notna().any() else [])
per_subj = df.groupby(keys + ["test_subject"])[mets].mean().reset_index()
rows = []
for k, g in per_subj.groupby(keys):
    row = dict(zip(keys, k), n_subj=len(g))
    for m in mets:
        row[m] = f"{g[m].mean():.3f} ± {g[m].std():.3f}" if g[m].notna().any() else "-"
    ref = per_subj[(per_subj["config"] == a.ref) & np.all([per_subj[c] == v for c, v in zip(keys[1:], k[1:])], 0)]
    if len(ref) and k[0] != a.ref:
        mrg = g.merge(ref, on="test_subject", suffixes=("", "_ref"))
        if len(mrg) >= 5:
            d = mrg["r2_ref"] - mrg["r2"]
            row[f"wins_of_ref"] = f"{int((d > 0).sum())}/{len(d)}"
            row["p_wilcoxon"] = round(wilcoxon(d).pvalue, 4) if np.any(d != 0) else 1.0
    rows.append(row)

S = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(S.to_string(index=False))
print("\nskill = 1 - SSE(model)/SSE(template): >0 means better than the average-movement template.")
print("p_wilcoxon: reference vs this config across subjects (n=10 -> smallest possible p ≈ 0.002).")
if a.out:
    S.to_csv(a.out, index=False)
