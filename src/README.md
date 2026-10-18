# src/ — corrected pipeline (v2)

Run everything from the repo root (`EEG-EMG-IMU/`). Needs: numpy, pandas, scipy, scikit-learn, torch.

| file | what it does |
|---|---|
| `data.py` | loads `data/March_trials/S01…S10`, aligns EMG and wrist IMU on the **real timestamps** (fixes the ~0.9 s offset bug), 50 Hz, 4.8 s per trial |
| `models.py` | CNN-LSTM (paper), CNN-only, LSTM-only, CNN-TCN (seq2seq version), paper loss + MSE |
| `run_loso.py` | leave-one-subject-out training/testing, baselines (template, ridge), metrics in deg/s |
| `summarize.py` | mean ± std across subjects, Wilcoxon tests → paper table |

## Run order

```bash
# 0. sanity check (seconds) — should print 270 trials, 10 subjects, delay ≈ 0.83–1.19 s
python src/data.py

# 1. THE deciding experiment: does the CNN-LSTM beat the template on unseen people?
python src/run_loso.py --models template ridge cnn_lstm --out results/main --save_preds

# 2. loss ablation
python src/run_loso.py --models cnn_lstm --loss mse --out results/mse

# 3. architecture baselines / ablations
python src/run_loso.py --models cnn lstm cnn_tcn --out results/ablation

# 4. old paper protocol (bugged alignment, random split, per-trial z) — the "before" row
python src/run_loso.py --alignment paper_v1 --split random --norm per_trial \
       --models template cnn_lstm --fixed_epochs 180 --out results/paper_protocol

# 5. table
python src/summarize.py results/main/results.csv results/mse/results.csv results/ablation/results.csv
python src/summarize.py results/paper_protocol/results.csv
```

Each run prints one line per fold; `results.csv` is saved after every fold, so you can stop and look anytime.
Key column: **skill** = 1 − SSE(model)/SSE(template). Positive = the model beats "predict the average movement".
