"""
run_loso.py — train/evaluate models with subject-independent (leave-one-subject-out) testing.

Each fold: 1 test subject, 1 validation subject (the next one; used for early stopping and
ridge alpha), 8 training subjects. Nothing from the test subject is used for training.

Normalisation (--norm):
  deploy    (default) EMG envelopes z-scored per SUBJECT and channel using only that subject's
            EMG (an unlabelled calibration a real device could do); target scaled with
            training-set mean/std; predictions converted back to deg/s for scoring.
  per_trial the paper's scheme (each trial z-scored with its own stats, inputs AND target).

Metrics per test subject (all its trials pooled), in target units:
  r2, rmse, pearson, and skill = 1 - SSE(model)/SSE(template)
  skill > 0  means the model beats "predict the average training movement".

Examples (from the repo root):
  python src/run_loso.py --models template ridge cnn_lstm --out results/main
  python src/run_loso.py --models cnn_lstm --loss mse --out results/mse
  python src/run_loso.py --alignment paper_v1 --split random --norm per_trial \
         --models template cnn_lstm --fixed_epochs 180 --out results/paper_protocol
  python src/run_loso.py --models template oracle oracle_shift oracle_scale emg_align --out results/align

Non-network models:
  template      average training movement (no EMG)
  oracle*       CEILINGS that use the true test signal (best shift/scale of the template)
  ridge         lagged-EMG linear regression
  emg_align     EMG-timed template: EMG features -> per-trial shift, scale, offset of the template
  emg_align_b   same, blended with the plain template (weight chosen on validation)
  emg_shift     EMG lag only -> shift only;  emg_shift_b  same, blended
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from data import build_dataset, TARGETS, FS
from models import make_model, LOSSES

NETS = {"cnn_lstm", "cnn", "lstm", "cnn_tcn"}
ORACLES = {"oracle": (True, True), "oracle_shift": (True, False), "oracle_scale": (False, True)}


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/March_trials")
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["template", "ridge", "cnn_lstm"])
    ap.add_argument("--loss", default="composite", choices=list(LOSSES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--target", default="gyro_y", choices=TARGETS)
    ap.add_argument("--channels", nargs="+", type=int, default=list(range(1, 9)), help="1-based EMG channels")
    ap.add_argument("--alignment", default="absolute", choices=["absolute", "paper_v1"])
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--split", default="loso", choices=["loso", "random", "within"],
                    help="within: 5-fold over the test subject's own trials; training = other subjects + its other folds")
    ap.add_argument("--norm", default="deploy", choices=["deploy", "per_trial"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--fixed_epochs", type=int, default=0, help=">0: no early stopping, train exactly this many epochs")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--residual", action="store_true",
                    help="nets/ridge learn the deviation from the training template (template added back)")
    ap.add_argument("--wd", type=float, default=0.0, help="Adam weight decay")
    return ap.parse_args()


# ---------------------------------------------------------------- folds
def make_folds(subj, split, seed):
    """Yields (name, train_idx, val_idx, test_idx)."""
    subjects = sorted(set(subj))
    idx = np.arange(len(subj))
    if split == "loso":
        for i, s in enumerate(subjects):
            v = subjects[(i + 1) % len(subjects)]
            yield s, idx[(subj != s) & (subj != v)], idx[subj == v], idx[subj == s]
    elif split == "within":
        rng = np.random.RandomState(100 + seed)
        for s in subjects:
            own = rng.permutation(idx[subj == s])
            parts = np.array_split(own, 5)
            for k in range(5):
                te, va = parts[k], parts[(k + 1) % 5]
                tr = np.concatenate([idx[subj != s]] + [parts[j] for j in range(5) if j not in (k, (k + 1) % 5)])
                yield f"{s}_k{k}", tr, va, te
    else:  # random trial-level 80/20 (the paper's protocol); 10% of train held out for validation
        rng = np.random.RandomState(42 + seed)
        p = rng.permutation(idx)
        n_te = int(round(0.2 * len(p)))
        te, rest = p[:n_te], p[n_te:]
        n_va = int(round(0.1 * len(rest)))
        yield f"random{seed}", rest[n_va:], rest[:n_va], te


# ---------------------------------------------------------------- normalisation
class Norm:
    def __init__(self, mode, X, Y, subj, train_idx):
        self.mode = mode
        if mode == "deploy":
            self.xstats = {s: (X[subj == s].mean((0, 1)), X[subj == s].std((0, 1)) + 1e-8) for s in set(subj)}
            self.ymu, self.ysd = Y[train_idx].mean(), Y[train_idx].std() + 1e-8

    def x(self, X, subj):
        if self.mode == "per_trial":
            return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
        out = np.empty_like(X)
        for s in set(subj):
            m = subj == s
            mu, sd = self.xstats[s]
            out[m] = (X[m] - mu) / sd
        return out

    def y(self, Y):
        if self.mode == "per_trial":
            return (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-8)
        return (Y - self.ymu) / self.ysd

    def y_inv(self, Yn):
        # per_trial: targets cannot be un-normalised without the true signal -> score in z units
        return Yn if self.mode == "per_trial" else Yn * self.ysd + self.ymu


# ---------------------------------------------------------------- models
def lag_features(X, max_lag_s=0.5, step_s=0.04):
    """Causal lagged EMG features for ridge regression: (n,T,C) -> (n,T,C*L)."""
    lags = range(0, int(round(max_lag_s * FS)) + 1, max(1, int(round(step_s * FS))))
    feats = []
    for l in lags:
        feats.append(X if l == 0 else np.concatenate([np.repeat(X[:, :1], l, 1), X[:, :-l]], 1))
    return np.concatenate(feats, 2)


def fit_ridge(Xtr, Ytr, Xva, Yva):
    Ftr, Fva = lag_features(Xtr), lag_features(Xva)
    best = None
    for a in [0.1, 1, 10, 100, 1000]:
        m = Ridge(alpha=a).fit(Ftr.reshape(-1, Ftr.shape[2]), Ytr.ravel())
        err = np.mean((m.predict(Fva.reshape(-1, Fva.shape[2])).reshape(Yva.shape) - Yva) ** 2)
        if best is None or err < best[0]:
            best = (err, m)
    model = best[1]

    def predict(X):
        F = lag_features(X)
        return model.predict(F.reshape(-1, F.shape[2])).reshape(X.shape[:2])
    return predict


MAX_SHIFT = int(round(1.0 * FS))        # +-1 s search range for time shifts (samples)


def shift_sig(v, k):
    """Delay v (T,) or (T,C) by k samples (k>0: later), padding with the edge value."""
    out = np.roll(v, k, axis=0)
    if k > 0:
        out[:k] = v[0]
    elif k < 0:
        out[k:] = v[-1]
    return out


def fit_template(tm, y, shift=True, scale=True):
    """Best (k, a, b) so that a*shift(tm,k)+b fits y by least squares. Returns (k, a, b)."""
    T, best = len(tm), None
    for k in (range(-MAX_SHIFT, MAX_SHIFT + 1) if shift else [0]):
        sh = shift_sig(tm, k)
        if scale:
            A = np.stack([sh, np.ones(T)], 1)
            (a_, b_), *_ = np.linalg.lstsq(A, y, rcond=None)
        else:
            a_, b_ = 1.0, float(np.mean(y - sh))
        err = ((a_ * sh + b_ - y) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, k, a_, b_)
    return best[1:]


def warp(tm, k, a, b):
    return a * shift_sig(tm, int(round(k))) + b


def oracle_template(tm, Yte, shift=True, scale=True):
    """CEILING, uses the TRUE signal (not a model): per test trial, the template shifted
    (+-1 s) and/or scaled+offset by least squares to fit the truth.
      oracle        shift + scale + offset
      oracle_shift  shift + offset only (timing)
      oracle_scale  scale + offset only (amplitude, no timing)"""
    return np.array([warp(tm, *fit_template(tm, y, shift, scale)) for y in Yte], dtype=np.float32)


def emg_lag(E_tm, x):
    """Shift (samples) of the mean training EMG envelope E_tm (T,C) that best matches trial x (T,C)."""
    xc = (x - x.mean(0)).ravel()
    best = None
    for k in range(-MAX_SHIFT, MAX_SHIFT + 1):
        s = shift_sig(E_tm, k)
        s = (s - s.mean(0)).ravel()
        c = float(s @ xc) / (np.linalg.norm(s) * np.linalg.norm(xc) + 1e-8)
        if best is None or c > best[0]:
            best = (c, k)
    return best[1]


def align_features(E_tm, X):
    """Per-trial EMG features: EMG timing lag (s) + per-channel mean and peak envelope."""
    return np.array([np.concatenate([[emg_lag(E_tm, x) / FS], x.mean(0), x.max(0)]) for x in X])


# variants of the EMG-timed template: (features, predicted parameters, blend with template)
#   emg_align    all 17 features -> shift, scale, offset
#   emg_align_b  same, output blended with the plain template (weight w chosen on validation)
#   emg_shift    EMG lag only    -> shift only (scale 1, offset 0)
#   emg_shift_b  same, blended
ALIGN = {"emg_align": ("all", "kab", False), "emg_align_b": ("all", "kab", True),
         "emg_shift": ("lag", "k", False), "emg_shift_b": ("lag", "k", True)}


def fit_emg_align(Xtr, Ytr, Xva, Yva, tm, E_tm, feats="all", params="kab", blend=False):
    """EMG-timed template: predict each trial's shift k (and optionally scale a, offset b) from
    EMG features with ridge regression, then warp the average movement accordingly.
    blend: final = w*warped + (1-w)*template, hedging against wrong shift guesses.
    Ridge alpha (and w) are chosen on the validation subject's reconstruction error.
    Returns predict(X) -> (Y_pred, k_pred), training corr(EMG lag, true shift), chosen (alpha, w)."""
    Ftr, Fva = align_features(E_tm, Xtr), align_features(E_tm, Xva)
    if feats == "lag":
        Ftr, Fva = Ftr[:, :1], Fva[:, :1]
    if params == "kab":
        P = np.array([fit_template(tm, y) for y in Ytr])                       # true (k, a, b)
    else:
        P = np.array([fit_template(tm, y, scale=False)[:1] for y in Ytr])      # true k only
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8

    def params_of(ms, F):
        prm = np.stack([m.predict((F - mu) / sd) for m in ms], 1)
        prm[:, 0] = np.clip(prm[:, 0], -MAX_SHIFT, MAX_SHIFT)
        if prm.shape[1] == 1:                                                  # shift only
            prm = np.concatenate([prm, np.ones((len(prm), 1)), np.zeros((len(prm), 1))], 1)
        return prm

    best = None
    for al in [0.1, 1, 10, 100, 1000]:
        ms = [Ridge(alpha=al).fit((Ftr - mu) / sd, P[:, j]) for j in range(P.shape[1])]
        Yw = np.array([warp(tm, *p) for p in params_of(ms, Fva)])
        for w in ([0.0, 0.25, 0.5, 0.75, 1.0] if blend else [1.0]):
            err = np.mean((w * Yw + (1 - w) * tm - Yva) ** 2)
            if best is None or err < best[0]:
                best = (err, ms, al, w)
    _, ms, al, w = best
    r_lag = float(np.corrcoef(Ftr[:, 0], P[:, 0])[0, 1])

    def predict(X):
        prm = params_of(ms, align_features(E_tm, X)[:, :Ftr.shape[1]])
        Yw = np.array([warp(tm, *p) for p in prm])
        return (w * Yw + (1 - w) * tm).astype(np.float32), prm[:, 0]
    return predict, r_lag, (al, w)


def train_net(name, Xtr, Ytr, Xva, Yva, seed, a, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(name, Xtr.shape[2]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    loss_fn = LOSSES[a.loss]
    Xt, Yt = torch.tensor(Xtr, device=dev), torch.tensor(Ytr, device=dev)
    Xv, Yv = torch.tensor(Xva, device=dev), torch.tensor(Yva, device=dev)
    rng = np.random.RandomState(seed)
    n_ep = a.fixed_epochs if a.fixed_epochs > 0 else a.epochs
    best, best_state, bad, ep = np.inf, None, 0, 0
    for ep in range(1, n_ep + 1):
        model.train()
        for b in np.array_split(rng.permutation(len(Xt)), max(1, int(np.ceil(len(Xt) / a.batch)))):
            if len(b) < 2:              # BatchNorm needs >1 sample
                continue
            b = torch.as_tensor(b, device=dev)
            loss = loss_fn(Yt[b], model(Xt[b]))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if a.fixed_epochs > 0:
            continue
        model.eval()
        with torch.no_grad():
            v = torch.mean((model(Xv) - Yv) ** 2).item()
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            return model(torch.tensor(X, device=dev)).cpu().numpy()
    return predict, ep, sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- metrics
def metrics(Yt, Yp, Ytmpl):
    e = (Yp - Yt).ravel()
    sse = float((e ** 2).sum())
    sst = float(((Yt - Yt.mean()) ** 2).sum())
    sse_t = float(((Ytmpl - Yt) ** 2).sum())
    r = np.corrcoef(Yt.ravel(), Yp.ravel())[0, 1] if np.std(Yp) > 0 else np.nan
    return dict(r2=1 - sse / sst, rmse=float(np.sqrt(np.mean(e ** 2))), pearson=float(r), skill=1 - sse / sse_t)


# ---------------------------------------------------------------- main
def main():
    a = get_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")

    D = build_dataset(a.data, a.alignment, a.shift, cache_dir=a.cache)
    ch = [c - 1 for c in a.channels]
    X = D["X"][:, :, ch].astype(np.float32)
    Y = D["Y"][:, :, TARGETS.index(a.target)].astype(np.float32)
    subj = D["subj"]

    rows, csv = [], out / "results.csv"
    seeds_needed = a.seeds if (a.split == "random" or any(m in NETS for m in a.models)) else a.seeds[:1]
    for seed in seeds_needed:
        for fold, tr, va, te in make_folds(subj, a.split, seed):
            nz = Norm(a.norm, X, Y, subj, tr)
            Xn = nz.x(X, subj).astype(np.float32)
            Yn = nz.y(Y).astype(np.float32)
            if a.split == "within":              # subject-specific template from its own training trials
                tidx = tr[subj[tr] == subj[te][0]]
            else:
                tidx = tr
            tmpl_n = Yn[tidx].mean(0, keepdims=True)
            E_tm = Xn[tidx].mean(0)              # mean EMG envelope over the same trials (for emg_align)
            Yt_true = nz.y_inv(Yn[te])
            Ytmpl = np.repeat(nz.y_inv(tmpl_n), len(te), 0)
            off = tmpl_n if a.residual else 0.0       # residual mode: learn Y - template
            Yfit = Yn - off
            for m in a.models:
                if m not in NETS and seed != seeds_needed[0] and a.split == "loso":
                    continue                     # template / ridge are deterministic
                t0 = time.time(); ep, npar = 0, 0
                k_pred = k_true = None
                if m == "template":
                    Pn = np.repeat(tmpl_n, len(te), 0)
                elif m in ORACLES:                   # CEILINGS, use the true signal (not models)
                    Pn = oracle_template(tmpl_n[0], Yn[te], *ORACLES[m])
                elif m in ALIGN:                     # EMG-timed template (<= 3 numbers per trial)
                    pred, r_tr, (al, w) = fit_emg_align(Xn[tr], Yn[tr], Xn[va], Yn[va], tmpl_n[0], E_tm,
                                                        *ALIGN[m])
                    Pn, k_pred = pred(Xn[te])
                    k_true = np.array([fit_template(tmpl_n[0], y)[0] for y in Yn[te]])
                    print(f"    {m}: train corr(EMG lag, true shift) {r_tr:+.2f}  alpha {al}  blend w {w}",
                          flush=True)
                elif m == "ridge":
                    Pn = fit_ridge(Xn[tr], Yfit[tr], Xn[va], Yfit[va])(Xn[te]) + off
                else:
                    pred, ep, npar = train_net(m, Xn[tr], Yfit[tr], Xn[va], Yfit[va], seed, a, dev)
                    Pn = pred(Xn[te]) + off
                P = nz.y_inv(Pn)
                # per-subject scores (for random split, the test set mixes subjects)
                for s in sorted(set(subj[te])):
                    k = subj[te] == s
                    # how well the predicted time shift tracks the true one (emg_align only)
                    sh_r = (float(np.corrcoef(k_pred[k], k_true[k])[0, 1])
                            if k_pred is not None and k.sum() > 2 and np.std(k_true[k]) > 0 else np.nan)
                    row = dict(model=m, loss=a.loss if m in NETS else "-", seed=seed, fold=fold, test_subject=s,
                               n_trials=int(k.sum()), alignment=a.alignment, split=a.split, norm=a.norm,
                               target=a.target, channels=" ".join(map(str, a.channels)), epochs=ep,
                               residual=int(a.residual and (m in NETS or m == "ridge")),
                               wd=a.wd if m in NETS else 0.0, shift_r=sh_r,
                               params=npar, secs=round(time.time() - t0, 1),
                               **metrics(Yt_true[k], P[k], Ytmpl[k]))
                    rows.append(row)
                if a.save_preds:
                    np.savez(out / f"pred_{m}_seed{seed}_{fold}.npz", y=Yt_true, p=P,
                             subj=subj[te], trial=D["trial"][te])
                last = [r for r in rows if r["model"] == m and r["fold"] == fold and r["seed"] == seed]
                print(f"seed {seed} fold {fold:>8} {m:>9}: R2 {np.mean([r['r2'] for r in last]):.3f}  "
                      f"skill {np.mean([r['skill'] for r in last]):+.3f}  ({time.time() - t0:.0f}s, ep {ep})",
                      flush=True)
                pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\nsaved {csv}\nnext: python src/summarize.py {csv}")


if __name__ == "__main__":
    main()
