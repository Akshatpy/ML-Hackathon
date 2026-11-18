"""Model comparison: 5-fold trip-grouped CV on train + evaluation on student_test.

Every model sees the same training subsample: a fixed share of fixes, drawn per fold,
always taking all candidates of a chosen fix so rankers get complete groups.
Usage:  python -m src.train --models logreg gnb linsvm rf lgbm xgb mlp lgbm_rank
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from .config import load_config
from .data_prep import list_trips
from .features import FEATURES
from .metrics import evaluate, fix_groups, load_arcs
from .models import MODEL_INFO, RANKERS, make_model, predict_score


def sample_fixes(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Keep `frac` of fixes (whole candidate groups), preserving row order."""
    g = fix_groups(df)
    rng = np.random.default_rng(seed)
    keep_group = rng.random(g.max() + 1) < frac
    return df[keep_group[g]]


def fit(name: str, train_df: pd.DataFrame, seed: int, **overrides):
    model = make_model(name, seed, **overrides)
    X, y = train_df[FEATURES].to_numpy(np.float32), train_df.label.to_numpy()
    t0 = time.time()
    if name in RANKERS:
        sizes = np.bincount(fix_groups(train_df))
        model.fit(X, y, group=sizes)
    else:
        model.fit(X, y)
    return model, time.time() - t0


def score(model, name, df):
    X = df[FEATURES].to_numpy(np.float32)
    t0 = time.time()
    s = predict_score(model, name, X)
    return s, time.time() - t0


def run(models: list[str], frac: float):
    cfg = load_config()
    seed = cfg["seed"]
    cache = cfg["paths"]["artifacts"] / "cache"
    out = cfg["paths"]["reports"] / "models"
    mdir = cfg["paths"]["artifacts"] / "models"
    out.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    train = pd.read_pickle(cache / "features_train.pkl")
    test = pd.read_pickle(cache / "features_student_test.pkl")
    # train only on fixes whose label we trust; evaluate on all rows, metrics mask internally
    train_valid = train[train.valid].reset_index(drop=True)
    dirs = {p.name: p for d in (cfg["paths"]["train_dir"], cfg["paths"]["student_test_dir"]) for p in list_trips(d)}
    arcs_by_trip = load_arcs(dirs)

    fold_rows, summary = [], []
    results_path = out / "model_results.csv"
    prev = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()

    for name in models:
        is_prob = name not in RANKERS
        print(f"== {name}", flush=True)
        for k in sorted(train.fold.unique()):
            tr = sample_fixes(train_valid[train_valid.fold != k], frac, seed + k)
            va = train[train.fold == k]
            model, t_fit = fit(name, tr, seed)
            s, t_pred = score(model, name, va)
            m = evaluate(va, s, arcs_by_trip, has_probability=is_prob)
            m.update(model=name, fold=k, fit_s=t_fit, pred_s=t_pred, n_train_rows=len(tr),
                     train_top1=evaluate(tr, score(model, name, tr)[0], None, is_prob)["top1_acc"])
            fold_rows.append(m)
            print(f"   fold {k}: top1={m['top1_acc']:.4f} f1_top1={m['macro_f1_top1']:.4f} "
                  f"ll={m['log_loss']:.4f} cont={m['route_continuity']:.4f} "
                  f"train_top1={m['train_top1']:.4f} fit={t_fit:.0f}s", flush=True)

        # refit on all train folds (same sampling share) -> student_test
        tr = sample_fixes(train_valid, frac, seed)
        model, t_fit = fit(name, tr, seed)
        s, t_pred = score(model, name, test)
        mt = evaluate(test, s, arcs_by_trip, has_probability=is_prob)
        joblib.dump(model, mdir / f"{name}.joblib")
        np.save(mdir / f"{name}_student_test_scores.npy", s)

        cv = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        row = {
            "model": name, "family": MODEL_INFO[name][0], "config": MODEL_INFO[name][1],
            "cv_top1_mean": cv.top1_acc.mean(), "cv_top1_std": cv.top1_acc.std(),
            "cv_macro_f1_top1": cv.macro_f1_top1.mean(), "cv_macro_f1_top1_std": cv.macro_f1_top1.std(),
            "cv_macro_f1_thr05": cv.macro_f1_thr05.mean(), "cv_log_loss": cv.log_loss.mean(),
            "cv_route_continuity": cv.route_continuity.mean(), "cv_train_top1": cv.train_top1.mean(),
            "test_top1": mt["top1_acc"], "test_macro_f1_top1": mt["macro_f1_top1"],
            "test_macro_f1_thr05": mt["macro_f1_thr05"], "test_precision_top1": mt["precision_top1"],
            "test_recall_top1": mt["recall_top1"], "test_log_loss": mt["log_loss"],
            "test_roc_auc": mt["roc_auc"], "test_route_continuity": mt["route_continuity"],
            "fit_s_full": t_fit, "infer_ms_per_1k_fixes": 1000 * t_pred / test.groupby(["trip_id", "obs"]).ngroups * 1000,
        }
        summary.append(row)
        print(f"   student_test: top1={mt['top1_acc']:.4f} f1_top1={mt['macro_f1_top1']:.4f} "
              f"ll={mt['log_loss']:.4f} cont={mt['route_continuity']:.4f}", flush=True)

        if name == "mlp":
            mlp = model[-1]
            hist = pd.DataFrame({"epoch": np.arange(1, len(mlp.loss_curve_) + 1),
                                 "train_loss": mlp.loss_curve_,
                                 "val_accuracy": mlp.validation_scores_})
            hist.to_csv(out / "mlp_history.csv", index=False)

        # persist after every model so a crash doesn't lose finished runs
        folds_df = pd.DataFrame(fold_rows)
        fpath = out / "model_folds.csv"
        old_f = pd.read_csv(fpath) if fpath.exists() else pd.DataFrame()
        if len(old_f):
            old_f = old_f[~old_f.model.isin(folds_df.model.unique())]
        pd.concat([old_f, folds_df]).to_csv(fpath, index=False)
        res = pd.DataFrame(summary)
        merged = pd.concat([prev[~prev.model.isin(res.model)] if len(prev) else prev, res])
        merged.to_csv(results_path, index=False)

    (out / "train_settings.json").write_text(json.dumps(
        {"sample_frac": frac, "seed": seed, "features": FEATURES, "models": MODEL_INFO}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_INFO))
    ap.add_argument("--frac", type=float, default=0.35)
    args = ap.parse_args()
    run(args.models, args.frac)
