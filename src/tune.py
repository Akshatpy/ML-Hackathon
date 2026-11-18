"""Bayesian hyperparameter optimisation of the selected LightGBM classifier.

Sampler: Optuna TPE (tree-structured Parzen estimator), a sequential model-based
(Bayesian) optimiser. Objective: mean observation-level top-1 accuracy over the same
5 trip-grouped folds and the same 35% fix sample used in the model comparison, so the
tuned and untuned numbers are directly comparable. Student-test is not touched until
the search is finished.

Usage:
  python -m src.tune --trials 40          # search (resumes if the study exists)
  python -m src.tune --finalise           # refit best params, compare with untuned, save model
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from .config import load_config
from .data_prep import list_trips
from .metrics import evaluate, load_arcs
from .train import fit, sample_fixes, score

STUDY = "lgbm_top1"

# Ranges are centred on the untuned config (63 leaves, lr 0.05, 400 trees, min_child 50) and
# widened toward regularisation, because the untuned model fit its training sample 5 pts
# better than held-out trips (0.961 vs 0.908).
def suggest(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


class Data:
    def __init__(self, cfg):
        cache = cfg["paths"]["artifacts"] / "cache"
        self.train = pd.read_pickle(cache / "features_train.pkl")
        self.test = pd.read_pickle(cache / "features_student_test.pkl")
        self.train_valid = self.train[self.train.valid].reset_index(drop=True)
        dirs = {p.name: p for d in (cfg["paths"]["train_dir"], cfg["paths"]["student_test_dir"])
                for p in list_trips(d)}
        self.arcs = load_arcs(dirs)
        self.folds = sorted(self.train.fold.unique())


def objective_factory(data: Data, frac: float, seed: int):
    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        scores, f1s, lls = [], [], []
        t0 = time.time()
        for step, k in enumerate(data.folds):
            tr = sample_fixes(data.train_valid[data.train_valid.fold != k], frac, seed + k)
            va = data.train[data.train.fold == k]
            model, _ = fit("lgbm", tr, seed, **params)
            s, _ = score(model, "lgbm", va)
            m = evaluate(va, s, None)
            scores.append(m["top1_acc"]); f1s.append(m["macro_f1_top1"]); lls.append(m["log_loss"])
            trial.report(float(np.mean(scores)), step)
            # pruning on the running mean; folds differ by ~3 pts so never prune after 1 fold
            if step >= 1 and trial.should_prune():
                trial.set_user_attr("folds_done", step + 1)
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_top1_std", float(np.std(scores, ddof=1)))
        trial.set_user_attr("cv_macro_f1_top1", float(np.mean(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("fold_top1", [round(x, 5) for x in scores])
        trial.set_user_attr("secs", round(time.time() - t0, 1))
        return float(np.mean(scores))
    return objective


def run_search(n_trials: int, frac: float):
    cfg = load_config()
    out = cfg["paths"]["reports"] / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    data = Data(cfg)
    study = optuna.create_study(
        study_name=STUDY, storage=f"sqlite:///{(out / 'optuna_lgbm.db').as_posix()}",
        direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    if len(study.trials) == 0:
        # trial 0 = the untuned configuration, so the search provably starts from the baseline
        study.enqueue_trial({"num_leaves": 63, "min_child_samples": 50, "learning_rate": 0.05,
                             "n_estimators": 400, "colsample_bytree": 0.8, "subsample": 0.8,
                             "reg_lambda": 1e-3, "reg_alpha": 1e-3})
    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(objective_factory(data, frac, seed), n_trials=remaining, gc_after_trial=True)
    export(study, out)


def export(study: optuna.Study, out):
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration", "user_attrs"))
    df.to_csv(out / "optuna_trials.csv", index=False)
    done = df[df.state == "COMPLETE"].sort_values("value", ascending=False)
    print(done.head(8).drop(columns=["duration"]).round(4).to_string(index=False))
    print("pruned:", int((df.state == "PRUNED").sum()), "| complete:", len(done))

    from optuna.visualization.matplotlib import (plot_optimization_history,
                                                 plot_param_importances)
    figs = out.parent / "figures"
    ax = plot_optimization_history(study)
    ax.set_title("Bayesian optimisation history (CV top-1, LightGBM)")
    ax.figure.set_size_inches(8, 4.5); ax.figure.tight_layout()
    ax.figure.savefig(figs / "tuning_history.png", dpi=120); plt.close(ax.figure)
    try:
        ax = plot_param_importances(study)
        ax.figure.set_size_inches(7, 4.5); ax.figure.tight_layout()
        ax.figure.savefig(figs / "tuning_param_importance.png", dpi=120); plt.close(ax.figure)
    except Exception as exc:  # fANOVA needs enough completed trials
        print("param importance skipped:", exc)


def finalise(frac: float):
    """Refit untuned and tuned LightGBM on identical data, score both on student-test,
    then fit the deployment model (tuned params, every valid training fix)."""
    cfg = load_config()
    seed = cfg["seed"]
    out = cfg["paths"]["reports"] / "tuning"
    mdir = cfg["paths"]["artifacts"] / "models"
    study = optuna.load_study(study_name=STUDY, storage=f"sqlite:///{(out / 'optuna_lgbm.db').as_posix()}")
    best = study.best_trial
    data = Data(cfg)
    n_fix = data.test.groupby(["trip_id", "obs"]).ngroups
    untuned = pd.read_csv(cfg["paths"]["reports"] / "models" / "model_results.csv").set_index("model").loc["lgbm"]

    rows = [{
        "model": "LightGBM untuned", "cv_top1": untuned.cv_top1_mean, "cv_top1_std": untuned.cv_top1_std,
        "cv_macro_f1_top1": untuned.cv_macro_f1_top1, "cv_log_loss": untuned.cv_log_loss,
        "test_top1": untuned.test_top1, "test_macro_f1_top1": untuned.test_macro_f1_top1,
        "test_log_loss": untuned.test_log_loss, "test_route_continuity": untuned.test_route_continuity,
        "fit_s": untuned.fit_s_full, "infer_ms_per_1k_fixes": untuned.infer_ms_per_1k_fixes,
        "train_rows": "35% of fixes"}]

    for tag, share in [("LightGBM tuned", frac), ("LightGBM tuned, all fixes (deployed)", 1.0)]:
        tr = data.train_valid if share >= 1.0 else sample_fixes(data.train_valid, share, seed)
        model, t_fit = fit("lgbm", tr, seed, **best.params)
        s, t_pred = score(model, "lgbm", data.test)
        m = evaluate(data.test, s, data.arcs)
        rows.append({
            "model": tag, "cv_top1": best.value if share < 1 else np.nan,
            "cv_top1_std": best.user_attrs.get("cv_top1_std") if share < 1 else np.nan,
            "cv_macro_f1_top1": best.user_attrs.get("cv_macro_f1_top1") if share < 1 else np.nan,
            "cv_log_loss": best.user_attrs.get("cv_log_loss") if share < 1 else np.nan,
            "test_top1": m["top1_acc"], "test_macro_f1_top1": m["macro_f1_top1"],
            "test_log_loss": m["log_loss"], "test_route_continuity": m["route_continuity"],
            "fit_s": t_fit, "infer_ms_per_1k_fixes": 1e6 * t_pred / n_fix,
            "train_rows": f"{len(tr):,} rows"})
        if share >= 1.0:
            joblib.dump(model, mdir / "lgbm_tuned_full.joblib")
            np.save(mdir / "lgbm_tuned_full_student_test_scores.npy", s)
        else:
            np.save(mdir / "lgbm_tuned_student_test_scores.npy", s)

    cmp = pd.DataFrame(rows)
    cmp.to_csv(out / "tuned_vs_untuned.csv", index=False)
    (out / "best_params.json").write_text(json.dumps(
        {"trial": best.number, "cv_top1": best.value, "params": best.params,
         "n_trials": len(study.trials),
         "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials)}, indent=2))
    print(cmp.round(4).to_string(index=False))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--frac", type=float, default=0.35)
    ap.add_argument("--finalise", action="store_true")
    args = ap.parse_args()
    if args.finalise:
        finalise(args.frac)
    else:
        run_search(args.trials, args.frac)
