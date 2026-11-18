"""Comparison plots for the model phase, built only from saved experiment outputs.
Usage: python -m src.model_report
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .config import load_config
from .metrics import fix_groups, top1_rows


def main():
    cfg = load_config()
    rep = cfg["paths"]["reports"] / "models"
    figs = cfg["paths"]["reports"] / "figures"
    res = pd.read_csv(rep / "model_results.csv").sort_values("cv_top1_mean", ascending=False)
    folds = pd.read_csv(rep / "model_folds.csv")
    names = res.model.tolist()
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.bar(x, res.cv_macro_f1_top1, yerr=res.cv_macro_f1_top1_std, capsize=4, color="#4c72b0")
    ax.set_xticks(x, names, rotation=30); ax.set_ylim(0.5, 1); ax.set_title("CV macro F1 (top-1 decision), mean +- std over 5 folds")

    ax = axes[0, 1]
    w = 0.38
    ax.bar(x - w / 2, res.test_top1, w, label="top-1 accuracy", color="#55a868")
    ax.bar(x + w / 2, res.test_macro_f1_top1, w, label="macro F1", color="#c44e52")
    ax.set_xticks(x, names, rotation=30); ax.set_ylim(0.5, 1); ax.legend(); ax.set_title("Student-test performance")

    ax = axes[1, 0]
    ll = res.dropna(subset=["test_log_loss"])
    xi = np.arange(len(ll))
    ax.bar(xi - w / 2, ll.cv_log_loss, w, label="CV", color="#8172b2")
    ax.bar(xi + w / 2, ll.test_log_loss, w, label="student-test", color="#ccb974")
    ax.set_xticks(xi, ll.model, rotation=30); ax.set_yscale("log"); ax.legend()
    ax.set_title("Log loss (probabilistic models only; lower is better)")

    ax = axes[1, 1]
    data = [folds.loc[folds.model == n, "top1_acc"].values for n in names]
    ax.boxplot(data, labels=names)
    ax.scatter(x + 1, res.cv_train_top1, marker="x", color="red", label="train-sample top-1", zorder=3)
    ax.tick_params(axis="x", rotation=30); ax.legend()
    ax.set_title("Top-1 per CV fold (box) vs training fit (x): overfitting check")
    fig.tight_layout()
    fig.savefig(figs / "model_comparison.png", dpi=120)
    plt.close(fig)

    hist_path = rep / "mlp_history.csv"
    if hist_path.exists():
        h = pd.read_csv(hist_path)
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(h.epoch, h.train_loss, color="#4c72b0", marker="o", label="training log loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss")
        ax2 = ax1.twinx()
        ax2.plot(h.epoch, h.val_accuracy, color="#c44e52", marker="s", label="held-out accuracy (10%)")
        ax2.set_ylabel("validation accuracy")
        fig.legend(loc="center right"); ax1.set_title("MLP training history (sklearn early-stopping split)")
        fig.tight_layout()
        fig.savefig(figs / "mlp_history.png", dpi=120)
        plt.close(fig)

    # confusion matrices on student_test, candidate level, top-1 decision, valid fixes
    test = pd.read_pickle(cfg["paths"]["artifacts"] / "cache" / "features_student_test.pkl")
    g = fix_groups(test)
    valid = test.valid.to_numpy()
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    for ax, n in zip(axes.ravel(), names):
        s = np.load(cfg["paths"]["artifacts"] / "models" / f"{n}_student_test_scores.npy")
        dec = np.zeros(len(test), int)
        dec[top1_rows(g, s)] = 1
        cm = confusion_matrix(test.label.to_numpy()[valid], dec[valid])
        ConfusionMatrixDisplay(cm, display_labels=["other", "reference"]).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(n)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Student-test confusion matrices (candidate level, top-1 decision)")
    fig.tight_layout()
    fig.savefig(figs / "model_confusion_matrices.png", dpi=110)
    print(res[["model", "cv_top1_mean", "cv_top1_std", "cv_macro_f1_top1", "cv_log_loss", "cv_train_top1",
               "test_top1", "test_macro_f1_top1", "test_log_loss", "test_route_continuity",
               "fit_s_full", "infer_ms_per_1k_fixes"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
