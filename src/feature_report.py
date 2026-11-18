"""Feature diagnostics on the cached tables (valid fixes only).

1. Single-feature ROC AUC for candidate = reference arc (train, sampled).
2. Rule baselines scored as observation-level top-1 accuracy; they show what each
   feature group buys before any learning happens.
Usage: python -m src.feature_report
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import load_config
from .features import FEATURES


def top1(df: pd.DataFrame, score: np.ndarray) -> float:
    """Highest score per fix wins; accuracy over fixes."""
    s = pd.Series(score, index=df.index)
    best = s.groupby([df.trip_id, df.obs]).idxmax()
    return float(df.label.loc[best.values].mean())


RULES = {
    "nearest arc": lambda d: -d.dist,
    "nearest arc, heading must agree": lambda d: -d.dist + 1000 * (d.cos_heading > 0),
    "window-mean distance": lambda d: -d.ctx_dist_mean - d.dist,
    "window-mean distance + heading": lambda d: -(d.ctx_dist_mean + d.dist) + 1000 * (d.cos_heading > 0),
}


def run():
    cfg = load_config()
    cache = cfg["paths"]["artifacts"] / "cache"
    out = cfg["paths"]["reports"] / "features"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    rule_rows = []
    for split in ["train", "student_test"]:
        df = pd.read_pickle(cache / f"features_{split}.pkl")
        df = df[df.valid].reset_index(drop=True)
        sample = df.sample(min(len(df), 600_000), random_state=cfg["seed"])
        for f in FEATURES:
            auc = roc_auc_score(sample.label, sample[f])
            rows.append({"split": split, "feature": f, "auc": auc, "auc_dir_free": max(auc, 1 - auc)})
        for name, rule in RULES.items():
            rule_rows.append({"split": split, "rule": name, "top1": top1(df, rule(df).to_numpy())})

    auc = pd.DataFrame(rows).pivot(index="feature", columns="split", values="auc_dir_free")
    auc = auc.sort_values("train", ascending=False)
    auc.to_csv(out / "feature_auc.csv")
    rules = pd.DataFrame(rule_rows).pivot(index="rule", columns="split", values="top1")
    rules.to_csv(out / "rule_baselines.csv")
    print(auc.round(4).to_string())
    print()
    print(rules.round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 9))
    auc[["train", "student_test"]].iloc[::-1].plot.barh(ax=ax)
    ax.axvline(0.5, color="grey", lw=0.8)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("single-feature ROC AUC (direction-free)")
    ax.set_title("How well each feature alone separates the reference arc")
    fig.tight_layout()
    fig.savefig(cfg["paths"]["reports"] / "figures" / "feature_auc.png", dpi=120)


if __name__ == "__main__":
    run()
