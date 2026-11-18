"""Builds the candidate-level feature tables for train and student_test.

Output: artifacts/cache/features_<split>.pkl (one row per fix x candidate arc)
        reports/features/candidate_stats.csv (per-trip candidate coverage)
Usage:  python -m src.build_features
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .candidates import generate
from .config import load_config
from .data_prep import list_trips, load_trip
from .features import FEATURES, TripGraphCache, build, candidate_labels
from .labels import label_trip
from .preprocess import prepare_trip
from .spatial import ArcIndex


def trip_table(trip, cfg: dict, with_labels: bool = True):
    """Shared by training and inference; inference passes with_labels=False."""
    labels = label_trip(trip) if (with_labels and trip.route is not None) else None
    prep = prepare_trip(trip, cfg, labels)
    index = ArcIndex(trip.node_xy, trip.arcs)
    cc = cfg["candidates"]
    cand = generate(trip, index, cc["radius_m"], cc["fallback_radius_m"], cc["fallback_k"])
    graph = TripGraphCache(trip)
    feats = build(trip, prep, cand, graph, cfg)

    df = pd.DataFrame(feats)
    df.insert(0, "trip_id", trip.trip_id)
    df.insert(1, "obs", cand["obs"].astype(np.int32))
    df.insert(2, "arc", cand["arc"].astype(np.int64))
    df["fallback"] = cand["fallback"]
    df["segment"] = prep["segment"][cand["obs"]].astype(np.int32)
    if labels is not None:
        df["label"] = candidate_labels(trip, cand, prep)
        df["valid"] = prep["valid"][cand["obs"]]
    return df, prep, cand


def _one(trip_dir, cfg):
    t0 = time.time()
    trip = load_trip(trip_dir)
    df, prep, cand = trip_table(trip, cfg)
    valid = prep["valid"]
    has_pos = np.zeros(trip.n_obs, bool)
    has_pos[df.obs[df.label == 1].to_numpy()] = True
    stats = {
        "trip_id": trip.trip_id, "n_obs": trip.n_obs, "n_rows": len(df),
        "valid_frac": float(valid.mean()),
        "cand_miss_rate_valid": float(1 - has_pos[valid].mean()) if valid.any() else np.nan,
        "mean_cands": len(df) / trip.n_obs,
        "fallback_fixes": int(df.loc[df.fallback, "obs"].nunique()),
        "n_segments": int(prep["segment"].max() + 1),
        "pos_rate": float(df.label.mean()),
        "secs": round(time.time() - t0, 1),
    }
    return df, stats


def run(n_jobs: int = 4):
    cfg = load_config()
    cache = cfg["paths"]["artifacts"] / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rep = cfg["paths"]["reports"] / "features"
    rep.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(cfg["paths"]["root_configs"] / "folds.csv", dtype={"trip_id": str})

    all_stats = []
    for split, d in [("train", cfg["paths"]["train_dir"]), ("student_test", cfg["paths"]["student_test_dir"])]:
        results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_one)(p, cfg) for p in list_trips(d))
        table = pd.concat([r[0] for r in results], ignore_index=True)
        if split == "train":
            table = table.merge(folds[["trip_id", "fold"]], on="trip_id", how="left")
        else:
            table["fold"] = -1
        table.to_pickle(cache / f"features_{split}.pkl")
        for _, s in results:
            s["split"] = split
            all_stats.append(s)
        print(f"{split}: {len(table):,} rows, {table.trip_id.nunique()} trips, "
              f"{table.obs.groupby(table.trip_id).nunique().sum():,} fixes, pos_rate={table.label.mean():.4f}")

    stats = pd.DataFrame(all_stats)
    stats.to_csv(rep / "candidate_stats.csv", index=False)
    print(stats.groupby("split")[["valid_frac", "cand_miss_rate_valid", "mean_cands", "fallback_fixes",
                                   "n_segments", "secs"]].describe().T.round(4).to_string())
    print("features:", len(FEATURES))


if __name__ == "__main__":
    run()
