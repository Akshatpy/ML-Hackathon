"""Trip-grouped cross-validation folds.

Grouping by trip alone is not enough: EDA found train trips that drive the same
streets (10/16/29 in Moscow share up to 56% of route nodes, 53/93 share 33%,
43/54 share 19%). Those trips are merged into one group so a fold never
validates on roads it has already seen labelled. Student-test trips share no
route nodes with train, so this mirrors the held-out evaluation setting.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import load_config


def overlap_groups(trip_ids: list[str], overlap_pairs: pd.DataFrame) -> dict[str, str]:
    parent = {t: t for t in trip_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(overlap_pairs.trip_a, overlap_pairs.trip_b):
        a, b = a.split("/")[-1], b.split("/")[-1]
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    return {t: find(t) for t in trip_ids}


def make_folds(n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    cfg = load_config()
    eda_dir = cfg["paths"]["reports"] / "eda"
    trips = pd.read_csv(eda_dir / "trip_inventory.csv", dtype={"trip_id": str})
    trips = trips[trips.split == "train"].reset_index(drop=True)
    pairs = pd.read_csv(eda_dir / "route_overlap_pairs.csv")
    groups = overlap_groups(trips.trip_id.tolist(), pairs)
    trips["group"] = trips.trip_id.map(groups)

    # GroupKFold balances fold sizes by sample count; weighting by n_obs keeps the
    # 15-19k-observation trips (21, 54, 84) from piling into one fold. Shuffle the
    # group order first so folds aren't sorted by trip id.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(trips))
    trips = trips.iloc[order].reset_index(drop=True)
    weights = np.repeat(np.arange(len(trips)), trips.n_obs.clip(upper=5000) // 100 + 1)
    gkf = GroupKFold(n_splits=n_splits)
    fold_of = {}
    for k, (_, val_idx) in enumerate(gkf.split(weights, groups=trips.group.values[weights])):
        for i in np.unique(weights[val_idx]):
            fold_of[trips.trip_id[i]] = k
    trips["fold"] = trips.trip_id.map(fold_of)
    out = trips[["trip_id", "group", "fold", "n_obs"]].sort_values("trip_id")
    out.to_csv(cfg["paths"]["root_configs"] / "folds.csv", index=False)
    return out


if __name__ == "__main__":
    folds = make_folds()
    print(folds.groupby("fold").agg(trips=("trip_id", "count"), obs=("n_obs", "sum")))
    multi = folds.groupby("group").filter(lambda g: len(g) > 1)
    print(multi.sort_values("group").to_string(index=False))
