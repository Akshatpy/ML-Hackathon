"""Candidate- and observation-level metrics.

Rows of a feature table are contiguous per (trip, fix), which lets every per-fix
reduction run as a numpy segment operation instead of a pandas groupby.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score


def fix_groups(df: pd.DataFrame) -> np.ndarray:
    """Group id per row; a new group starts whenever trip or fix index changes."""
    trip_change = df.trip_id.to_numpy()[1:] != df.trip_id.to_numpy()[:-1]
    obs_change = df.obs.to_numpy()[1:] != df.obs.to_numpy()[:-1]
    return np.cumsum(np.r_[True, trip_change | obs_change]) - 1


def top1_rows(groups: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Row index of the highest-scoring candidate in each group."""
    order = np.lexsort((-score, groups))
    first = np.r_[True, groups[order][1:] != groups[order][:-1]]
    return order[first]


def softmax_by_group(groups: np.ndarray, score: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    mx = np.maximum.reduceat(score, starts)[groups]
    e = np.exp(score - mx)
    return e / np.add.reduceat(e, starts)[groups]


def load_arcs(trip_dirs: dict) -> dict:
    """trip_id -> (n_arcs, 2) array; continuity checks only need the endpoints."""
    return {tid: np.fromfile(p / f"{tid}.arcs", sep=" ", dtype=np.int64).reshape(-1, 2)
            for tid, p in trip_dirs.items()}


def route_continuity(pred: pd.DataFrame, arcs_by_trip: dict) -> float:
    """Share of consecutive fixes (same gap segment) whose predicted arcs are the same arc
    or directly connected (end node of one = start node of the next). `pred` has columns
    trip_id, obs, arc, segment and must be sorted by trip and obs."""
    ok, total = 0, 0
    for tid, g in pred.groupby("trip_id", sort=False):
        arcs = arcs_by_trip[tid]
        a = g.arc.to_numpy()
        seg = g.segment.to_numpy()
        same_seg = seg[1:] == seg[:-1]
        prev, nxt = a[:-1][same_seg], a[1:][same_seg]
        cont = (prev == nxt) | (arcs[prev, 1] == arcs[nxt, 0])
        ok += int(cont.sum())
        total += int(same_seg.sum())
    return ok / max(total, 1)


def evaluate(df: pd.DataFrame, prob: np.ndarray, arcs_by_trip: dict | None = None,
             has_probability: bool = True) -> dict:
    """df: candidate rows (all fixes of the evaluated trips); prob: score per row.
    Classification metrics use valid fixes only; continuity uses every fix, because
    the route has to be plausible even where the reference label isn't trustworthy."""
    groups = fix_groups(df)
    best = top1_rows(groups, prob)
    top1_dec = np.zeros(len(df), np.int8)
    top1_dec[best] = 1

    valid = df.valid.to_numpy()
    y = df.label.to_numpy()[valid]
    p = prob[valid]
    best_valid = best[valid[best]]

    out = {
        "top1_acc": float(df.label.to_numpy()[best_valid].mean()),
        "macro_f1_top1": float(f1_score(y, top1_dec[valid], average="macro")),
        "precision_top1": float(precision_score(y, top1_dec[valid])),
        "recall_top1": float(recall_score(y, top1_dec[valid])),
        "roc_auc": float(roc_auc_score(y, p)),
    }
    if has_probability:
        out["macro_f1_thr05"] = float(f1_score(y, (p >= 0.5).astype(int), average="macro"))
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))
    else:
        out["macro_f1_thr05"] = np.nan
        out["log_loss"] = np.nan
    if arcs_by_trip is not None:
        chosen = df.iloc[best][["trip_id", "obs", "arc", "segment"]].sort_values(["trip_id", "obs"])
        out["route_continuity"] = route_continuity(chosen, arcs_by_trip)
    return out
