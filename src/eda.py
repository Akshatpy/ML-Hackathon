"""Dataset exploration for train and student_test partitions.

Writes per-trip tables to reports/eda/ and figures to reports/figures/.
Usage: python -m src.eda
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import load_config
from .data_prep import list_trips, load_trip
from .labels import label_trip, route_is_connected
from .spatial import ArcIndex, point_segment_distance


def reverse_twin(arcs: np.ndarray) -> np.ndarray:
    """twin[k] = id of arc (v,u) for arc k=(u,v), or -1."""
    n_nodes = int(arcs.max()) + 1
    key = arcs[:, 0] * n_nodes + arcs[:, 1]
    rkey = arcs[:, 1] * n_nodes + arcs[:, 0]
    order = np.argsort(key)
    pos = np.searchsorted(key[order], rkey)
    pos = np.clip(pos, 0, len(key) - 1)
    found = key[order][pos] == rkey
    return np.where(found, order[pos], -1)


def track_stats(trip, cfg) -> dict:
    t, xy = trip.t, trip.track_xy
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = np.divide(step, dt, out=np.full_like(step, np.nan), where=dt > 0)
    lon, lat = trip.track_lonlat[:, 0], trip.track_lonlat[:, 1]
    return {
        "n_obs": trip.n_obs,
        "nan_rows": int(np.isnan(trip.track_lonlat).any(axis=1).sum() + np.isnan(t).sum()),
        "t_start": float(t[0]),
        "t_integer": bool(np.allclose(t, np.round(t))),
        "duration_s": float(t[-1] - t[0]),
        "dt_nonpositive": int((dt <= 0).sum()),
        "dt_eq_1_frac": float(np.mean(np.isclose(dt, 1.0))),
        "dt_max": float(dt.max()),
        "n_gaps_gt_thr": int((dt > cfg["eda"]["large_gap_s"]).sum()),
        "dup_positions": int((step < 1e-6).sum()),
        "path_len_km": float(step.sum() / 1000),
        "speed_median_mps": float(np.nanmedian(speed)),
        "speed_p99_mps": float(np.nanpercentile(speed, 99)),
        "n_implausible_speed": int(np.nansum(speed > cfg["eda"]["implausible_speed_mps"])),
        "lon_min": lon.min(), "lon_max": lon.max(), "lat_min": lat.min(), "lat_max": lat.max(),
    }


def graph_stats(trip) -> dict:
    arcs, n_nodes = trip.arcs, len(trip.node_lonlat)
    adj = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n_nodes, n_nodes)).tocsr()
    n_weak, weak_lbl = connected_components(adj, directed=True, connection="weak")
    n_strong, strong_lbl = connected_components(adj, directed=True, connection="strong")
    degree = np.bincount(arcs.ravel(), minlength=n_nodes)
    seg_len = np.linalg.norm(trip.node_xy[arcs[:, 1]] - trip.node_xy[arcs[:, 0]], axis=1)
    twin = reverse_twin(arcs)
    bad_node = ~np.isfinite(trip.node_xy).all(axis=1)
    out = {
        "n_nodes": n_nodes,
        "n_arcs": len(arcs),
        "nonfinite_nodes": int(bad_node.sum()),
        "arcs_on_nonfinite_nodes": int(bad_node[arcs].any(axis=1).sum()),
        "route_on_nonfinite_nodes": int(bad_node[arcs[trip.route]].any(axis=1).sum()) if trip.route is not None else -1,
        "arc_id_out_of_range": int((arcs.max() >= n_nodes) or (arcs.min() < 0)),
        "self_loops": int((arcs[:, 0] == arcs[:, 1]).sum()),
        "dup_arcs": int(len(arcs) - len(np.unique(arcs, axis=0))),
        "isolated_nodes": int((degree == 0).sum()),
        "weak_components": n_weak,
        "largest_weak_frac": float(np.bincount(weak_lbl).max() / n_nodes),
        "strong_components": n_strong,
        "largest_strong_frac": float(np.bincount(strong_lbl).max() / n_nodes),
        "bidirectional_frac": float((twin >= 0).mean()),
        "arc_len_median_m": float(np.median(seg_len)),
        "arc_len_p99_m": float(np.percentile(seg_len, 99)),
        "zero_len_arcs": int((seg_len < 0.01).sum()),
    }
    if trip.route is not None:
        lbl = strong_lbl[arcs[trip.route, 0]]
        out["route_in_main_scc"] = float(np.mean(lbl == np.argmax(np.bincount(strong_lbl))))
    return out, twin, seg_len


def route_and_label_stats(trip, twin, seg_len, index: ArcIndex, radii) -> tuple[dict, pd.DataFrame]:
    route = trip.route
    conn = route_is_connected(trip.arcs, route)
    lab = label_trip(trip)
    d = lab["dist"]

    # Naive baseline + candidate density: what a nearest-arc rule would pick
    r_max = max(radii)
    nearest_arc = np.full(trip.n_obs, -1)
    label_rank = np.full(trip.n_obs, -1)
    n_cand = np.zeros((trip.n_obs, len(radii)), dtype=int)
    for i, p in enumerate(trip.track_xy):
        ids, dist, _ = index.query(p, r_max)
        if len(ids) == 0:
            continue
        order = np.argsort(dist)
        nearest_arc[i] = ids[order[0]]
        n_cand[i] = [(dist <= r).sum() for r in radii]
        hit = np.where(ids[order] == lab["arc_id"][i])[0]
        label_rank[i] = hit[0] if len(hit) else -1

    twin_of_label = twin[lab["arc_id"]]
    rev_route_arcs = np.isin(route, twin[route][twin[route] >= 0])
    stats = {
        "route_len": len(route),
        "route_unique_arcs": len(np.unique(route)),
        "route_out_of_range": int((route >= len(trip.arcs)).sum()),
        "route_breaks": int((~conn).sum()),
        "route_uses_both_directions": int(rev_route_arcs.sum()),
        "route_km": float(seg_len[route].sum() / 1000),
        "obs_per_route_arc": trip.n_obs / len(route),
        "route_arcs_with_no_obs_frac": float(1 - len(np.unique(lab["route_pos"])) / len(route)),
        "label_dist_median": float(np.median(d)),
        "label_dist_p95": float(np.percentile(d, 95)),
        "label_dist_max": float(d.max()),
        "align_vs_nearest_route_disagree": float(np.mean(d > lab["nearest_route_dist"] + 1e-6)),
        "naive_nearest_top1": float(np.mean(nearest_arc == lab["arc_id"])),
        "naive_picks_reverse_twin": float(np.mean((twin_of_label >= 0) & (nearest_arc == twin_of_label))),
        "start_dist_to_route0": float(point_segment_distance(
            trip.track_xy[0], trip.node_xy[trip.arcs[route[0], 0]], trip.node_xy[trip.arcs[route[0], 1]])[0]),
        "first_label_pos": int(lab["route_pos"][0]),
        "last_label_pos_from_end": int(len(route) - 1 - lab["route_pos"][-1]),
    }
    for k, r in enumerate(radii):
        stats[f"miss_rate_r{r}"] = float(np.mean(d > r))
        stats[f"mean_cand_r{r}"] = float(n_cand[:, k].mean())
    obs = pd.DataFrame({
        "trip_id": trip.trip_id, "t": trip.t, "label_arc": lab["arc_id"],
        "route_pos": lab["route_pos"], "label_dist": d, "label_rank": label_rank,
        "nearest_arc": nearest_arc,
    })
    return stats, obs


def node_signature(trip) -> set:
    return set(map(tuple, np.round(trip.node_lonlat, 6)))


def run():
    cfg = load_config()
    out_dir = cfg["paths"]["reports"] / "eda"
    fig_dir = cfg["paths"]["reports"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    radii = cfg["eda"]["radii_m"]

    rows, obs_frames, route_coords = [], [], {}
    splits = {"train": cfg["paths"]["train_dir"], "student_test": cfg["paths"]["student_test_dir"]}
    for split, d in splits.items():
        for trip_dir in list_trips(d):
            t0 = time.time()
            trip = load_trip(trip_dir)
            files = {ext: (trip_dir / f"{trip.trip_id}.{ext}").exists() for ext in ("track", "nodes", "arcs", "route")}
            row = {"split": split, "trip_id": trip.trip_id,
                   "missing_files": ",".join(k for k, v in files.items() if not v)}
            row.update(track_stats(trip, cfg))
            g, twin, seg_len = graph_stats(trip)
            row.update(g)
            if trip.route is not None:
                index = ArcIndex(trip.node_xy, trip.arcs)
                rs, obs = route_and_label_stats(trip, twin, seg_len, index, radii)
                row.update(rs)
                obs["split"] = split
                obs_frames.append(obs)
                a = trip.node_lonlat[trip.arcs[trip.route]]
                route_coords[(split, trip.trip_id)] = set(map(tuple, np.round(a.reshape(-1, 2), 6)))
            row["node_sig_n"] = len(node_signature(trip))
            row["proc_s"] = round(time.time() - t0, 2)
            rows.append(row)
            print(f"{split:12s} {trip.trip_id} obs={trip.n_obs:5d} arcs={len(trip.arcs):6d} "
                  f"top1_naive={row.get('naive_nearest_top1', np.nan):.3f} "
                  f"miss@30={row.get('miss_rate_r30', np.nan):.3f} ({row['proc_s']}s)")

    trips = pd.DataFrame(rows)
    trips.to_csv(out_dir / "trip_inventory.csv", index=False)
    obs_all = pd.concat(obs_frames, ignore_index=True)
    obs_all.to_parquet(out_dir / "obs_labels.parquet") if _has_parquet() else obs_all.to_csv(out_dir / "obs_labels.csv", index=False)

    # Geographic overlap of student-test routes with training routes (same streets driven?)
    train_pts = set().union(*[v for (s, _), v in route_coords.items() if s == "train"])
    overlap = {tid: len(v & train_pts) / len(v) for (s, tid), v in route_coords.items() if s == "student_test"}
    # Trips that share driven road geometry (e.g. several Moscow trips): relevant for
    # deciding whether trip-level grouping alone is enough to stop CV leakage.
    keys = list(route_coords)
    shared = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = route_coords[keys[i]], route_coords[keys[j]]
            inter = len(a & b)
            if inter:
                shared.append({"trip_a": "/".join(keys[i]), "trip_b": "/".join(keys[j]),
                               "shared_route_nodes": inter, "frac_of_smaller": inter / min(len(a), len(b))})
    pd.DataFrame(shared).to_csv(out_dir / "route_overlap_pairs.csv", index=False)

    summary = {
        "test_route_node_overlap_with_train": overlap,
        "n_trip_pairs_sharing_route_nodes": len(shared),
        "obs_label_dist_quantiles_m": obs_all["label_dist"].quantile([.5, .9, .95, .99, .999]).round(2).to_dict(),
        "global_miss_rate": {r: float((obs_all["label_dist"] > r).mean()) for r in radii},
        "naive_top1_by_split": obs_all.assign(ok=obs_all.nearest_arc == obs_all.label_arc)
            .groupby("split")["ok"].mean().round(4).to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot_figures(trips, obs_all, radii, fig_dir)
    return trips, obs_all, summary


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def plot_figures(trips, obs_all, radii, fig_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for split, g in trips.groupby("split"):
        axes[0, 0].hist(g["n_obs"], bins=20, alpha=.6, label=split)
        axes[0, 1].hist(g["n_arcs"], bins=20, alpha=.6, label=split)
        axes[0, 2].hist(g["dt_eq_1_frac"], bins=20, alpha=.6, label=split)
    axes[0, 0].set_title("GPS observations per trip")
    axes[0, 1].set_title("Road arcs per trip graph")
    axes[0, 2].set_title("Fraction of 1 s sampling steps")

    d = obs_all["label_dist"].clip(upper=100)
    axes[1, 0].hist(d, bins=100)
    axes[1, 0].set_title("Distance GPS fix -> reference arc (m, clipped 100)")
    for split, g in obs_all.groupby("split"):
        axes[1, 1].plot(radii, [(g.label_dist > r).mean() for r in radii], marker="o", label=split)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("candidate radius (m)")
    axes[1, 1].set_title("Candidate miss rate vs radius")
    cols = [f"mean_cand_r{r}" for r in radii]
    axes[1, 2].plot(radii, trips[cols].mean().values, marker="o")
    axes[1, 2].set_xlabel("candidate radius (m)")
    axes[1, 2].set_title("Mean candidates per fix")
    for ax in axes.ravel():
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_overview.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    run()
