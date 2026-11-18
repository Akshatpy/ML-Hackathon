"""Extra EDA outputs requested by the dataset-exploration brief that src/eda.py doesn't draw:
trip locations, sampling intervals, network size/connectivity, per-trip label quality,
speed distribution and an explicit missing-value table.

Run after `python -m src.eda`:  python scripts/eda_extras.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data_prep import list_trips  # noqa: E402

SPLIT_COLOURS = {"train": "#d95f02", "student_test": "#1b9e77"}


def raw_track(trip_dir: Path) -> np.ndarray:
    return np.fromfile(trip_dir / f"{trip_dir.name}.track", sep=" ").reshape(-1, 3)


def main():
    cfg = load_config()
    eda = cfg["paths"]["reports"] / "eda"
    figs = cfg["paths"]["reports"] / "figures"
    trips = pd.read_csv(eda / "trip_inventory.csv", dtype={"trip_id": str})
    obs = pd.read_csv(eda / "obs_labels.csv", dtype={"trip_id": str})

    # 1. missing values per file type (raw parse, before any cleaning)
    rows, dts = [], {"train": [], "student_test": []}
    for split, d in [("train", cfg["paths"]["train_dir"]), ("student_test", cfg["paths"]["student_test_dir"])]:
        for p in list_trips(d):
            tid = p.name
            tr = raw_track(p)
            nodes = np.fromfile(p / f"{tid}.nodes", sep=" ").reshape(-1, 2)
            arcs = np.fromfile(p / f"{tid}.arcs", sep=" ").reshape(-1, 2)
            route = np.fromfile(p / f"{tid}.route", sep=" ")
            rows.append({"split": split, "trip_id": tid,
                         "track_rows": len(tr), "track_nan": int(np.isnan(tr).sum()),
                         "nodes_rows": len(nodes), "nodes_nan": int(np.isnan(nodes).sum()),
                         "arcs_rows": len(arcs), "arcs_nan": int(np.isnan(arcs).sum()),
                         "route_rows": len(route), "route_nan": int(np.isnan(route).sum()),
                         "track_dup_timestamps": int((np.diff(tr[:, 2]) == 0).sum())})
            dts[split].append(np.diff(tr[:, 2]))
    missing = pd.DataFrame(rows)
    missing.to_csv(eda / "missing_values.csv", index=False)
    print("missing-value totals:\n", missing.filter(like="_nan").sum().to_string())

    # 2. trip locations: worldwide spread is why each trip gets its own projection
    trips["lon_c"] = (trips.lon_min + trips.lon_max) / 2
    trips["lat_c"] = (trips.lat_min + trips.lat_max) / 2
    world = folium.Map(location=[30, 20], zoom_start=2, tiles="CartoDB positron")
    for split, g in trips.groupby("split"):
        layer = folium.FeatureGroup(name=f"{split} ({len(g)} trips)")
        for r in g.itertuples():
            folium.CircleMarker((r.lat_c, r.lon_c), radius=5, color=SPLIT_COLOURS[split], fill=True,
                                fill_opacity=0.8,
                                tooltip=f"{split} {r.trip_id}: {r.n_obs} fixes, {r.n_arcs} arcs").add_to(layer)
        layer.add_to(world)
    folium.LayerControl(collapsed=False).add_to(world)
    world.save(str(cfg["paths"]["reports"] / "maps" / "eda_trip_locations.html"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    for split, g in trips.groupby("split"):
        ax.scatter(g.lon_c, g.lat_c, s=25, c=SPLIT_COLOURS[split], label=split)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Trip centroids (worldwide -> per-trip projection)")
    ax.legend()

    # 3. sampling intervals (log y): the 1 s spike plus a long tail of gaps
    ax = axes[0, 1]
    bins = np.r_[0.5, 1.5, 2.5, 5, 10, 30, 60, 300, 3600, 2e5]
    for split, arr in dts.items():
        c, _ = np.histogram(np.concatenate(arr), bins=bins)
        ax.bar(np.arange(len(c)) + (0.2 if split == "train" else -0.2), c / c.sum(), width=0.4,
               color=SPLIT_COLOURS[split], label=split)
    ax.set_xticks(range(len(bins) - 1))
    ax.set_xticklabels(["1", "2", "3-5", "5-10", "10-30", "30-60", "1-5m", "5m-1h", ">1h"])
    ax.set_yscale("log"); ax.set_xlabel("time step (s)"); ax.set_ylabel("share of steps")
    ax.set_title("Sampling intervals"); ax.legend()

    # 4. network size vs fragmentation
    ax = axes[1, 0]
    for split, g in trips.groupby("split"):
        ax.scatter(g.n_arcs, g.weak_components, s=25, c=SPLIT_COLOURS[split], label=split)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arcs in trip graph"); ax.set_ylabel("weakly connected components")
    ax.set_title("Road-network size vs fragmentation (graph cropped to track bbox)")
    ax.legend()

    # 5. label quality per trip: truncated reference routes stand out
    ax = axes[1, 1]
    order = obs.groupby("trip_id").label_dist.median().sort_values().index
    data = [obs.loc[obs.trip_id == t, "label_dist"].clip(upper=300).values for t in order]
    ax.boxplot(data, showfliers=False, widths=0.6)
    ax.axhline(cfg["preprocess"]["label_valid_m"], color="red", lw=0.8, ls="--", label="50 m validity cut")
    ax.set_xticks([]); ax.set_yscale("symlog", linthresh=10); ax.set_ylim(bottom=0)
    ax.set_xlabel("trips sorted by median distance"); ax.set_ylabel("fix -> aligned reference arc (m)")
    ax.set_title("Reference-label quality per trip (rightmost: 33, 54)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "eda_extras.png", dpi=120)

    # 6. speed distribution on 1 s steps only (gaps would bias it down)
    speeds = []
    for split, d in [("train", cfg["paths"]["train_dir"]), ("student_test", cfg["paths"]["student_test_dir"])]:
        for p in list_trips(d):
            tr = raw_track(p)
            lat = np.radians(tr[:, 1])
            # equirectangular step length is fine at 1 s / <60 m scale
            dx = np.radians(np.diff(tr[:, 0])) * np.cos(lat[:-1]) * 6371000
            dy = np.radians(np.diff(tr[:, 1])) * 6371000
            one = np.isclose(np.diff(tr[:, 2]), 1)
            speeds.append(pd.DataFrame({"split": split, "speed": np.hypot(dx, dy)[one]}))
    sp = pd.concat(speeds)
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, g in sp.groupby("split"):
        ax.hist(g.speed.clip(upper=70), bins=70, density=True, alpha=0.6, color=SPLIT_COLOURS[split], label=split)
    ax.set_xlabel("speed on 1 s steps (m/s, clipped at 70)"); ax.set_ylabel("density")
    ax.set_title("Speed distribution"); ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "eda_speed.png", dpi=120)
    print("share of 1-s steps faster than 60 m/s:", round(float((sp.speed > 60).mean()), 5))
    print("stationary share (<0.5 m/s):", sp.groupby("split").speed.apply(lambda s: round(float((s < 0.5).mean()), 4)).to_dict())


if __name__ == "__main__":
    main()
