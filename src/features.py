"""Per-candidate features. One row = (GPS fix, candidate arc).

Feature groups, each tied to something EDA showed:
  geometry   - distance to the arc; but the label is the closest arc only 55% of the time
  direction  - heading vs arc bearing; separates a street from its reverse twin (~30% of errors)
  context    - distances of neighbouring fixes to the same arc; one noisy fix rarely
               moves the whole window onto a parallel street
  relative   - each value compared with the best competitor at the same fix, because
               what matters is "closer than the alternatives", not absolute metres
  topology   - node degree, arc length, position along the arc; intersections are where
               parallel and crossing candidates pile up
  kinematics - speed, turning, gaps; heading is meaningless when the vehicle is stopped
"""
import numpy as np

from .preprocess import wrap_angle
from .spatial import point_segment_distance

FEATURES = [
    "dist", "log_dist", "dist_rank", "dist_minus_best", "dist_ratio_best",
    "n_cand", "n_cand_15m",
    "cos_heading", "abs_heading_diff", "heading_ok", "heading_carried", "cos_minus_best", "along_speed",
    "ctx_dist_mean", "ctx_dist_max", "ctx_mean_minus_best", "ctx_prev_dist", "ctx_next_dist",
    # twin_in_cands is still computed but left out: a twin shares the geometry, so it is
    # always inside the radius and the column duplicated has_twin exactly (AUC 0.5311 both)
    "has_twin",
    "arc_len", "frac", "dist_to_arc_end", "from_out_deg", "to_out_deg", "to_in_deg",
    "speed", "turn", "dt_prev", "step_prev",
]


def _group_starts(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows are sorted by obs. Returns start index of each fix's block and the row->block map."""
    starts = np.flatnonzero(np.r_[True, obs[1:] != obs[:-1]])
    block = np.cumsum(np.r_[True, obs[1:] != obs[:-1]]) - 1
    return starts, block


def _group_reduce(values, starts, block, fn):
    return fn.reduceat(values, starts)[block]


def reverse_twin(arcs: np.ndarray) -> np.ndarray:
    n = int(arcs.max()) + 1
    key = arcs[:, 0] * n + arcs[:, 1]
    rkey = arcs[:, 1] * n + arcs[:, 0]
    order = np.argsort(key)
    pos = np.clip(np.searchsorted(key[order], rkey), 0, len(key) - 1)
    return np.where(key[order][pos] == rkey, order[pos], -1)


class TripGraphCache:
    """Per-trip arc attributes reused by features and the decoder."""

    def __init__(self, trip):
        a, b = trip.node_xy[trip.arcs[:, 0]], trip.node_xy[trip.arcs[:, 1]]
        d = b - a
        self.bearing = np.arctan2(d[:, 0], d[:, 1])
        self.length = np.linalg.norm(d, axis=1)
        self.unit = d / np.where(self.length > 0, self.length, 1)[:, None]
        n_nodes = len(trip.node_xy)
        self.out_deg = np.bincount(trip.arcs[:, 0], minlength=n_nodes)
        self.in_deg = np.bincount(trip.arcs[:, 1], minlength=n_nodes)
        self.twin = reverse_twin(trip.arcs)
        self.a, self.b = a, b


def build(trip, prep: dict, cand: dict, graph: TripGraphCache, cfg: dict) -> dict:
    obs, arc, dist = cand["obs"], cand["arc"], cand["dist"]
    starts, block = _group_starts(obs)
    n_per = np.diff(np.r_[starts, len(obs)])[block]
    rank = np.arange(len(obs)) - starts[block]        # rows already sorted by distance within fix
    best = _group_reduce(dist, starts, block, np.minimum)

    # direction
    ok = prep["heading_ok"][obs]
    diff = np.abs(wrap_angle(prep["heading"][obs] - graph.bearing[arc]))
    cos_h = np.where(ok, np.cos(diff), 0.0)
    abs_diff = np.where(ok, diff, np.pi / 2)            # unknown heading -> neutral 90 degrees

    # context: same arc, neighbouring fixes (clamped inside the gap segment)
    seg = prep["segment"]
    seg_start = np.searchsorted(seg, seg, side="left")
    seg_end = np.searchsorted(seg, seg, side="right") - 1
    ctx = []
    for off in cfg["features"]["context_offsets"]:
        j = np.clip(obs + off, seg_start[obs], seg_end[obs])
        d_j, _ = point_segment_distance(trip.track_xy[j], graph.a[arc], graph.b[arc])
        ctx.append(d_j)
    ctx = np.vstack(ctx)
    offsets = np.asarray(cfg["features"]["context_offsets"])
    ctx_mean = ctx.mean(axis=0)

    twin = graph.twin[arc]
    # twin present among this fix's candidates: same fix block contains the twin id
    key_self = obs.astype(np.int64) * len(graph.length) + arc
    key_twin = obs.astype(np.int64) * len(graph.length) + np.where(twin >= 0, twin, 0)
    twin_in = (twin >= 0) & np.isin(key_twin, key_self)

    arcs = trip.arcs
    feats = {
        "dist": dist,
        "log_dist": np.log1p(dist),
        "dist_rank": rank.astype(float),
        "dist_minus_best": dist - best,
        "dist_ratio_best": (dist + 1.0) / (best + 1.0),
        "n_cand": n_per.astype(float),
        "n_cand_15m": _group_reduce((dist <= 15).astype(float), starts, block, np.add),
        "cos_heading": cos_h,
        "abs_heading_diff": abs_diff,
        "heading_ok": ok.astype(float),
        "heading_carried": prep["heading_carried"][obs].astype(float),
        "cos_minus_best": cos_h - _group_reduce(cos_h, starts, block, np.maximum),
        "along_speed": prep["speed"][obs] * cos_h,
        "ctx_dist_mean": ctx_mean,
        "ctx_dist_max": ctx.max(axis=0),
        "ctx_mean_minus_best": ctx_mean - _group_reduce(ctx_mean, starts, block, np.minimum),
        "ctx_prev_dist": ctx[offsets == -1][0],
        "ctx_next_dist": ctx[offsets == 1][0],
        "has_twin": (twin >= 0).astype(float),
        "twin_in_cands": twin_in.astype(float),
        "arc_len": graph.length[arc],
        "frac": cand["frac"],
        "dist_to_arc_end": (1 - cand["frac"]) * graph.length[arc],
        "from_out_deg": graph.out_deg[arcs[arc, 0]].astype(float),
        "to_out_deg": graph.out_deg[arcs[arc, 1]].astype(float),
        "to_in_deg": graph.in_deg[arcs[arc, 1]].astype(float),
        "speed": prep["speed"][obs],
        "turn": prep["turn"][obs],
        "dt_prev": np.minimum(prep["dt_prev"][obs], 600.0),
        "step_prev": np.minimum(prep["step_prev"][obs], 2000.0),
    }
    return {k: v.astype(np.float32) for k, v in feats.items()}


def candidate_labels(trip, cand: dict, prep: dict) -> np.ndarray:
    """1 if the candidate is the reference arc. Arcs with identical (from, to) nodes are
    duplicates in the source graph (EDA: up to 322 per trip) and indistinguishable, so
    they count as the same arc."""
    lab = prep["label_arc"][cand["obs"]]
    same_id = cand["arc"] == lab
    same_nodes = (trip.arcs[cand["arc"], 0] == trip.arcs[lab, 0]) & (trip.arcs[cand["arc"], 1] == trip.arcs[lab, 1])
    return (same_id | same_nodes).astype(np.int8)
