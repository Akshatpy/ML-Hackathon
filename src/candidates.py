"""Candidate road arcs per GPS fix.

Candidates are every directed arc within `radius_m` of the fix. Both directions
of a two-way street are kept on purpose: picking the right one is the hardest
part of the task (EDA: the reverse twin is the nearest arc for ~30% of fixes).
"""
import numpy as np

from .spatial import ArcIndex, point_segment_distance


def _pairs_within(index: ArcIndex, xy: np.ndarray, radius: float):
    hits = index.tree.query_ball_point(xy, radius + index.step / 2, return_sorted=False)
    lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    if lens.sum() == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    obs = np.repeat(np.arange(len(xy)), lens)
    arc = index.owner[np.concatenate([h for h in hits if h])]
    key = np.unique(obs * len(index.a) + arc)          # densified points -> one row per (fix, arc)
    return key // len(index.a), key % len(index.a)


def generate(trip, index: ArcIndex, radius: float, fallback_radius: float, fallback_k: int) -> dict:
    xy = trip.track_xy
    obs, arc = _pairs_within(index, xy, radius)
    dist, frac = point_segment_distance(xy[obs], index.a[arc], index.b[arc])
    keep = dist <= radius
    obs, arc, dist, frac = obs[keep], arc[keep], dist[keep], frac[keep]
    fallback = np.zeros(len(obs), bool)

    # Fixes with nothing nearby (off-network excursions, cropped graph edges) still need
    # candidates or the decoder has a hole; take the k nearest within a wider radius.
    empty = np.setdiff1d(np.arange(len(xy)), obs)
    if len(empty):
        f_obs, f_arc = _pairs_within(index, xy[empty], fallback_radius)
        f_dist, f_frac = point_segment_distance(xy[empty][f_obs], index.a[f_arc], index.b[f_arc])
        order = np.lexsort((f_dist, f_obs))
        f_obs, f_arc, f_dist, f_frac = f_obs[order], f_arc[order], f_dist[order], f_frac[order]
        first = np.searchsorted(f_obs, f_obs, side="left")
        sel = (np.arange(len(f_obs)) - first) < fallback_k
        obs = np.r_[obs, empty[f_obs[sel]]]
        arc = np.r_[arc, f_arc[sel]]
        dist = np.r_[dist, f_dist[sel]]
        frac = np.r_[frac, f_frac[sel]]
        fallback = np.r_[fallback, np.ones(sel.sum(), bool)]

    order = np.lexsort((dist, obs))                       # sorted by fix, then distance
    return {k: v[order] for k, v in
            {"obs": obs, "arc": arc, "dist": dist, "frac": frac, "fallback": fallback}.items()}
