"""Derive one reference arc per GPS observation from the ordered .route list.

The .route file is the sequence of arcs driven, not a per-observation label
(trip 1: 1315 fixes vs 525 arcs). Nearest-route-arc assignment breaks on trips
that revisit a street or drive both directions of it, so instead we solve a
monotone alignment: observation i maps to route position j_i with
j_0 <= j_1 <= ... and total point-to-segment distance minimised. Exact DP,
O(n_obs * n_route) time with a running prefix-min.
"""
import numpy as np

from .spatial import point_segment_distance


def route_distance_matrix(track_xy, node_xy, arcs, route) -> np.ndarray:
    a = node_xy[arcs[route, 0]][None, :, :]
    b = node_xy[arcs[route, 1]][None, :, :]
    dist, _ = point_segment_distance(track_xy[:, None, :], a, b)
    return dist


def monotone_align(dist: np.ndarray) -> np.ndarray:
    n, m = dist.shape
    back = np.empty((n, m), dtype=np.int32)
    cost = dist[0].copy()
    cols = np.arange(m)
    for i in range(1, n):
        run_min = np.minimum.accumulate(cost)
        # index of the latest position reaching the running min, carried forward
        arg = np.maximum.accumulate(np.where(cost <= run_min, cols, 0))
        back[i] = arg
        cost = dist[i] + run_min
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def route_is_connected(arcs: np.ndarray, route: np.ndarray) -> np.ndarray:
    """Boolean per consecutive pair: does arc k end where arc k+1 starts."""
    return arcs[route[:-1], 1] == arcs[route[1:], 0]


def label_trip(trip):
    """Returns dict with per-observation route position, arc id and distance."""
    dist = route_distance_matrix(trip.track_xy, trip.node_xy, trip.arcs, trip.route)
    pos = monotone_align(dist)
    return {
        "route_pos": pos,
        "arc_id": trip.route[pos],
        "dist": dist[np.arange(len(pos)), pos],
        "nearest_route_dist": dist.min(axis=1),
    }
