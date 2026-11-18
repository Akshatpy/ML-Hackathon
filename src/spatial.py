"""Point-to-segment geometry and a radius index over road arcs.

Arcs are straight node-to-node segments, so exact distances are cheap once we
know which arcs are nearby. The index densifies every arc into points spaced
<= `step` metres and uses a KD-tree on those; a radius query of r + step/2 is
then guaranteed to return every arc whose true distance is <= r.
"""
import numpy as np
from scipy.spatial import cKDTree


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised over broadcastable arrays of shape (..., 2).
    Returns (distance, fraction along segment in [0, 1])."""
    ab = b - a
    denom = np.einsum("...i,...i->...", ab, ab)
    frac = np.einsum("...i,...i->...", p - a, ab) / np.where(denom == 0, 1.0, denom)
    frac = np.clip(frac, 0.0, 1.0)
    proj = a + frac[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1), frac


class ArcIndex:
    def __init__(self, node_xy: np.ndarray, arcs: np.ndarray, step: float = 10.0):
        self.a = node_xy[arcs[:, 0]]
        self.b = node_xy[arcs[:, 1]]
        self.length = np.linalg.norm(self.b - self.a, axis=1)
        self.step = step
        # some node files carry non-finite coordinates; arcs touching them can never be candidates
        self.valid = np.isfinite(self.length)
        valid_ids = np.flatnonzero(self.valid)
        n_pts = np.maximum(np.ceil(self.length[valid_ids] / step).astype(int), 1) + 1
        owner = np.repeat(valid_ids, n_pts)
        # fraction along each arc for its densified points
        offsets = np.concatenate([np.linspace(0, 1, k) for k in n_pts])
        samples = self.a[owner] + offsets[:, None] * (self.b - self.a)[owner]
        self.owner = owner
        self.tree = cKDTree(samples)

    def query(self, p: np.ndarray, radius: float):
        """Arcs within `radius` metres of point p -> (arc_ids, distances, fractions)."""
        hits = self.tree.query_ball_point(p, radius + self.step / 2)
        if not hits:
            return np.empty(0, int), np.empty(0), np.empty(0)
        cand = np.unique(self.owner[hits])
        dist, frac = point_segment_distance(p, self.a[cand], self.b[cand])
        keep = dist <= radius
        return cand[keep], dist[keep], frac[keep]

    def nearest(self, p: np.ndarray, start_radius: float = 50.0, max_radius: float = 2000.0):
        r = start_radius
        while r <= max_radius:
            ids, dist, _ = self.query(p, r)
            if len(ids):
                k = np.argmin(dist)
                return ids[k], dist[k]
            r *= 2
        return -1, np.inf
