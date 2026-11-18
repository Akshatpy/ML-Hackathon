"""HMM map matching with Viterbi decoding over classifier-scored candidates.

Hidden state at fix i  : one of fix i's candidate arcs (with the fix's projection on it).
Emission log-score     : w_e * log P_clf(arc | fix), P normalised over the fix's candidates.
Transition log-score   : -|route_dist - straight_dist| / beta  (Newson & Krumm 2009 style):
                         a car moving between two fixes drives about as far along the road
                         network as the straight line between them; a candidate pair that
                         needs a long detour is unlikely. A U-turn onto the reverse twin
                         costs an extra fixed penalty.
Infeasible transitions : no network path within the search cutoff -> -inf.
Breaks                 : gap segments (dt > 30 s) and steps where every transition is
                         infeasible restart the chain from the emission scores alone.
"""
import heapq
from dataclasses import dataclass

import numpy as np


@dataclass
class HMMParams:
    beta: float = 20.0            # metres; scale of the route-vs-straight-line mismatch
    emission_weight: float = 1.0
    uturn_penalty: float = 3.0    # log-units charged for switching to the reverse twin
    cutoff_factor: float = 3.0    # network search limited to cutoff_factor * gc + cutoff_slack
    cutoff_slack: float = 200.0
    cutoff_max: float = 5000.0
    backward_tol: float = 15.0    # metres of apparent backward motion on one arc treated as GPS noise


class RoadGraph:
    """Directed road graph with co-located duplicate nodes merged.

    EDA found 92 node pairs with identical coordinates; without merging, two arcs that
    physically meet can look unconnected and the decoder would declare a break there."""

    def __init__(self, node_xy: np.ndarray, arcs: np.ndarray):
        _, canon = np.unique(np.round(node_xy, 3), axis=0, return_inverse=True)
        self.canon = canon.ravel()
        self.frm = self.canon[arcs[:, 0]]
        self.to = self.canon[arcs[:, 1]]
        a, b = node_xy[arcs[:, 0]], node_xy[arcs[:, 1]]
        self.length = np.linalg.norm(b - a, axis=1)
        self.length = np.where(np.isfinite(self.length), self.length, np.inf)
        order = np.argsort(self.frm, kind="stable")
        self.out_arcs = order
        n_nodes = int(self.canon.max()) + 1
        self.out_start = np.searchsorted(self.frm[order], np.arange(n_nodes + 1))
        self._searches: dict[int, "_Search"] = {}

    def search(self, src: int) -> "_Search":
        s = self._searches.get(src)
        if s is None:
            s = _Search(self, src)
            self._searches[src] = s
        return s

    def node_dist(self, src: int, dst: int, cutoff: float) -> float:
        if src == dst:
            return 0.0
        return self.search(src).dist_to(dst, cutoff)

    def path_arcs(self, src: int, dst: int, cutoff: float) -> list[int]:
        """Arc ids of the shortest path src -> dst (empty if same node or unreachable)."""
        if src == dst or not np.isfinite(self.node_dist(src, dst, cutoff)):
            return []
        s = self.search(src)
        path, node = [], dst
        while node != src:
            arc = s.parent[node]
            path.append(arc)
            node = int(self.frm[arc])
        return path[::-1]


class _Search:
    """Dijkstra from one node that can be resumed with a larger radius later, so the
    many repeated queries from the same arc end across consecutive fixes are cheap."""

    __slots__ = ("g", "dist", "parent", "heap", "settled", "radius")

    def __init__(self, g: RoadGraph, src: int):
        self.g = g
        self.dist = {src: 0.0}
        self.parent: dict[int, int] = {}
        self.heap = [(0.0, src)]
        self.settled: set[int] = set()
        self.radius = 0.0

    def dist_to(self, dst: int, cutoff: float) -> float:
        if dst in self.settled:
            return self.dist[dst]
        g = self.g
        while self.heap and self.heap[0][0] <= cutoff:
            d, u = heapq.heappop(self.heap)
            if u in self.settled:
                continue
            self.settled.add(u)
            self.radius = d
            for k in range(g.out_start[u], g.out_start[u + 1]):
                arc = g.out_arcs[k]
                v = int(g.to[arc])
                nd = d + g.length[arc]
                if nd < self.dist.get(v, np.inf):
                    self.dist[v] = nd
                    self.parent[v] = int(arc)
                    heapq.heappush(self.heap, (nd, v))
            if u == dst:
                return d
        return self.dist[dst] if dst in self.settled else np.inf


def step_distances(g: RoadGraph, arcs_a, frac_a, arcs_b, frac_b, gc: float, p: HMMParams):
    """Network distance for every (candidate at fix i, candidate at fix i+1) pair."""
    cutoff = min(p.cutoff_factor * gc + p.cutoff_slack, p.cutoff_max)
    la, lb = g.length[arcs_a], g.length[arcs_b]
    out = np.full((len(arcs_a), len(arcs_b)), np.inf)
    for i, (a, fa) in enumerate(zip(arcs_a, frac_a)):
        tail = (1.0 - fa) * la[i]                       # rest of arc a
        src = int(g.to[a])
        for j, (b, fb) in enumerate(zip(arcs_b, frac_b)):
            if a == b:
                along = (fb - fa) * la[i]
                if along >= -p.backward_tol:
                    out[i, j] = abs(along)
                    continue
            mid = g.node_dist(src, int(g.frm[b]), cutoff)
            if np.isfinite(mid):
                out[i, j] = tail + mid + fb * lb[j]
    return out


@dataclass
class TripLattice:
    """Everything the Viterbi pass needs that doesn't depend on the scoring parameters.
    Built once per trip; decoding with new beta / weights is then pure numpy."""
    starts: np.ndarray
    ends: np.ndarray
    gc: np.ndarray                      # straight-line distance fix i-1 -> i
    dists: list                         # network distance matrix per step (None = chain restart)
    uturn: list                         # bool matrix per step: candidate j is the reverse twin of i


def build_lattice(g: RoadGraph, cand_obs, cand_arc, cand_frac, track_xy, segment, twin,
                  p: HMMParams) -> TripLattice:
    n_obs = len(track_xy)
    starts = np.searchsorted(cand_obs, np.arange(n_obs), side="left")
    ends = np.searchsorted(cand_obs, np.arange(n_obs), side="right")
    gc = np.r_[0.0, np.linalg.norm(np.diff(track_xy, axis=0), axis=1)]
    dists, uturn = [None] * n_obs, [None] * n_obs
    for i in range(1, n_obs):
        a_rows = np.arange(starts[i - 1], ends[i - 1])
        b_rows = np.arange(starts[i], ends[i])
        if len(a_rows) == 0 or len(b_rows) == 0 or segment[i] != segment[i - 1]:
            continue
        dists[i] = step_distances(g, cand_arc[a_rows], cand_frac[a_rows],
                                  cand_arc[b_rows], cand_frac[b_rows], gc[i], p)
        uturn[i] = twin[cand_arc[a_rows]][:, None] == cand_arc[b_rows][None, :]
    return TripLattice(starts, ends, gc, dists, uturn)


def decode(lat: TripLattice, log_emit: np.ndarray, p: HMMParams):
    """Viterbi over all fixes of a trip. Returns per-fix chosen candidate row
    (-1 if the fix has no candidates) and the number of forced chain restarts."""
    starts, ends = lat.starts, lat.ends
    n_obs = len(starts)
    choice = np.full(n_obs, -1, dtype=np.int64)
    breaks = 0

    back: list[np.ndarray | None] = [None] * n_obs
    delta = None
    chain_start = 0

    def backtrack(last_fix: int, delta_last: np.ndarray):
        k = int(np.argmax(delta_last))
        i = last_fix
        while True:
            choice[i] = starts[i] + k
            if i == chain_start or back[i] is None:
                break
            k = int(back[i][k])
            i -= 1

    for i in range(n_obs):
        rows = np.arange(starts[i], ends[i])
        if len(rows) == 0:
            if delta is not None:
                backtrack(i - 1, delta)
            delta, chain_start = None, i + 1
            continue
        emit = p.emission_weight * log_emit[rows]
        new_chain = delta is None or lat.dists[i] is None
        if not new_chain:
            trans = -np.abs(lat.dists[i] - lat.gc[i]) / p.beta - p.uturn_penalty * lat.uturn[i]
            score = delta[:, None] + trans
            best_prev = np.argmax(score, axis=0)
            best = score[best_prev, np.arange(len(rows))]
            if np.all(~np.isfinite(best)):
                breaks += 1
                new_chain = True
            else:
                back[i] = best_prev
                delta = best + emit
        if new_chain:
            if delta is not None:
                backtrack(i - 1, delta)
            chain_start = i
            back[i] = None
            delta = emit.copy()
    if delta is not None:
        backtrack(n_obs - 1, delta)
    return choice, breaks


def stitch_route(g: RoadGraph, fix_arcs: np.ndarray, segment: np.ndarray, cutoff: float = 5000.0):
    """Turn per-fix arcs into a drivable arc sequence: collapse repeats and fill the arcs
    driven between consecutive matched arcs with the shortest connecting path."""
    route: list[int] = []
    prev, prev_seg = -1, None
    for a, s in zip(fix_arcs, segment):
        if a < 0:
            continue
        a = int(a)
        if a == prev:
            continue
        if prev >= 0 and s == prev_seg:
            route.extend(g.path_arcs(int(g.to[prev]), int(g.frm[a]), cutoff))
        route.append(a)
        prev, prev_seg = a, s
    return np.asarray(route, dtype=np.int64)
