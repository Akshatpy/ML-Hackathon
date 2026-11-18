"""End-to-end inference for one trip: raw files -> per-fix arcs + drivable route.

Used by evaluation and by the API. It never reads the .route file: trips are loaded
with with_route=False and the feature table is built without labels.
"""
import time
from dataclasses import asdict

import joblib
import numpy as np

from .build_features import trip_table
from .features import FEATURES, reverse_twin
from .hmm import HMMParams, RoadGraph, build_lattice, decode, stitch_route
from .metrics import fix_groups, top1_rows

EPS = 1e-6


def fix_log_probs(df, prob: np.ndarray) -> np.ndarray:
    """Classifier probabilities renormalised over each fix's candidates, in log space.
    The classifier scores candidates independently, so raw probabilities at one fix
    needn't sum to 1; the HMM needs a distribution over that fix's states."""
    g = fix_groups(df)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    p = np.clip(prob, EPS, 1.0)
    return np.log(p) - np.log(np.add.reduceat(p, starts))[g]


class MapMatcher:
    def __init__(self, model_path, cfg: dict, params: HMMParams | None = None):
        self.model = joblib.load(model_path)
        self.cfg = cfg
        self.params = params or HMMParams()

    def score(self, trip):
        t0 = time.time()
        df, prep, cand = trip_table(trip, self.cfg, with_labels=False)
        t_feat = time.time() - t0
        t0 = time.time()
        prob = self.model.predict_proba(df[FEATURES].to_numpy(np.float32))[:, 1]
        t_clf = time.time() - t0
        return df, prep, prob, {"features_s": t_feat, "classifier_s": t_clf}

    def lattice(self, trip, df, prep):
        g = RoadGraph(trip.node_xy, trip.arcs)
        twin = reverse_twin(trip.arcs)
        lat = build_lattice(g, df.obs.to_numpy(), df.arc.to_numpy(), df.frac.to_numpy(),
                            trip.track_xy, prep["segment"], twin, self.params)
        return g, lat

    def match(self, trip) -> dict:
        df, prep, prob, timing = self.score(trip)
        n = trip.n_obs
        obs = df.obs.to_numpy()

        clf_arc = np.full(n, -1, dtype=np.int64)
        best = top1_rows(fix_groups(df), prob)
        clf_arc[obs[best]] = df.arc.to_numpy()[best]

        t0 = time.time()
        g, lat = self.lattice(trip, df, prep)
        choice, breaks = decode(lat, fix_log_probs(df, prob), self.params)
        hmm_arc = np.where(choice >= 0, df.arc.to_numpy()[np.maximum(choice, 0)], -1)
        route = stitch_route(g, hmm_arc, prep["segment"])
        timing["hmm_s"] = time.time() - t0

        return {
            "trip_id": trip.trip_id,
            "n_fixes": n,
            "classifier_arcs": clf_arc,
            "hmm_arcs": hmm_arc,
            "route_arcs": route,
            "segments": prep["segment"],
            "chain_breaks": breaks,
            "fixes_without_candidates": int((choice < 0).sum()),
            "timing": timing,
            "params": asdict(self.params),
        }
