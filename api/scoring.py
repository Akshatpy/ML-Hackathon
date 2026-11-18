"""Server-side scoring against the reference route.

Kept apart from the prediction path on purpose: `predict` in main.py runs the matcher
on a trip loaded WITHOUT its .route file, and only afterwards this module reads the
reference to compute accuracy. Nothing from here is sent to the UI except the numbers.
"""
import numpy as np

from src.data_prep import load_trip
from src.hmm import RoadGraph
from src.labels import label_trip
from src.preprocess import prepare_trip
from src.route_eval import continuity, route_prf, same_arc


def score_prediction(trip_dir, result: dict, cfg: dict) -> dict:
    ref = load_trip(trip_dir, with_route=True)
    if ref.route is None or len(ref.route) == 0:
        return {"available": False}
    labels = label_trip(ref)
    valid = prepare_trip(ref, cfg, labels)["valid"]
    g = RoadGraph(ref.node_xy, ref.arcs)
    seg = result["segments"]
    out = {"available": True, "n_fixes": int(ref.n_obs), "n_scored_fixes": int(valid.sum())}
    for tag in ("classifier", "hmm"):
        arcs = result[f"{tag}_arcs"]
        ok = same_arc(arcs, labels["arc_id"], ref.arcs)
        out[f"{tag}_top1_accuracy"] = float(ok[valid].mean()) if valid.any() else None
        out[f"{tag}_route_continuity"] = continuity(arcs, seg, g)
    p, r, f1 = route_prf(result["route_arcs"], ref.route, ref.arcs, g.length)
    out.update(route_precision=p, route_recall=r, route_f1=f1)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
