"""Folium maps of predictions on student-test trips (report section 6).

Layers: road network near the track, reference route, classifier-only arcs, HMM route,
GPS fixes, and the fixes the HMM still gets wrong. Student-test reference routes are
allowed in the report; evaluator trips are never drawn with their reference.
Usage: python -m src.prediction_maps [trip_id ...]
"""
import json
import sys

import folium
import numpy as np

from .config import load_config
from .data_prep import list_trips, load_trip
from .hmm import HMMParams
from .labels import label_trip
from .pipeline import MapMatcher
from .route_eval import same_arc
from .viz import trip_map


def _arc_line(trip, a):
    u, v = trip.arcs[a]
    return [(trip.node_lonlat[u][1], trip.node_lonlat[u][0]), (trip.node_lonlat[v][1], trip.node_lonlat[v][0])]


def build(trip_id: str, matcher: MapMatcher, cfg) -> str:
    d = cfg["paths"]["student_test_dir"] / trip_id
    ref = load_trip(d)
    result = matcher.match(load_trip(d, with_route=False))
    fmap = trip_map(ref, predicted_route=result["route_arcs"])

    clf = folium.FeatureGroup(name="Classifier-only arc per fix", show=False)
    for a in np.unique(result["classifier_arcs"][result["classifier_arcs"] >= 0]):
        folium.PolyLine(_arc_line(ref, a), color="#7570b3", weight=3, opacity=0.8).add_to(clf)
    clf.add_to(fmap)

    labels = label_trip(ref)["arc_id"]
    wrong = ~same_arc(result["hmm_arcs"], labels, ref.arcs)
    err = folium.FeatureGroup(name=f"Fixes the HMM gets wrong ({int(wrong.sum())})", show=True)
    for k in np.flatnonzero(wrong):
        lon, lat = ref.track_lonlat[k]
        folium.CircleMarker((lat, lon), radius=4, color="#e7298a", fill=True, fill_opacity=0.9,
                            tooltip=f"fix {k}: predicted arc {result['hmm_arcs'][k]}, reference {labels[k]}").add_to(err)
    err.add_to(fmap)
    # trip_map already attached a LayerControl before these layers existed; drop it and add
    # one at the end so the control lists every layer
    for key in [k for k in fmap._children if k.startswith("layer_control")]:
        del fmap._children[key]
    folium.LayerControl(collapsed=False).add_to(fmap)

    out = cfg["paths"]["reports"] / "maps" / f"prediction_student_test_{trip_id}.html"
    fmap.save(str(out))
    return out.name


if __name__ == "__main__":
    cfg = load_config()
    choice = json.loads((cfg["paths"]["reports"] / "hmm" / "decoder_choice.json").read_text())
    matcher = MapMatcher(cfg["paths"]["artifacts"] / "models" / choice["model_file"], cfg,
                         HMMParams(**choice["hmm_params"]))
    ids = sys.argv[1:] or [p.name for p in list_trips(cfg["paths"]["student_test_dir"])]
    for tid in ids:
        print(build(tid, matcher, cfg))
