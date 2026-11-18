"""Folium maps: raw GPS fixes, local road network, reference route, predicted route."""
from pathlib import Path

import folium
import numpy as np


def _arc_latlon(trip, arc_ids):
    a = trip.node_lonlat[trip.arcs[arc_ids, 0]]
    b = trip.node_lonlat[trip.arcs[arc_ids, 1]]
    return [[(p[1], p[0]), (q[1], q[0])] for p, q in zip(a, b)]


def _route_polyline(trip, route):
    pts = [trip.node_lonlat[trip.arcs[route[0], 0]]] + [trip.node_lonlat[trip.arcs[r, 1]] for r in route]
    return [(p[1], p[0]) for p in pts]


def trip_map(trip, predicted_route=None, network_buffer_m: float = 150.0,
             max_network_arcs: int = 15000, gps_stride: int = 1) -> folium.Map:
    lat_c, lon_c = trip.track_lonlat[:, 1].mean(), trip.track_lonlat[:, 0].mean()
    fmap = folium.Map(location=[lat_c, lon_c], zoom_start=14, tiles="CartoDB positron")

    # Only draw arcs near the trajectory; the full trip graph (100k+ arcs) makes the HTML unusable
    from scipy.spatial import cKDTree
    mid = 0.5 * (trip.node_xy[trip.arcs[:, 0]] + trip.node_xy[trip.arcs[:, 1]])
    near = cKDTree(trip.track_xy).query(mid, distance_upper_bound=network_buffer_m)[0] < np.inf
    near_ids = np.flatnonzero(near)[:max_network_arcs]
    net = folium.FeatureGroup(name=f"Road network (<{network_buffer_m:.0f} m)", show=True)
    for seg in _arc_latlon(trip, near_ids):
        folium.PolyLine(seg, color="#9aa0a6", weight=1, opacity=0.6).add_to(net)
    net.add_to(fmap)

    if trip.route is not None:
        ref = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_route_polyline(trip, trip.route), color="#1b9e77", weight=6, opacity=0.8,
                        tooltip="reference route").add_to(ref)
        ref.add_to(fmap)

    if predicted_route is not None and len(predicted_route):
        pred = folium.FeatureGroup(name="Predicted route", show=True)
        for seg in _arc_latlon(trip, np.asarray(predicted_route)):
            folium.PolyLine(seg, color="#d95f02", weight=3, opacity=0.9).add_to(pred)
        pred.add_to(fmap)

    gps = folium.FeatureGroup(name="GPS observations", show=True)
    for k in range(0, trip.n_obs, gps_stride):
        lon, lat = trip.track_lonlat[k]
        folium.CircleMarker((lat, lon), radius=2, color="#3060d0", fill=True, fill_opacity=0.8,
                            tooltip=f"t={trip.t[k]:.0f}s").add_to(gps)
    gps.add_to(fmap)

    lo, hi = trip.track_lonlat.min(axis=0), trip.track_lonlat.max(axis=0)
    folium.Marker((trip.track_lonlat[0, 1], trip.track_lonlat[0, 0]), tooltip="start",
                  icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker((trip.track_lonlat[-1, 1], trip.track_lonlat[-1, 0]), tooltip="end",
                  icon=folium.Icon(color="red")).add_to(fmap)
    fmap.fit_bounds([[lo[1], lo[0]], [hi[1], hi[0]]])
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def save_trip_map(trip, out_path: Path, **kwargs) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trip_map(trip, **kwargs).save(str(out_path))
    return out_path
