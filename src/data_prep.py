"""Raw trip loading. One trip = one folder holding <id>.track/.nodes/.arcs/.route."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pyproj import Transformer

def local_crs(lon0: float, lat0: float) -> str:
    return f"+proj=aeqd +lat_0={lat0:.6f} +lon_0={lon0:.6f} +datum=WGS84 +units=m +no_defs"


def to_metric(lon: np.ndarray, lat: np.ndarray, crs: str) -> tuple[np.ndarray, np.ndarray]:
    # always_xy: files store lon first, and pyproj's EPSG:4326 axis order is lat-first otherwise
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)


@dataclass
class Trip:
    trip_id: str
    track_lonlat: np.ndarray      # (n_obs, 2)
    t: np.ndarray                 # (n_obs,) seconds
    node_lonlat: np.ndarray       # (n_nodes, 2)
    arcs: np.ndarray              # (n_arcs, 2) int, row index = arc_id
    route: np.ndarray | None      # ordered arc_ids, None if not supplied
    track_xy: np.ndarray = field(default=None, repr=False)
    node_xy: np.ndarray = field(default=None, repr=False)
    crs: str = ""

    @property
    def n_obs(self) -> int:
        return len(self.t)

    def arc_endpoints_xy(self, arc_ids=None) -> tuple[np.ndarray, np.ndarray]:
        arcs = self.arcs if arc_ids is None else self.arcs[arc_ids]
        return self.node_xy[arcs[:, 0]], self.node_xy[arcs[:, 1]]


def _read_matrix(path: Path, dtype) -> np.ndarray:
    # np.loadtxt is ~10x slower than this on the 160k-row arc files
    return np.fromfile(path, sep=" ", dtype=dtype) if path.stat().st_size else np.empty(0, dtype=dtype)


def load_trip(trip_dir: Path, with_route: bool = True) -> Trip:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read_matrix(trip_dir / f"{tid}.track", np.float64).reshape(-1, 3)
    nodes = _read_matrix(trip_dir / f"{tid}.nodes", np.float64).reshape(-1, 2)
    arcs = _read_matrix(trip_dir / f"{tid}.arcs", np.int64).reshape(-1, 2)
    route_path = trip_dir / f"{tid}.route"
    route = _read_matrix(route_path, np.int64) if (with_route and route_path.exists()) else None

    trip = Trip(tid, track[:, :2], track[:, 2], nodes, arcs, route)
    # centre on the GPS track, not the node cloud, so the track itself gets the least distortion
    crs = local_crs(*np.nanmean(track[:, :2], axis=0))
    trip.crs = crs
    trip.track_xy = np.column_stack(to_metric(track[:, 0], track[:, 1], crs))
    trip.node_xy = np.column_stack(to_metric(nodes[:, 0], nodes[:, 1], crs))
    return trip


def list_trips(split_dir: Path) -> list[Path]:
    return sorted(p for p in Path(split_dir).iterdir() if p.is_dir())
