"""EDA Folium maps for the report. Picks one clean trip and the two truncated-route trips."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data_prep import load_trip  # noqa: E402
from src.viz import save_trip_map  # noqa: E402

MAPS = {
    "train/00000001": "clean highway trip (Darmstadt), dense graph",
    "train/00000033": "reference route stops ~6 km into a 48 km drive",
    "train/00000054": "reference route covers 31 of 170 km",
    "student_test/00000052": "smallest student-test trip",
}

if __name__ == "__main__":
    cfg = load_config()
    out_dir = cfg["paths"]["reports"] / "maps"
    for rel, why in MAPS.items():
        trip = load_trip(cfg["paths"]["data_root"] / rel)
        stride = max(1, trip.n_obs // 3000)  # keep HTML light for the 19k-fix trips
        path = save_trip_map(trip, out_dir / f"eda_{rel.replace('/', '_')}.html", gps_stride=stride)
        print(f"{path.name}: {why}")
