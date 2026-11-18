"""PathSnap API (laptop 1): hosts the tuned classifier, the HMM decoder and the
server-side evaluation. Run from the project root:

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health                 liveness + which model/params are loaded
  GET  /trips?dataset=...      trip ids available for evaluation (default: evaluator)
  POST /predict                {"trip_id": "...", "dataset": "evaluator"} -> route + metrics
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.scoring import score_prediction  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data_prep import list_trips, load_trip  # noqa: E402
from src.hmm import HMMParams  # noqa: E402
from src.pipeline import MapMatcher  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap")

cfg = load_config()
DATASETS = {
    "evaluator": cfg["paths"]["evaluator_dir"],
    "student_test": cfg["paths"]["student_test_dir"],
}
# The deployed classifier + decoder settings are whatever won on train out-of-fold decoding
# (reports/hmm/decoder_choice.json); PATHSNAP_MODEL overrides the model file for testing.
CHOICE_PATH = ROOT / "reports" / "hmm" / "decoder_choice.json"
choice = json.loads(CHOICE_PATH.read_text()) if CHOICE_PATH.exists() else {}
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL") or
                  ROOT / "artifacts" / "models" / choice.get("model_file", "lgbm_tuned_full.joblib"))
params = HMMParams(**choice["hmm_params"]) if "hmm_params" in choice else HMMParams()
matcher = MapMatcher(MODEL_PATH, cfg, params)
log.info("model loaded: %s | HMM params: %s", MODEL_PATH.name, params)

app = FastAPI(title="PathSnap map-matching API", version="1.0")
# the UI is served from another machine (laptop 2 or Vercel), so the browser needs CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Optional shared key. When the API is exposed through a public tunnel, anyone with the URL
# could call it; setting PATHSNAP_KEY on laptop 1 makes the UI send the same key.
API_KEY = os.environ.get("PATHSNAP_KEY", "")
if API_KEY:
    log.info("access key required (X-PathSnap-Key)")


def check_key(x_pathsnap_key: str = Header(default="")):
    if API_KEY and x_pathsnap_key != API_KEY:
        raise HTTPException(401, "missing or wrong access key")


def _client(request: Request) -> str:
    # behind a Cloudflare tunnel the real caller is in this header
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


class PredictRequest(BaseModel):
    trip_id: str
    dataset: str = "evaluator"


def _trip_dir(dataset: str, trip_id: str) -> Path:
    if dataset not in DATASETS:
        raise HTTPException(400, f"unknown dataset '{dataset}'")
    # trip ids are 8-digit folder names; reject anything else before touching the filesystem
    if not (trip_id.isdigit() and len(trip_id) == 8):
        raise HTTPException(400, "trip_id must be an 8-digit id")
    d = Path(DATASETS[dataset]) / trip_id
    if not d.is_dir():
        raise HTTPException(404, f"trip {trip_id} not found in {dataset}")
    return d


def _lines(trip, arc_ids: np.ndarray) -> list[list[list[float]]]:
    """Arc sequence -> polylines of [lat, lon]; a new line starts where arcs don't touch."""
    lines, cur, prev_to = [], [], None
    for a in arc_ids:
        u, v = trip.arcs[a]
        pu, pv = trip.node_lonlat[u], trip.node_lonlat[v]
        if prev_to is None or u != prev_to:
            if len(cur) > 1:
                lines.append(cur)
            cur = [[round(pu[1], 6), round(pu[0], 6)]]
        cur.append([round(pv[1], 6), round(pv[0], 6)])
        prev_to = v
    if len(cur) > 1:
        lines.append(cur)
    return lines


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "hmm_params": params.__dict__,
            "key_required": bool(API_KEY)}


@app.get("/trips", dependencies=[Depends(check_key)])
def trips(request: Request, dataset: str = "evaluator"):
    if dataset not in DATASETS:
        raise HTTPException(400, f"unknown dataset '{dataset}'")
    ids = [p.name for p in list_trips(DATASETS[dataset])]
    log.info("GET /trips dataset=%s from %s -> %d trips", dataset, _client(request), len(ids))
    return {"dataset": dataset, "trips": ids}


@app.post("/predict", dependencies=[Depends(check_key)])
def predict(req: PredictRequest, request: Request):
    t0 = time.time()
    d = _trip_dir(req.dataset, req.trip_id)
    log.info("POST /predict trip=%s dataset=%s from %s: running pipeline",
             req.trip_id, req.dataset, _client(request))

    # prediction path: the reference route is not loaded at all
    trip = load_trip(d, with_route=False)
    result = matcher.match(trip)
    t_pred = time.time() - t0

    # evaluation path: reads the reference server-side, returns numbers only
    metrics = score_prediction(d, result, cfg)
    total = time.time() - t0

    log.info("trip=%s fixes=%d route_arcs=%d breaks=%d | clf_top1=%s hmm_top1=%s route_f1=%s | %.2fs",
             req.trip_id, result["n_fixes"], len(result["route_arcs"]), result["chain_breaks"],
             metrics.get("classifier_top1_accuracy"), metrics.get("hmm_top1_accuracy"),
             metrics.get("route_f1"), total)

    stride = max(1, trip.n_obs // 3000)   # keep the payload small for the 4k+ fix trips
    gps = [[round(lat, 6), round(lon, 6)] for lon, lat in trip.track_lonlat[::stride]]
    return {
        "trip_id": req.trip_id,
        "dataset": req.dataset,
        "n_fixes": result["n_fixes"],
        "gps_points": gps,
        "predicted_route": _lines(trip, result["route_arcs"]),
        "n_route_arcs": int(len(result["route_arcs"])),
        "chain_breaks": result["chain_breaks"],
        "metrics": metrics,
        "timing_s": {**{k: round(v, 3) for k, v in result["timing"].items()},
                     "prediction_total": round(t_pred, 3), "with_scoring": round(total, 3)},
        "model": MODEL_PATH.name,
    }
