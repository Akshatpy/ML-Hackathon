# PathSnap: map matching and route reconstruction

PathSnap takes a noisy 1 Hz GPS track and the road graph around it, and works out which road
segment (arc) the vehicle was on at every fix and which route it drove. A LightGBM classifier
scores candidate arcs for each fix; an HMM decoded with Viterbi turns those scores into a
connected route. A FastAPI backend serves predictions to a small web UI.

UE24CS352A Machine Learning hackathon, PES University.

## Results (student-test, 10 trips, observation-level top-1 accuracy)

| approach | top-1 |
|---|---|
| nearest arc | 0.535 |
| nearest arc whose direction agrees with the GPS heading | 0.857 |
| Logistic Regression | 0.874 |
| LightGBM (selected classifier) | 0.908 |
| **LightGBM + HMM/Viterbi** | **0.952** |

With the HMM, route F1 (length-weighted, against the reference route) goes from 0.850 to 0.988,
with no chain breaks on student-test. Under 10 s per trip including the shortest-path search.

Eight classifiers were compared on 5-fold trip-grouped CV; see `reports/models/FINDINGS.md`.
Bayesian tuning (Optuna TPE, 25 trials): `reports/tuning/FINDINGS.md`. Decoder evaluation:
`reports/hmm/`.

## Repository layout

```
configs/            config.yaml (all settings), folds.csv (CV fold per train trip)
src/
  data_prep.py      read .track/.nodes/.arcs/.route, per-trip metric projection
  spatial.py        KD-tree index over road arcs, point-to-segment distance
  labels.py         per-fix reference labels from the ordered route (monotone DP alignment)
  eda.py            dataset exploration -> reports/eda
  validation.py     trip-grouped CV folds (trips sharing roads kept together)
  preprocess.py     gap segmentation, adaptive heading, speed, label validity
  candidates.py     candidate arcs within 50 m of each fix
  features.py       29 features per (fix, candidate)
  build_features.py feature tables for train / student_test
  models.py         the 8 classifiers compared
  train.py          grouped CV + student-test evaluation
  tune.py           Bayesian hyperparameter search (Optuna TPE)
  hmm.py            road graph, shortest paths, Viterbi decoder, route stitching
  pipeline.py       inference for one trip (no reference route involved)
  route_eval.py     decoder tuning on out-of-fold train scores, classifier vs HMM comparison
  viz.py, prediction_maps.py   Folium maps
api/                FastAPI app (main.py) and server-side scoring (scoring.py)
ui/                 single-page web UI (index.html)
scripts/            EDA extras, map generation, start_api.ps1
reports/            findings, tables, figures, maps, logs
artifacts/models/   trained models (Random Forest excluded: 80 MB, rebuild with src.train)
```

## Setup

Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Raw data is not in the repository. Place it as:

```
Student folder/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
Student folder/student_test/<trip_id>/...
Evaluator folder/evaluator_test/<trip_id>/...      (only read by the API)
```

## Reproducing the results

```
python -m src.eda                  # EDA tables and figures (~5 min)
python scripts/eda_extras.py
python scripts/make_eda_maps.py
python -m src.validation           # CV folds
python -m src.build_features       # candidate feature tables (~1 min)
python -m src.feature_report
python -m src.train                # 8 models, grouped CV + student-test (~25 min)
python -m src.model_report
python -m src.tune --trials 26     # Bayesian search (~1 h on 6 cores)
python -m src.tune --finalise
python -m src.route_eval --tune --test   # decoder tuning + final comparison
python -m src.prediction_maps
```

Seeds are fixed (`seed: 42` in `configs/config.yaml`).

## Running the demo (two laptops)

**Laptop 1, API.** From the project root:

```
.\scripts\start_api.ps1
```

It prints the address to give to the UI, e.g. `http://192.168.43.17:8000`, and the request
log appears in the same window. Add `-Key <secret>` to require an access key, and `-Tunnel` to
also open a Cloudflare quick tunnel with a public https address (needs `cloudflared`).
If Windows blocks incoming connections, allow port 8000 once (admin PowerShell):

```
New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any
```

**Laptop 2, UI.** Open `ui/index.html` in a browser (or serve the folder with
`python -m http.server 8080`), enter laptop 1's address, click *Load trips*, pick a trip.
Both laptops must be on the same network; a phone hotspot avoids campus Wi-Fi client isolation.
The UI can also be hosted as a static site (e.g. Vercel, root directory `ui`), pointed at the
tunnel address.

### API

| method | path | purpose |
|---|---|---|
| GET | `/health` | loaded model and decoder settings |
| GET | `/trips?dataset=evaluator` | trip ids available for evaluation |
| POST | `/predict` | body `{"trip_id": "00000015", "dataset": "evaluator"}`; returns GPS points, predicted route, accuracy metrics and timings |

## Keeping the evaluation honest

- Prediction loads a trip without its `.route` file; features are built without labels.
- Accuracy is computed afterwards by `api/scoring.py` on the server. The response carries only
  the metrics; no reference geometry or labels are sent to the UI.
- Evaluator trips were never used for exploration, training, model selection or tuning.
- Two train trips (33, 54) have reference routes covering a fraction of the drive and are
  excluded from training and scoring; see `reports/features/FINDINGS.md`.
