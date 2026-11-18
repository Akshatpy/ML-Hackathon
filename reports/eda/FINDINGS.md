# EDA findings and decisions

Produced by `python -m src.eda` and `python -m src.validation`. Numbers below are from
`trip_inventory.csv`, `summary.json` and `route_overlap_pairs.csv` in this folder.
Partitions: train = 80 trips, student_test = 10 trips. Evaluator trips not available to us.

## Inventory
- Every trip folder has all four files (.track, .nodes, .arcs, .route), including student_test.
- No NaN rows, no non-monotonic timestamps, no out-of-range node or arc ids.
- Train: median 1,529 fixes/trip (361 to 19,313). Student-test: median 1,420 (262 to 3,958).
- Road graphs: median 19.5k arcs/trip in train, max 307k (trip 13). About 92% of arcs have a
  reverse twin (two-way streets stored as two directed arcs).

## Issues found and what we do about them
| # | Issue | Evidence | Decision |
|---|-------|----------|----------|
| 1 | Trips are worldwide, not one city | lon -100..137, lat -35..61; trip 57 (Ecuador) projects to inf in UTM 32N | Per-trip azimuthal-equidistant projection centred on the track |
| 2 | `.route` is an arc sequence, not per-fix labels | trip 1: 1,315 fixes vs 525 route arcs; ~10% of route arcs get no fix | Monotone DP alignment of fixes to route positions (`src/labels.py`) |
| 3 | Some reference routes are truncated | trip 33: 48 km driven, 6.4 km of route; trip 54: 170 km vs 31 km; tail of trip 2 | Fix is labelled only if within 50 m of its aligned arc; others excluded from training and scoring |
| 4 | Direction ambiguity | the nearest arc is the reverse twin of the label for ~28% (train) / 32% (test) of fixes; naive nearest-arc top-1 is 0.50 / 0.53 | Heading-vs-bearing features and directed transitions in the HMM are mandatory |
| 5 | Irregular sampling in some train trips | dt=1 s for 97% of steps overall, but trip 90 has a 42 h gap, trip 78 a 14.7 h gap, trip 81 only 72% 1-s steps | Split tracks into segments at large gaps for decoding; time-normalised features |
| 6 | Speed outliers | trip 32: 365 steps above 60 m/s, trip 13: 103 | Speed features clipped; not used to drop fixes |
| 7 | Graph cropping | graph = track bbox + ~0.01 deg; up to 421 weak components; trip 5 route only 47% inside the largest SCC | Transitions use bounded shortest paths with a fallback for unreachable pairs, never "main component only" |
| 8 | Reference route has occasional jumps | 13 train trips + 1 test trip have 1-3 disconnected consecutive arcs | Continuity of the reference itself is reported alongside ours |
| 9 | Train trips share roads | 10/16/29 (Moscow) share up to 56% of route nodes; 53/93 33%; 43/54 19% | Overlapping trips merged into one CV group |
| 10 | Train is noisier than student_test | test: max gap 210 s, invalid-label rate 0.14% at 50 m; train: 11.2% (mostly trips 33, 54) | Report metrics on valid fixes; check per-fold spread before trusting CV |

## Candidate radius
Miss rate (label arc farther than r) on student_test: 2.9% at 20 m, 0.77% at 30 m,
0.14% at 50 m, 0.09% at 100 m. Mean candidates per fix: 6.8 at 30 m, 12 at 50 m, 31 at 100 m.
**50 m** chosen as the starting point: the miss rate is near its floor while the candidate set is still small.

Among valid fixes, the label is the closest arc 55% of the time and the second closest 37%
(almost always the reverse twin). This is where the classifier has to earn its keep.

## Validation strategy
- 5-fold GroupKFold over the 80 train trips, groups = trips merged by road overlap (`configs/folds.csv`).
- student_test is used for development-stage comparison only, never for fitting.
- Primary selection metric: observation-level Top-1 accuracy after Viterbi on valid fixes.
  Candidate-level macro F1 and log loss are reported for the classifiers.
