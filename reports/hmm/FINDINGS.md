# HMM / Viterbi route reconstruction

Code: `src/hmm.py` (graph, shortest paths, lattice, Viterbi, route stitching), `src/pipeline.py`
(inference for one trip), `src/route_eval.py` (decoder tuning and evaluation).
Outputs: `decoder_grid_train_oof.csv`, `decoder_choice.json`, `hmm_params.json`,
`train_oof_per_trip.csv`, `student_test_per_trip.csv`, `student_test_summary.csv`, `run.log`.
Figures: `reports/figures/hmm_decoder_sensitivity.png`, `hmm_vs_classifier_student_test.png`.

## Model
- **Hidden states** at fix i: the fix's candidate arcs (all directed arcs within 50 m, about 12),
  each with the fix's projection on the arc.
- **Emission:** `w_e * log P(arc | fix)`, where P is the LightGBM probability renormalised over
  the fix's candidates.
- **Transition** a -> b between consecutive fixes:
  `-|d_route(a, b) - d_straight| / beta - u * [b is the reverse twin of a]`, where `d_route` is
  the driving distance from a's projection to b's projection (rest of a, shortest path between
  them, part of b) and `d_straight` is the straight-line distance between the two fixes.
  Apparent backward motion of up to 15 m on the same arc counts as GPS noise.
- **Shortest paths:** Dijkstra on the directed graph, limited to 3 x d_straight + 200 m (at most
  5 km). Searches are cached per source node and resumed when a larger radius is needed.
  92 co-located duplicate nodes are merged so roads that meet are connected.
- **Viterbi:** `delta_i(b) = max_a [delta_{i-1}(a) + T(a, b)] + E_i(b)` with back-pointers.
- **Gaps and failures:**
  - a time gap above 30 s starts a new chain
  - a fix with no candidates ends the chain and gets no arc
  - if no transition into fix i is feasible, the chain restarts at i from emissions alone
    (counted as a *chain break*)
- **Route:** per-fix arcs, repeats collapsed, gaps filled with the shortest connecting path.

## Decoder tuning (train only, out-of-fold)
- Every train trip is scored by a LightGBM trained without its CV fold (same folds and 35%
  sample as before), so the decoder is tuned on honest probabilities. Trips 33 and 54 have no
  trustworthy labels and are skipped (78 trips).
- Both classifier configurations were decoded, because tuning improved CV top-1 but worsened
  calibration: untuned and Bayesian-tuned LightGBM.
- Grid: beta {5, 10, 20, 40, 80} m x emission weight {0.5, 1, 2} x U-turn penalty {0, 3, 10},
  45 settings per classifier, objective = pooled top-1 on valid fixes.
- Network distances are computed once per trip (the lattice), so each setting is a fast re-decode.

| classifier | classifier-only OOF top-1 | best HMM OOF top-1 |
|---|---|---|
| untuned LightGBM | 0.9056 | 0.9383 (beta 10, w 0.5, u 10) |
| **tuned LightGBM** | 0.9067 | **0.9386** (beta 5, w 0.5, u 10) |

**Chosen:** tuned LightGBM, beta = 5 m, emission weight = 0.5, U-turn penalty = 10. The two
classifiers end within 0.03 pts of each other after decoding; the HMM washes out the
calibration difference.

**Sensitivity:** every one of the 90 settings beats classifier-only (worst tuned setting 0.9196
vs 0.9067). Accuracy rises as beta falls (stricter "drive about the straight-line distance") and
as the emission weight falls (trust the road network more than any single fix's classifier score).
A higher U-turn penalty also helps. **Limitation:** the best values sit at the edge of the grid on
all three axes (smallest beta and weight, largest penalty), so a slightly better setting may lie
just outside it. Not explored for time reasons.

Train OOF with the chosen setting (78 trips):
| metric | classifier only | + HMM |
|---|---|---|
| top-1 (valid fixes) | 0.9067 | **0.9386** |
| route continuity | 0.9499 | 0.9731 |
| route precision (length-weighted) | 0.7366 | 0.9611 |
| route recall | 0.9911 | 0.9931 |
| route F1 | 0.8290 | 0.9757 |
| chain breaks | | 81 |

## Student-test (10 trips, deployed model trained on all valid train fixes)
| metric | classifier only | + HMM |
|---|---|---|
| **top-1 (valid fixes)** | 0.9079 | **0.9519** |
| route continuity | 0.9574 | 0.9785 |
| route precision | 0.7521 | 0.9801 |
| route recall | 0.9978 | 0.9966 |
| route F1 | 0.8504 | 0.9883 |
| chain breaks | | 0 |

Per trip (`student_test_per_trip.csv`): HMM top-1 is higher on 9 of 10 trips. Biggest gain on trip
77 (0.804 -> 0.919); trip 52 reaches 1.000. Trip 65 is flat (0.867 -> 0.866). Decoding takes
0.4-9.9 s per trip including the shortest-path lattice.

## Where the HMM helps and where it doesn't (train OOF, 78 trips)
- Better on 70 trips, worse on 8. Largest gains on the slow city trips that were hardest for the
  classifier: 16 (0.625 -> 0.907), 10 (0.702 -> 0.834), 2 (0.794 -> 0.923), 8 (0.714 -> 0.838).
  These are the reverse-twin / junction cases single-fix features can't resolve.
- Worst cases: trip 82 (0.847 -> 0.785, 5 chain breaks), 83 (-2.9 pts), 79 (-1.9 pts). When the
  classifier is right but a shortest path is missing or long (cropped graph, odd geometry), the
  transition score can pull the chain onto a wrong but well-connected road.
- 49 of the 81 chain breaks are in trip 43, the most fragmented graph in the dataset (421 weakly
  connected components): consecutive fixes' candidates often have no path between them inside
  the crop. Trip 2 (12 breaks) has 78 sampling gaps.

## Why precision jumps so much
Classifier-only picks arcs independently, so its stitched route includes spurious detours to
whichever wrong arc won at a single fix; those inflate the route length outside the reference
(precision 0.75). The HMM only chooses sequences that are drivable and plausible in length, so
almost all reconstructed length lies on the reference route (precision 0.98) while recall stays
at about 0.997.
