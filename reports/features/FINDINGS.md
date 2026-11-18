# Preprocessing and feature engineering

Code: `src/preprocess.py`, `src/candidates.py`, `src/features.py`, `src/build_features.py`,
`src/feature_report.py`. Settings in `configs/config.yaml` (`preprocess`, `candidates`, `features`).

## Preprocessing
- Each track is cut into segments wherever consecutive fixes are more than 30 s apart.
  Student-test has at most 2 segments per trip; train trips have up to 38 (trip 90: 42 h hole).
  The decoder restarts at each segment, and neighbour-based features never look across a cut.
- Heading (adaptive, the final setting): bearing of the displacement between fixes i-w and i+w
  in the same segment, with w widened through 2, 4, 8, 15 until the vehicle has moved at least 8 m.
  If even w=15 is too short (a real stop), the last known heading in the segment is carried
  forward (or the next one backward at the segment start). `heading_carried` flags those fixes.
- Speed is clipped at 60 m/s; gaps and step lengths are capped. No fix is ever dropped, since the
  API must return a segment for every observation.
- A fix is *valid* (used for training and scoring) if its aligned reference arc is within 50 m,
  **and** its trip's median fix-to-label distance is also within 50 m. The trip-level rule only
  catches trips 33 (median 122 m) and 54 (140 m); every other trip's median is under 12 m. It was
  added after the first model run showed every model at 51% top-1 on trip 33's per-fix-"valid"
  fixes: the broken route matched those fixes to the wrong place even within 50 m.
  Valid share: 86.7% of train fixes (trips 33 and 54 fully excluded, about 25k fixes),
  99.86% of student-test fixes.

## Candidates
- Every directed arc within 50 m. If none, the 5 nearest within 300 m (fallback).
- Train: 1,957,275 candidate rows over 208,417 fixes (12.0 per fix, 9.5% positives).
  Student-test: 209,038 rows over 19,160 fixes (11.8 per fix).
- Candidate miss rate on valid fixes: 0.00% student-test, 0.01% train (worst trip 0.44%).
- Duplicate arcs with identical (from, to) nodes count as the same arc when labelling.

## Features (29 used)
| group | features | why |
|---|---|---|
| geometry | dist, log_dist | GPS error is ~5 m median, so distance matters but is not decisive |
| relative | dist_rank, dist_minus_best, dist_ratio_best, cos_minus_best, ctx_mean_minus_best | the classifier scores one candidate at a time; these tell it how it compares with its competitors |
| direction | cos_heading, abs_heading_diff, heading_ok, heading_carried, along_speed | separates a street from its reverse twin |
| context | ctx_dist_mean, ctx_dist_max, ctx_prev_dist, ctx_next_dist | fixes i-3..i+3 measured against the same arc |
| topology | arc_len, frac, dist_to_arc_end, from_out_deg, to_out_deg, to_in_deg, has_twin, n_cand, n_cand_15m | intersections and dense areas are where errors concentrate |
| kinematics | speed, turn, dt_prev, step_prev | heading reliability depends on motion |

Dropped: `twin_in_cands`. It was identical to `has_twin`, because a twin shares the geometry and is always a candidate.

## Evidence (valid fixes; `feature_auc.csv`, `rule_baselines.csv`, `figures/feature_auc.png`)
Best single-feature AUCs (train / student-test): dist_minus_best 0.933 / 0.943,
ctx_mean_minus_best 0.930 / 0.943, dist 0.897 / 0.905, cos_heading 0.838 / 0.852.

Observation-level top-1 of hand rules, no learning:

| rule | train | student-test |
|---|---|---|
| nearest arc | 0.554 | 0.535 |
| nearest arc where heading agrees | 0.868 | 0.857 |
| window-mean distance + heading | 0.859 | 0.848 |

(Final settings: adaptive heading plus the trip-level label guard. The adaptive-vs-fixed A/B
below was run before the guard existed, so its train column is 0.007 lower; student-test is
unaffected. The fixed-window outputs are in `rule_baselines_fixed.csv` / `feature_auc_fixed.csv`.)

## Error analysis of the first version (fixed +-2 heading window; train error 15.1%)
- 64% of errors are within 15 m of a road node: the rule picks the arc before or after the true
  one at an intersection. Single-fix features can't settle this; graph transitions in the HMM can.
- 37% of errors are at speed < 2 m/s (12% of fixes, 47% error rate): stopped vehicles have no usable heading.
- 31% of errors are at fixes with 20 or more candidates (dense centres).

## Experiment: adaptive heading window
Hypothesis: slow-fix errors come from noise-dominated short displacements, so a wider window
plus carrying the heading through stops should fix them without hurting moving fixes.

| metric | fixed +-2 | adaptive | change |
|---|---|---|---|
| heading rule top-1, train | 0.8494 | 0.8612 | +1.18 pts |
| heading rule top-1, student-test | 0.8482 | 0.8567 | +0.85 pts |
| rule error at speed < 2 m/s (train) | 47.4% | 37.4% | -10.0 pts |
| rule error at speed >= 2 m/s (train) | 10.8% | 10.8% | 0 |
| cos_minus_best AUC, train / test | 0.854 / 0.861 | 0.872 / 0.871 | |

Kept. The gain comes entirely from slow fixes, as predicted. What remains is mostly intersection
ambiguity, which is the decoder's job.
