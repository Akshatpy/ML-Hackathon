# Bayesian hyperparameter optimisation (LightGBM)

Code: `src/tune.py`. Record: `optuna_lgbm.db` (Optuna SQLite study `lgbm_top1`, every trial),
`optuna_trials.csv` (same, flat), `best_params.json`, `tuned_vs_untuned.csv`, `run.log`.
Figures: `reports/figures/tuning_history.png`, `reports/figures/tuning_param_importance.png`.

## Setup
- **Optimiser:** Optuna TPE sampler (sequential model-based / Bayesian optimisation), seed 42,
  `multivariate=True`, first 10 trials random (startup), then guided by TPE.
- **Objective:** maximise mean observation-level top-1 accuracy over the 5 trip-grouped CV folds,
  using the same 35% fix sample per fold as the model comparison. Student-test is not used
  during the search.
- **Pruning:** MedianPruner, never before 2 folds are done (fold spread is about 3 pts).
- **Trial 0** was the untuned configuration, enqueued so the search starts from the baseline.
- **Budget:** 26 trials started: 20 complete, 5 pruned, 1 failed. Trial 1 was interrupted when the
  budget was cut from 40 to 25 (about 2.5 min per trial on a 6-core laptop). Trial 17 shows 97
  min because the laptop went to sleep during it; its result is unaffected.

## Search space
| parameter | range | reasoning |
|---|---|---|
| num_leaves | 15-255, log | tree complexity |
| min_child_samples | 20-1000, log | main control against overfitting (untuned fit train 5 pts above CV) |
| learning_rate | 0.02-0.15, log | trades off with the number of trees |
| n_estimators | 200-1000, step 100 | number of trees |
| colsample_bytree | 0.4-1.0 | feature subsampling |
| subsample | 0.5-1.0 | row subsampling |
| reg_lambda | 0.001-30, log | L2 penalty on leaf values |
| reg_alpha | 0.001-10, log | L1 penalty on leaf values |

## Best configuration (trial 25)
num_leaves 116, min_child_samples 51, learning_rate 0.058, n_estimators 800,
colsample_bytree 0.83, subsample 0.91, reg_lambda 0.023, reg_alpha 0.0095. CV top-1 0.9095.

## Tuned vs untuned (same folds, same data, same metrics)
| model | CV top-1 | CV macro F1 | CV log loss | test top-1 | test macro F1 | test log loss | test continuity | fit s | ms / 1k fixes |
|---|---|---|---|---|---|---|---|---|---|
| untuned (35% of fixes) | 0.9084 | 0.9487 | 0.0565 | 0.9092 | 0.9500 | 0.0460 | 0.9564 | 21 | 101 |
| tuned (35% of fixes) | 0.9095 | 0.9493 | 0.0627 | 0.9074 | 0.9490 | 0.0493 | 0.9568 | 52 | 353 |
| tuned, all valid fixes | n/a | n/a | n/a | 0.9079 | 0.9493 | 0.0470 | 0.9585 | 111 | 306 |

Per-fold CV top-1, trial 0 (untuned) -> trial 25 (best): +0.25, +0.05, +0.24, +0.19, +0.19 pts
(mean +0.18). The tuned model is better in all 5 folds.

## What this shows
- **The landscape is flat.** All 20 completed trials land between 0.9062 and 0.9095 CV top-1
  (std 0.09 pts across trials). The classifier was already close to what these 29 features
  allow; accuracy is driven by the features, not by hyperparameters.
- **The CV gain is small but consistent** (+0.18 pts, every fold). On the 10 student-test trips the
  tuned model is 0.18 pts *lower*. With only 10 trips, that is within noise; we don't select on it.
- **num_leaves is the most important parameter** (fANOVA importance 0.56), then
  min_child_samples (0.17) and n_estimators (0.09). The best trials use larger trees (95-223
  leaves) with moderate min_child_samples (26-73). That is the opposite of the heavy
  regularisation we expected to help, so the 5-pt train/CV gap was not hurting held-out accuracy.
- **Costs of tuning:** log loss is about 10% worse (0.0565 -> 0.0627 CV), and inference is
  3.5x slower (larger trees, 800 trees vs 400).

## Decision
Because the HMM consumes the probabilities, and tuning traded better top-1 for worse
calibration, the deployed classifier is chosen by **decoded** accuracy: both configurations go
through HMM decoder tuning on out-of-fold train probabilities, and the better pair wins.
See `reports/hmm/decoder_choice.json` and `reports/hmm/FINDINGS.md`.

Outcome: after decoding, tuned 0.9386 vs untuned 0.9383 OOF top-1. The tuned model is deployed
(`artifacts/models/lgbm_tuned_full.joblib`), but the two are effectively equal once the HMM is applied.
