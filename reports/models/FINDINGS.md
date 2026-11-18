# Model comparison

Code: `src/models.py` (definitions), `src/train.py` (CV + student-test), `src/metrics.py`,
`src/model_report.py` (figures). Raw results: `model_results.csv` (one row per model),
`model_folds.csv` (one row per model x fold), `train.log`, `mlp_history.csv`.
Figures: `reports/figures/model_comparison.png`, `model_confusion_matrices.png`, `mlp_history.png`.

## Setup
- Task per row: P(candidate arc = reference arc), 29 features, about 12 candidates per fix.
- 5-fold trip-grouped CV (`configs/folds.csv`); trips that share roads are in the same fold.
- Every model trains on the same 35% of fixes per fold (whole fixes with all their candidates;
  seed 42 + fold). After CV, each model is refitted on all 80 train trips (same share) and scored
  once on the 10 student-test trips.
- Metrics use valid fixes only (see `reports/features/FINDINGS.md`). Route continuity uses every fix.
- Selection metric: **CV observation-level top-1 accuracy**. Student-test is reported but not
  used to pick, so the choice doesn't overfit those 10 trips.

## Results (untuned models)
| model | family | CV top-1 (mean +- std) | CV macro F1 | CV log loss | train-sample top-1 | test top-1 | test macro F1 | test log loss | test continuity | fit (s) | infer ms / 1k fixes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM | boosting | **0.9084 +- 0.030** | **0.9487** | 0.0565 | 0.961 | 0.9092 | 0.9500 | **0.0460** | 0.9564 | 21 | 101 |
| XGBoost | boosting | 0.9078 +- 0.030 | 0.9484 | **0.0564** | 0.953 | **0.9101** | **0.9505** | 0.0464 | **0.9573** | 34 | 69 |
| LightGBM LambdaRank | learning-to-rank | 0.9060 +- 0.033 | 0.9474 | n/a | 0.988 | 0.9057 | 0.9481 | n/a | 0.9534 | 25 | 115 |
| Random Forest | bagging | 0.9053 +- 0.027 | 0.9470 | 0.0579 | 0.924 | 0.8949 | 0.9421 | 0.0501 | 0.9526 | 259 | 72 |
| MLP 64-32 | neural network | 0.9008 +- 0.030 | 0.9445 | 0.0646 | 0.916 | 0.8946 | 0.9420 | 0.0540 | 0.9517 | 109 | 11 |
| Logistic Regression | linear | 0.8869 +- 0.022 | 0.9368 | 0.0850 | 0.887 | 0.8744 | 0.9309 | 0.0717 | 0.9438 | 4 | 6 |
| Linear SVM (Platt) | kernel, linear | 0.8865 +- 0.021 | 0.9365 | 0.0849 | 0.886 | 0.8742 | 0.9308 | 0.0731 | 0.9435 | 17 | 12 |
| Gaussian NB | generative | 0.8639 +- 0.019 | 0.9240 | 0.7220 | 0.865 | 0.8393 | 0.9115 | 0.7442 | 0.9345 | 1 | 11 |

Reference points: nearest arc 0.535 and heading rule 0.857 (student-test top-1).
Macro F1 is the candidate-level F1 averaged over both classes, using the top-1 decision.
Log loss is n/a for the ranker because its scores are not probabilities.

## Observations
- **Non-linear models beat linear ones by 2-3 pts in every fold.** LightGBM - LogReg per fold:
  +2.4, +3.5, +1.1, +1.4, +2.3 pts. The useful signal is an interaction ("close AND heading
  agrees AND neighbours agree"), which a weighted sum can't express.
- **Gaussian NB is worst** and badly calibrated (log loss 0.72 vs 0.05). Its independence
  assumption breaks on our features: dist, log_dist and dist_rank all carry the same information,
  so NB counts the same evidence several times and becomes overconfident.
- **LightGBM vs XGBoost is a tie.** Per-fold difference: +0.17, -0.03, +0.21, -0.14, +0.07 pts
  (mean +0.06). Student-test differs by 0.09 pts in the other direction. Neither difference is
  meaningful.
- **Ranking objective didn't help.** LambdaRank is 0.2 pts below the LightGBM classifier on CV,
  and overfits more (train 0.988 vs CV 0.906). It also gives no probabilities, which the HMM needs.
- **Overfitting check** (train-sample top-1 vs CV top-1): RF (0.924 vs 0.905), MLP (0.916 vs
  0.901) and the linear models are close. LightGBM (0.961 vs 0.908) and XGBoost (0.953 vs 0.908)
  fit their training sample noticeably better than held-out trips. That's a variance gap that the
  Bayesian tuning should target (min_child_samples, regularisation, number of leaves).
- **Fold 2 is hardest for every model** (0.86-0.87 vs 0.88-0.94). It holds trips 8 and 84: slow
  city driving where 36-42% of fixes have the reverse twin as the nearest arc. Std across folds
  (about 0.03) is about 10x the gap between the top three models, so small CV differences mean little.
- **MLP training history** (`mlp_history.png`): training loss falls steadily and held-out accuracy
  plateaus; early stopping ends training before the 60-epoch cap. It is the only model here with
  epoch-wise curves.

## Selection: LightGBM classifier
- Highest CV top-1 and CV macro F1 (the pre-declared selection metric), tied with XGBoost within noise.
- Best calibrated probabilities on student-test (log loss 0.046), which matters because the
  HMM uses these probabilities directly as emission scores.
- Faster to fit than XGBoost (21 s vs 34 s), which allows more trials in the Bayesian search.
- Inference of about 0.1 s per 1,000 fixes is far inside what the API needs (largest
  student-test trip: about 4,000 fixes).
- Weakness to fix in tuning: the train-CV gap above.

## Effect of the trip-level label guard
First run (`v1_before_trip_filter/`) vs final run, same code and seeds:

| model | CV top-1 before | CV top-1 after | test top-1 before | test top-1 after |
|---|---|---|---|---|
| LightGBM | 0.9030 | 0.9084 | 0.9117 | 0.9092 |
| XGBoost | 0.9019 | 0.9078 | 0.9109 | 0.9101 |
| Random Forest | 0.8991 | 0.9053 | 0.8946 | 0.8949 |
| Logistic Regression | 0.8782 | 0.8869 | 0.8720 | 0.8744 |

CV went up for every model partly because trip 33's unreliable fixes no longer count against
them, so the CV numbers are not directly comparable between the two runs. Student-test labels
were unaffected by the guard, and test top-1 moved by at most 0.3 pts (LightGBM -0.25 pts:
slightly fewer training rows; others up). The ranking of models is unchanged.
