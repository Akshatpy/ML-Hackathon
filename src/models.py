"""Candidate classifiers compared in the model phase.

Every model maps a candidate's feature vector to P(candidate is the reference arc).
Initial (untuned) settings are recorded here so the report can quote them.
"""
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

N_JOBS = 6

MODEL_INFO = {
    "logreg": ("Linear / probabilistic", "Logistic Regression, L2, C=1.0, standardised inputs"),
    "gnb": ("Probabilistic (generative)", "Gaussian Naive Bayes, var_smoothing=1e-9, standardised inputs"),
    "linsvm": ("Kernel-based (linear kernel)", "LinearSVC C=0.5 + sigmoid calibration (3-fold)"),
    "rf": ("Ensemble (bagging)", "Random Forest, 200 trees, min_samples_leaf=20, sqrt features"),
    "lgbm": ("Boosting", "LightGBM, 400 trees, lr=0.05, 63 leaves, 0.8 row/col subsample"),
    "xgb": ("Boosting", "XGBoost hist, 400 trees, lr=0.05, depth 7, 0.8 row/col subsample"),
    "mlp": ("Neural network", "MLP 64-32 ReLU, Adam, alpha=1e-4, batch 1024, early stopping"),
    "lgbm_rank": ("Boosting (learning-to-rank)", "LightGBM LambdaRank, groups = fixes, same trees as lgbm"),
}

# models whose decision_function is a ranking score, not a probability
RANKERS = {"lgbm_rank"}


def make_model(name: str, seed: int = 42, **overrides):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=overrides.get("C", 1.0), max_iter=500))
    if name == "gnb":
        return make_pipeline(StandardScaler(), GaussianNB())
    if name == "linsvm":
        # LinearSVC has no probabilities; sigmoid (Platt) calibration gives them for the HMM
        svm = LinearSVC(C=overrides.get("C", 0.5), dual=False)
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(svm, cv=3, method="sigmoid"))
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features="sqrt",
                                      n_jobs=N_JOBS, random_state=seed)
    if name == "lgbm":
        params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                      n_jobs=N_JOBS, random_state=seed, verbose=-1)
        params.update(overrides)
        return LGBMClassifier(**params)
    if name == "xgb":
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", n_jobs=N_JOBS,
                             random_state=seed, eval_metric="logloss")
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
            max_iter=60, early_stopping=True, validation_fraction=0.1, n_iter_no_change=6,
            random_state=seed))
    if name == "lgbm_rank":
        params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                      n_jobs=N_JOBS, random_state=seed, verbose=-1)
        params.update(overrides)
        return LGBMRanker(objective="lambdarank", lambdarank_truncation_level=3, **params)
    raise ValueError(name)


def predict_score(model, name: str, X):
    if name in RANKERS:
        return model.predict(X)
    return model.predict_proba(X)[:, 1]
