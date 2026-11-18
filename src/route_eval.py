"""Route-level evaluation: classifier-only vs classifier + HMM/Viterbi.

Decoder parameters are tuned on the 80 train trips using out-of-fold classifier
probabilities (each trip scored by a model that never saw its fold), then the chosen
configuration is evaluated once on student-test.

Usage:
  python -m src.route_eval --tune      # OOF scores + decoder grid on train
  python -m src.route_eval --test      # final comparison on student-test
"""
import argparse
import itertools
import json
import time
from dataclasses import asdict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_features import trip_table
from .config import load_config
from .data_prep import list_trips, load_trip
from .features import FEATURES, reverse_twin
from .hmm import HMMParams, RoadGraph, build_lattice, decode, stitch_route
from .labels import label_trip
from .metrics import fix_groups, top1_rows
from .pipeline import fix_log_probs
from .preprocess import prepare_trip
from .train import fit, sample_fixes

GRID = {
    "beta": [5.0, 10.0, 20.0, 40.0, 80.0],
    "emission_weight": [0.5, 1.0, 2.0],
    "uturn_penalty": [0.0, 3.0, 10.0],
}


# ---------- metrics ----------
def same_arc(pred, lab, arcs):
    ok = pred >= 0
    p = np.where(ok, pred, 0)
    return ok & ((p == lab) | ((arcs[p, 0] == arcs[lab, 0]) & (arcs[p, 1] == arcs[lab, 1])))


def continuity(fix_arcs, segment, g: RoadGraph) -> float:
    a, b = fix_arcs[:-1], fix_arcs[1:]
    m = (segment[1:] == segment[:-1]) & (a >= 0) & (b >= 0)
    a, b = a[m], b[m]
    return float(np.mean((a == b) | (g.to[a] == g.frm[b]))) if m.any() else np.nan


def route_prf(pred_route, ref_route, arcs, length):
    """Length-weighted precision / recall of the reconstructed arc sequence against the
    reference route. Arcs with the same (from, to) nodes are treated as the same road."""
    n = int(arcs.max()) + 1
    key = lambda r: arcs[r, 0].astype(np.int64) * n + arcs[r, 1]
    pk, rk = np.unique(key(pred_route)), np.unique(key(ref_route))
    lp = {k: l for k, l in zip(key(pred_route), length[pred_route])}
    lr = {k: l for k, l in zip(key(ref_route), length[ref_route])}
    inter = np.intersect1d(pk, rk)
    prec = sum(lp[k] for k in inter) / max(sum(lp.values()), 1e-9)
    rec = sum(lr[k] for k in inter) / max(sum(lr.values()), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1


# ---------- per-trip state ----------
class TripCase:
    """Everything needed to decode a trip repeatedly with different HMM parameters."""

    def __init__(self, trip_dir, cfg, params: HMMParams):
        ref = load_trip(trip_dir)                    # with route: used ONLY for scoring below
        trip = load_trip(trip_dir, with_route=False)
        self.trip_id = trip.trip_id
        self.df, self.prep, _ = trip_table(trip, cfg, with_labels=False)
        labels = label_trip(ref)
        self.label_arc = labels["arc_id"]
        self.valid = prepare_trip(ref, cfg, labels)["valid"]
        self.ref_route = ref.route
        self.arcs = ref.arcs
        self.g = RoadGraph(trip.node_xy, trip.arcs)
        t0 = time.time()
        self.lat = build_lattice(self.g, self.df.obs.to_numpy(), self.df.arc.to_numpy(),
                                 self.df.frac.to_numpy(), trip.track_xy, self.prep["segment"],
                                 reverse_twin(trip.arcs), params)
        self.lattice_s = time.time() - t0
        self.n_obs = trip.n_obs
        self.prob = None

        self.prob_by: dict[str, np.ndarray] = {}

    def classifier_arcs(self):
        out = np.full(self.n_obs, -1, dtype=np.int64)
        best = top1_rows(fix_groups(self.df), self.prob)
        out[self.df.obs.to_numpy()[best]] = self.df.arc.to_numpy()[best]
        return out

    def evaluate(self, params: HMMParams, with_route: bool = True) -> dict:
        seg = self.prep["segment"]
        clf = self.classifier_arcs()
        choice, breaks = decode(self.lat, fix_log_probs(self.df, self.prob), params)
        hmm = np.where(choice >= 0, self.df.arc.to_numpy()[np.maximum(choice, 0)], -1)
        v = self.valid
        res = {
            "trip_id": self.trip_id, "n_fixes": self.n_obs, "n_valid": int(v.sum()),
            "clf_correct": int(same_arc(clf, self.label_arc, self.arcs)[v].sum()),
            "hmm_correct": int(same_arc(hmm, self.label_arc, self.arcs)[v].sum()),
            "clf_continuity": continuity(clf, seg, self.g),
            "hmm_continuity": continuity(hmm, seg, self.g),
            "chain_breaks": breaks,
        }
        if with_route and v.any():
            length = self.g.length
            for tag, arcs_fix in [("clf", clf), ("hmm", hmm)]:
                route = stitch_route(self.g, arcs_fix, seg)
                res[f"{tag}_route_precision"], res[f"{tag}_route_recall"], res[f"{tag}_route_f1"] = \
                    route_prf(route, self.ref_route, self.arcs, length)
        return res


def summarise(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out = {
        "clf_top1": df.clf_correct.sum() / df.n_valid.sum(),
        "hmm_top1": df.hmm_correct.sum() / df.n_valid.sum(),
        "clf_continuity": df.clf_continuity.mean(),
        "hmm_continuity": df.hmm_continuity.mean(),
        "chain_breaks": int(df.chain_breaks.sum()),
    }
    for c in ["clf_route_f1", "hmm_route_f1", "clf_route_precision", "hmm_route_precision",
              "clf_route_recall", "hmm_route_recall"]:
        if c in df:
            out[c] = df.loc[df.n_valid > 0, c].mean()
    return out


# ---------- tuning on train (out-of-fold) ----------
def classifier_candidates(cfg) -> dict:
    """Both LightGBM configurations go through the decoder: Bayesian tuning raised CV top-1
    by 0.18 pts (all 5 folds) but worsened log loss by ~10%, and the HMM consumes the
    probabilities, so the final pick is made on decoded accuracy, not classifier top-1."""
    best = json.loads((cfg["paths"]["reports"] / "tuning" / "best_params.json").read_text())["params"]
    return {"untuned": {}, "tuned": best}


def tune(cfg):
    out = cfg["paths"]["reports"] / "hmm"
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    candidates = classifier_candidates(cfg)
    train = pd.read_pickle(cfg["paths"]["artifacts"] / "cache" / "features_train.pkl")
    train_valid = train[train.valid].reset_index(drop=True)
    folds = pd.read_csv(cfg["paths"]["root_configs"] / "folds.csv", dtype={"trip_id": str})
    trip_dirs = {p.name: p for p in list_trips(cfg["paths"]["train_dir"])}
    base = HMMParams()

    cases = []
    for k in sorted(folds.fold.unique()):
        fold_train = sample_fixes(train_valid[train_valid.fold != k], 0.35, seed + k)
        models = {name: fit("lgbm", fold_train, seed, **params)[0] for name, params in candidates.items()}
        for tid in folds.trip_id[folds.fold == k]:
            c = TripCase(trip_dirs[tid], cfg, base)
            if not c.valid.any():          # trips 33, 54: no trustworthy label, nothing to tune on
                continue
            X = c.df[FEATURES].to_numpy(np.float32)
            c.prob_by = {name: m.predict_proba(X)[:, 1] for name, m in models.items()}
            cases.append(c)
        print(f"fold {k}: lattices built for {len(cases)} trips so far", flush=True)

    rows = []
    for name in candidates:
        for c in cases:
            c.prob = c.prob_by[name]
        for beta, w, u in itertools.product(GRID["beta"], GRID["emission_weight"], GRID["uturn_penalty"]):
            p = HMMParams(beta=beta, emission_weight=w, uturn_penalty=u)
            s = summarise([c.evaluate(p, with_route=False) for c in cases])
            s.update(classifier=name, beta=beta, emission_weight=w, uturn_penalty=u)
            rows.append(s)
            print(f"{name:7s} beta={beta:5.1f} w={w:.1f} u={u:4.1f}  hmm_top1={s['hmm_top1']:.4f} "
                  f"(clf {s['clf_top1']:.4f}) cont={s['hmm_continuity']:.4f}", flush=True)
    grid = pd.DataFrame(rows).sort_values("hmm_top1", ascending=False)
    grid.to_csv(out / "decoder_grid_train_oof.csv", index=False)
    best = grid.iloc[0]
    chosen = HMMParams(beta=float(best.beta), emission_weight=float(best.emission_weight),
                       uturn_penalty=float(best.uturn_penalty))
    chosen_clf = str(best.classifier)
    (out / "hmm_params.json").write_text(json.dumps(asdict(chosen), indent=2))

    # deployment model for the chosen classifier: its params, every valid training fix
    model_file = f"lgbm_{chosen_clf}_full.joblib"
    if not (cfg["paths"]["artifacts"] / "models" / model_file).exists():
        full, _ = fit("lgbm", train_valid, seed, **candidates[chosen_clf])
        joblib.dump(full, cfg["paths"]["artifacts"] / "models" / model_file)
    (out / "decoder_choice.json").write_text(json.dumps(
        {"classifier": chosen_clf, "classifier_params": candidates[chosen_clf], "model_file": model_file,
         "hmm_params": asdict(chosen), "selected_on": "train out-of-fold HMM top-1 (valid fixes)",
         "train_oof_hmm_top1": float(best.hmm_top1), "train_oof_clf_top1": float(best.clf_top1)}, indent=2))

    for c in cases:
        c.prob = c.prob_by[chosen_clf]
    per_trip = pd.DataFrame([c.evaluate(chosen) for c in cases])
    per_trip.to_csv(out / "train_oof_per_trip.csv", index=False)
    print("chosen:", chosen_clf, asdict(chosen))
    print(grid.groupby("classifier")[["clf_top1", "hmm_top1"]].max().round(4).to_string())
    print(pd.Series(summarise(per_trip.to_dict("records"))).round(4).to_string())

    # sensitivity plot: top-1 vs beta for each emission weight (chosen classifier and u-turn)
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = grid[(grid.uturn_penalty == chosen.uturn_penalty) & (grid.classifier == chosen_clf)]
    for w, g in sub.groupby("emission_weight"):
        g = g.sort_values("beta")
        ax.plot(g.beta, g.hmm_top1, marker="o", label=f"emission weight {w}")
    ax.axhline(sub.clf_top1.iloc[0], color="grey", ls="--", label="classifier only")
    ax.set_xscale("log"); ax.set_xlabel("beta (m)"); ax.set_ylabel("OOF top-1 (train)")
    ax.set_title(f"HMM decoder sensitivity ({chosen_clf} LightGBM, u-turn penalty {chosen.uturn_penalty})")
    ax.legend(); fig.tight_layout()
    fig.savefig(cfg["paths"]["reports"] / "figures" / "hmm_decoder_sensitivity.png", dpi=120)


# ---------- final test ----------
def test(cfg):
    out = cfg["paths"]["reports"] / "hmm"
    choice = json.loads((out / "decoder_choice.json").read_text())
    params = HMMParams(**choice["hmm_params"])
    model = joblib.load(cfg["paths"]["artifacts"] / "models" / choice["model_file"])
    print("student-test with", choice["model_file"], choice["hmm_params"])
    rows = []
    for d in list_trips(cfg["paths"]["student_test_dir"]):
        c = TripCase(d, cfg, params)
        t0 = time.time()
        c.prob = model.predict_proba(c.df[FEATURES].to_numpy(np.float32))[:, 1]
        r = c.evaluate(params)
        r["decode_s"] = round(time.time() - t0 + c.lattice_s, 2)
        rows.append(r)
    per_trip = pd.DataFrame(rows)
    per_trip["clf_top1"] = per_trip.clf_correct / per_trip.n_valid
    per_trip["hmm_top1"] = per_trip.hmm_correct / per_trip.n_valid
    per_trip.to_csv(out / "student_test_per_trip.csv", index=False)
    summary = summarise(rows)
    pd.Series(summary).to_csv(out / "student_test_summary.csv", header=["value"])
    print(per_trip[["trip_id", "n_fixes", "clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity",
                    "clf_route_f1", "hmm_route_f1", "chain_breaks", "decode_s"]].round(4).to_string(index=False))
    print(pd.Series(summary).round(4).to_string())

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(per_trip)); w = 0.4
    ax.bar(x - w / 2, per_trip.clf_top1, w, label="classifier only")
    ax.bar(x + w / 2, per_trip.hmm_top1, w, label="classifier + HMM/Viterbi")
    ax.set_xticks(x, per_trip.trip_id, rotation=45); ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("top-1 accuracy"); ax.set_title("Student-test: effect of HMM decoding per trip")
    ax.legend(); fig.tight_layout()
    fig.savefig(cfg["paths"]["reports"] / "figures" / "hmm_vs_classifier_student_test.png", dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.tune:
        tune(cfg)
    if args.test:
        test(cfg)
