"""CSBP711 Assignment 1: algorithm comparison (with training time and model size) and ablation.

Uses exactly the Phase 1 pipeline: Change_Log-reconstructed planning-time scope,
the per-project time-ordered 70/30 split, the same preprocessing and seed (42)
for every model. Two outcomes:

  late_closure  population: all modeled sprints (test n = 786)
  spillover     population: sprints with committed scope (test n = 511)

Table 1 (assignment_comparison.csv): task metrics, measured training time
(median of 5 fits, model fitting only) and model size for every algorithm.
Table 2 (assignment_ablation.csv): feature-group ablation of the random forest
that tests the explanation "spillover is a scope effect, late closure a
project/context effect" and the resulting change in the model ranking.

XGBoost is included only when the package is installed; otherwise it is
skipped and the rest of the table is unaffected.

Usage:
    python -m scripts.run_assignment_comparison
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.run_baseline import time_ordered_split
from src.baselines import heuristic_predict, make_logreg, make_random_forest
from src.data.load_tawos import load_reconstructed_sprints
from src.evaluate import evaluate
from src.features import (
    ALL_FEATURES,
    HISTORICAL_FEATURES,
    PLANNING_TIME_FEATURES,
    add_features,
    build_model_frame,
)
from src.labels import add_labels

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEED = 42
N_TIMING_REPEATS = 5

SCOPE_ONLY_FEATURES = ["committed_issue_count", "committed_story_points", "sprint_length_days"]
SCOPE_FEATURES = ["committed_issue_count", "committed_story_points"]

TASKS = {
    "late_closure": ("label_delay", "full"),
    "spillover": ("label_spillover", "nonempty"),
}


def make_scope_only_logreg() -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), SCOPE_ONLY_FEATURES)])),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def make_rf_on(numeric: list[str], with_project: bool) -> Pipeline:
    transformers = [("num", StandardScaler(), numeric)]
    if with_project:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), ["project"]))
    return Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3,
                                       class_weight="balanced", random_state=SEED)),
    ])


def model_size(model: Pipeline | None) -> dict:
    """Parameter count / structure size and pickled size of a fitted pipeline."""
    if model is None:
        return {"size_description": "no fitted model", "n_parameters": 0, "pickled_kb": 0.0}
    clf = model.named_steps["clf"]
    kb = round(len(pickle.dumps(model)) / 1024, 1)
    if hasattr(clf, "coef_"):
        n = int(clf.coef_.size + clf.intercept_.size)
        return {"size_description": f"{n} coefficients", "n_parameters": n, "pickled_kb": kb}
    if hasattr(clf, "estimators_") and hasattr(clf.estimators_[0], "tree_"):
        nodes = int(sum(e.tree_.node_count for e in clf.estimators_))
        return {"size_description": f"{len(clf.estimators_)} trees / {nodes} nodes",
                "n_parameters": nodes, "pickled_kb": kb}
    booster = clf.get_booster()
    nodes = int(len(booster.trees_to_dataframe()))
    return {"size_description": f"{booster.num_boosted_rounds()} trees / {nodes} nodes",
            "n_parameters": nodes, "pickled_kb": kb}


def fit_timed(factory, X, y):
    """Fit ``N_TIMING_REPEATS`` times; return the last model and the median fit time."""
    times, model = [], None
    for _ in range(N_TIMING_REPEATS):
        model = factory()
        t0 = perf_counter()
        model.fit(X, y)
        times.append(perf_counter() - t0)
    return model, float(median(times))


def algorithms():
    algos = {
        "heuristic": None,
        "scope_only_logreg": (make_scope_only_logreg, SCOPE_ONLY_FEATURES),
        "logreg": (make_logreg, ALL_FEATURES),
        "random_forest": (make_random_forest, ALL_FEATURES),
    }
    try:
        from src.baselines import make_xgboost
        import xgboost  # noqa: F401
        algos["xgboost"] = (make_xgboost, ALL_FEATURES)
    except ImportError:
        print("xgboost not installed: skipping the XGBoost row")
    return algos


def population_frames():
    featured = add_features(add_labels(load_reconstructed_sprints()))
    out = {}
    for task, (label, pop) in TASKS.items():
        frame = build_model_frame(featured, label)
        if pop == "nonempty":
            frame = frame[frame["committed_issue_count"] > 0]
        out[task] = (label,) + time_ordered_split(frame.reset_index(drop=True))
    return out


def comparison(frames) -> pd.DataFrame:
    rows = []
    algos = algorithms()
    for task, (label, train, test) in frames.items():
        y_test = test[label]
        # trivial baseline: always predict the positive class
        base = evaluate(y_test, np.ones(len(test), dtype=int), np.full(len(test), 0.5))
        rows.append({"task": task, "model": "always_positive", "role": "baseline", "n_train": len(train),
                     "n_test": len(test), **{k: base[k] for k in ("positive_rate", "precision", "recall", "f1")},
                     "roc_auc": 0.5, "pr_auc": float("nan"), "brier": float("nan"),
                     "training_time_seconds": 0.0, **model_size(None)})
        for name, spec in algos.items():
            if spec is None:   # heuristic: trailing 3-sprint completion ratio < 0.8, nothing fitted
                m = evaluate(y_test, heuristic_predict(test))
                rows.append({"task": task, "model": name, "role": "baseline", "n_train": len(train),
                             "n_test": len(test), **m, "training_time_seconds": 0.0, **model_size(None)})
                continue
            factory, cols = spec
            model, seconds = fit_timed(factory, train[cols], train[label])
            proba = model.predict_proba(test[cols])[:, 1]
            m = evaluate(y_test, (proba >= 0.5).astype(int), proba)
            rows.append({"task": task, "model": name,
                         "role": "baseline" if name == "scope_only_logreg" else "learned",
                         "n_train": len(train), "n_test": len(test), **m,
                         "training_time_seconds": seconds, **model_size(model)})
    return pd.DataFrame(rows)


ABLATIONS = {
    "full": (PLANNING_TIME_FEATURES + HISTORICAL_FEATURES, True),
    "planning_only": (PLANNING_TIME_FEATURES, True),
    "history_only": (HISTORICAL_FEATURES, True),
    "no_project": (PLANNING_TIME_FEATURES + HISTORICAL_FEATURES, False),
    "no_scope": ([c for c in PLANNING_TIME_FEATURES + HISTORICAL_FEATURES if c not in SCOPE_FEATURES], True),
}


def ablation(frames) -> pd.DataFrame:
    rows = []
    for task, (label, train, test) in frames.items():
        ref = make_scope_only_logreg().fit(train[SCOPE_ONLY_FEATURES], train[label])
        ref_auc = float(evaluate(test[label], (ref.predict_proba(test[SCOPE_ONLY_FEATURES])[:, 1] >= 0.5).astype(int),
                                 ref.predict_proba(test[SCOPE_ONLY_FEATURES])[:, 1])["roc_auc"])
        for name, (numeric, with_project) in ABLATIONS.items():
            cols = numeric + (["project"] if with_project else [])
            model = make_rf_on(numeric, with_project).fit(train[cols], train[label])
            proba = model.predict_proba(test[cols])[:, 1]
            m = evaluate(test[label], (proba >= 0.5).astype(int), proba)
            rows.append({"task": task, "rf_features": name, "n_features": len(cols), "n_test": len(test),
                         "f1": m["f1"], "roc_auc": m["roc_auc"],
                         "scope_only_logreg_auc": ref_auc,
                         "rf_minus_scope_only_logreg_auc": m["roc_auc"] - ref_auc})
    return pd.DataFrame(rows)


def main() -> None:
    frames = population_frames()
    comp = comparison(frames)
    abl = ablation(frames)
    comp.to_csv(RESULTS_DIR / "assignment_comparison.csv", index=False)
    abl.to_csv(RESULTS_DIR / "assignment_ablation.csv", index=False)
    (RESULTS_DIR / "assignment_summary.json").write_text(json.dumps({
        "seed": SEED, "timing_repeats": N_TIMING_REPEATS,
        "test_sizes": {t: int(len(v[2])) for t, v in frames.items()},
    }, indent=2))
    pd.set_option("display.width", 220)
    print(comp[["task", "model", "role", "f1", "roc_auc", "training_time_seconds",
                "size_description", "pickled_kb"]].round(4).to_string(index=False))
    print()
    print(abl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
