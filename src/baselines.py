"""
Models compared for sprint-risk prediction, per the Week 8 proposal's
Section 3 (Baselines) and Section 4 (Planned method -> Models):

  1. heuristic     - trailing 3-sprint average completion ratio < 0.8
  2. logreg        - logistic regression on the planning-time feature set
  3. random_forest - tree-ensemble reference (bagging)
  4. xgboost       - gradient-boosted trees (boosting), the strongest model
     named in the proposal's related work. Not installable in the development environment
     that built the Week 10 stage (see
     docs/data_access_and_versioning.md); added for the Week 12 real-TAWOS
     run once available on a normal machine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.features import CATEGORICAL_FEATURES, HISTORICAL_FEATURES, PLANNING_TIME_FEATURES

NUMERIC_FEATURES = PLANNING_TIME_FEATURES + HISTORICAL_FEATURES


def heuristic_predict(frame: pd.DataFrame) -> np.ndarray:
    """No learning: flag at-risk if trailing 3-sprint completion ratio < 0.8."""
    return (frame["prior_avg_completion_ratio_3"] < 0.8).astype(int).to_numpy()


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def make_logreg() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", _make_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def make_random_forest() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", _make_preprocessor()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def make_xgboost() -> Pipeline:
    # Imported lazily so that importing this module -- and using logreg /
    # random_forest -- doesn't require xgboost to be installed at all. Only
    # calling make_xgboost() itself does.
    from xgboost import XGBClassifier

    return Pipeline(
        steps=[
            ("prep", _make_preprocessor()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )


MODEL_FACTORIES = {
    "logreg": make_logreg,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
}
