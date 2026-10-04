"""Feature engineering for sprint-risk prediction.

Correction: cold-start filtering is now explicit. The first eligible sprint
per project is the only row removed for having no predecessor; rows are no
longer dropped merely because the previous sprint had zero linked issues.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLANNING_TIME_FEATURES = [
    "committed_issue_count",
    "committed_story_points",
    "sprint_length_days",
]

HISTORICAL_FEATURES = [
    "prior_avg_delivered_sp_3",
    "prior_avg_completion_ratio_3",
    "prior_sprint_carryover",
    "prior_sprint_completion_ratio",
]

CATEGORICAL_FEATURES = ["project"]
ALL_FEATURES = PLANNING_TIME_FEATURES + HISTORICAL_FEATURES + CATEGORICAL_FEATURES


def add_features(labeled_sprints: pd.DataFrame) -> pd.DataFrame:
    df = labeled_sprints.sort_values(["project", "start_date", "sprint_id"]).copy()

    # A zero-scope sprint has no incomplete committed work. Its raw 0/0
    # completion ratio is undefined, but for *historical* completion behavior
    # we use 1.0 so that the next sprint is not incorrectly discarded. The
    # zero-scope fact remains visible through committed_issue_count itself.
    history_completion_ratio = df["completion_ratio"].fillna(1.0)

    grp = df.groupby("project", group_keys=False)
    prior_delivered = grp["delivered_story_points"].shift(1)
    prior_ratio = history_completion_ratio.groupby(df["project"]).shift(1)

    df["prior_avg_delivered_sp_3"] = (
        prior_delivered.groupby(df["project"])
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["prior_avg_completion_ratio_3"] = (
        prior_ratio.groupby(df["project"])
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["prior_sprint_carryover"] = grp["not_completed_count"].shift(1)
    df["prior_sprint_completion_ratio"] = prior_ratio

    # Explicitly identify the true cold start instead of using NaN in one
    # particular historical feature as a proxy for "no predecessor".
    df["has_prior_sprint"] = df.groupby("project").cumcount() > 0
    return df


def build_model_frame(featured_sprints: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Return model columns, dropping only the true first sprint per project."""
    cols = ALL_FEATURES + [label_col, "project", "sprint_id", "start_date", "has_prior_sprint"]
    df = featured_sprints[list(dict.fromkeys(cols))].copy()
    df = df[df["has_prior_sprint"]].copy()

    # After the explicit first-sprint removal, feature NaNs indicate a real
    # data/engineering problem and should not silently remove hundreds of rows.
    feature_na = df[ALL_FEATURES].isna().any(axis=1)
    if feature_na.any():
        bad = df.loc[feature_na, ["project", "sprint_id"] + ALL_FEATURES].head(10)
        raise ValueError(
            "Unexpected missing feature values after explicit cold-start filtering. "
            f"First examples:\n{bad.to_string(index=False)}"
        )
    return df.reset_index(drop=True)
