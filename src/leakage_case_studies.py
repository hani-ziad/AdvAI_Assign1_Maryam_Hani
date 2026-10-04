"""
Two concrete, verified leakage mechanisms from the related-work literature,
demonstrated on the same data and pipeline -- this is the
evidence behind the "contribution" claim in docs/leakage_case_studies.md:
published sprint-risk results may look strong partly because of avoidable
methodological mistakes, not because the underlying task is easy.

Case study A ("post-outcome feature"): Perez Castillo et al. (2024, MDPI
Information 15(11):726) build all 11 of their sprint-classification
features from the sprint's own burndown/completion chart -- their own
Table 1 names feature R1 as "Percentage of work completed by the end of
the Sprint". That is the sprint's outcome, not a planning-time signal.
We reproduce the mechanism (not their exact feature set, which needs
day-by-day burndown data this dataset doesn't have) by adding our own
sprint's post-hoc completion_ratio into the feature set and showing how
much scores inflate.

Case study B ("oversample before split"): Obike, Ekong & Obot (2025,
IJCA 187(49)) apply SMOTE to generate 5,328 balanced samples from 969
PROMISE instances with no explicit train/test split described before
that step -- a classic order-of-operations leakage bug (oversampling the
whole pool lets near-duplicate synthetic points land in both train and
test). We demonstrate the same mechanism with plain duplication-based
oversampling (no need for PROMISE/COQUINA data, or even SMOTE itself --
the bug is about *when* you resample, not which resampling algorithm)
applied in the wrong order vs. the right order on our own split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

from src.evaluate import evaluate
from src.features import ALL_FEATURES


def case_study_a_post_outcome_feature(
    frame: pd.DataFrame, label_col: str, train_idx, test_idx
) -> dict:
    """Compare the leakage-safe feature set against the same set plus the
    sprint's own post-hoc completion_ratio (the mechanism behind Perez
    Castillo et al.'s R1 feature)."""
    train, test = frame.loc[train_idx], frame.loc[test_idx]

    results = {}
    for name, feature_cols in [
        ("leakage_safe (the standard protocol used here)", ALL_FEATURES),
        ("leaky (+ this sprint's own completion_ratio)", ALL_FEATURES + ["completion_ratio"]),
    ]:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        X_train = pd.get_dummies(train[feature_cols])
        X_test = pd.get_dummies(test[feature_cols]).reindex(columns=X_train.columns, fill_value=0)
        model.fit(X_train, train[label_col])
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = evaluate(test[label_col], pred, proba)
    return results


def case_study_b_oversample_order(frame: pd.DataFrame, label_col: str, train_idx, test_idx) -> dict:
    """Compare oversampling the minority class AFTER splitting (correct)
    vs. BEFORE splitting (the Obike et al. mechanism: oversample the whole
    pool, then split -- duplicated rows can land on both sides)."""
    train, test = frame.loc[train_idx].copy(), frame.loc[test_idx].copy()
    feature_cols = ALL_FEATURES

    def fit_and_eval(train_df, test_df):
        model = LogisticRegression(max_iter=1000)
        X_train = pd.get_dummies(train_df[feature_cols])
        X_test = pd.get_dummies(test_df[feature_cols]).reindex(columns=X_train.columns, fill_value=0)
        model.fit(X_train, train_df[label_col])
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return evaluate(test_df[label_col], pred, proba)

    results = {}

    # Correct order: split first, oversample only the training fold.
    minority = train[train[label_col] == 1]
    majority = train[train[label_col] == 0]
    if len(minority) and len(minority) < len(majority):
        minority_upsampled = resample(
            minority, replace=True, n_samples=len(majority), random_state=42
        )
        train_balanced = pd.concat([majority, minority_upsampled])
    else:
        train_balanced = train
    results["correct order (oversample train only, after split)"] = fit_and_eval(train_balanced, test)

    # Incorrect order: oversample the WHOLE pool, then split -- duplicated
    # rows (literal copies here, near-duplicate synthetic points in a real
    # SMOTE pipeline) can end up on both sides of the split.
    pool = pd.concat([train, test])
    minority_pool = pool[pool[label_col] == 1]
    majority_pool = pool[pool[label_col] == 0]
    if len(minority_pool) and len(minority_pool) < len(majority_pool):
        minority_pool_upsampled = resample(
            minority_pool, replace=True, n_samples=len(majority_pool), random_state=42
        )
        pool_balanced = pd.concat([majority_pool, minority_pool_upsampled]).sample(
            frac=1, random_state=42
        )
    else:
        pool_balanced = pool
    cut = int(len(pool_balanced) * 0.7)
    leaky_train, leaky_test = pool_balanced.iloc[:cut], pool_balanced.iloc[cut:]
    results["incorrect order (oversample whole pool, then split)"] = fit_and_eval(
        leaky_train, leaky_test
    )

    return results
