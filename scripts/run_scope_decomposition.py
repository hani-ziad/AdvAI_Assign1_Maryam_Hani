"""
Round-5 review addition: decompose the spillover task into the two
sub-problems a single all-sprint AUC conflates, and re-run spillover under
the zero-committed-scope sensitivity analysis the review asked for.

In the final modeled cohort, 911 of the 2,602 modeled sprints (35.0%) have
committed_issue_count == 0. Because label_spillover is mechanically false
whenever there is no reconstructed committed work, these zero-scope rows are
spillover-negative BY CONSTRUCTION, not because a model predicted anything.

This script reports two things separately:

Task A -- "does this sprint have any committed issue at all?"
Task B -- "given a sprint that committed at least one issue, does it spill over?"

Both tasks use the same time-ordered per-project split and LOPO scheme as
every other result in this repository.

Usage:
    python -m scripts.run_scope_decomposition
"""

from __future__ import annotations

import json
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from scripts.run_baseline import time_ordered_split
from src.baselines import heuristic_predict
from src.data.load_tawos import load_reconstructed_sprints
from src.evaluate import evaluate
from src.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    HISTORICAL_FEATURES,
    add_features,
    build_model_frame,
)
from src.labels import add_labels


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

N_BOOTSTRAP = 2000
RNG_SEED = 42


# Task A deliberately excludes committed_story_points because it is
# deterministically 0 whenever committed_issue_count == 0.
TASK_A_FEATURES = HISTORICAL_FEATURES + ["sprint_length_days"]
TASK_A_CATEGORICAL = CATEGORICAL_FEATURES


def _make_preprocessor(
    numeric_features,
    categorical_features,
) -> ColumnTransformer:

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )


def _make_logreg(
    numeric_features,
    categorical_features,
) -> Pipeline:

    return Pipeline(
        steps=[
            (
                "prep",
                _make_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _make_rf(
    numeric_features,
    categorical_features,
) -> Pipeline:

    return Pipeline(
        steps=[
            (
                "prep",
                _make_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
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


def bootstrap_ci_multi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:

    """Percentile-method 95% CIs for F1, ROC-AUC, PR-AUC and Brier."""

    from sklearn.metrics import (
        f1_score,
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
    )

    n = len(y_true)

    f1s = []
    aucs = []
    praucs = []
    briers = []

    for _ in range(n_boot):

        idx = rng.integers(
            0,
            n,
            n,
        )

        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_proba[idx]

        if len(np.unique(yt)) < 2:
            continue

        f1s.append(
            f1_score(
                yt,
                yp,
                zero_division=0,
            )
        )

        aucs.append(
            roc_auc_score(
                yt,
                ypr,
            )
        )

        praucs.append(
            average_precision_score(
                yt,
                ypr,
            )
        )

        briers.append(
            brier_score_loss(
                yt,
                ypr,
            )
        )

    out = {}

    for name, vals in [
        ("f1", f1s),
        ("roc_auc", aucs),
        ("pr_auc", praucs),
        ("brier", briers),
    ]:

        arr = np.array(vals)

        out[f"{name}_mean"] = (
            float(arr.mean())
            if len(arr)
            else float("nan")
        )

        out[f"{name}_ci95"] = (
            [
                float(
                    np.percentile(
                        arr,
                        2.5,
                    )
                ),
                float(
                    np.percentile(
                        arr,
                        97.5,
                    )
                ),
            ]
            if len(arr)
            else [
                float("nan"),
                float("nan"),
            ]
        )

    out["n_valid_resamples"] = int(
        len(f1s)
    )

    return out


def run_time_ordered(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str,
    numeric_features,
    categorical_features,
    rng: np.random.Generator,
) -> dict:

    row = {}

    # Heuristic does not require training.
    row["heuristic"] = evaluate(
        test[label_col],
        heuristic_predict(test),
    )

    row["heuristic"][
        "training_time_seconds"
    ] = 0.0

    feat_cols = (
        numeric_features
        + categorical_features
    )

    for name, factory in [
        ("logreg", _make_logreg),
        ("random_forest", _make_rf),
    ]:

        model = factory(
            numeric_features,
            categorical_features,
        )

        # -----------------------------------------
        # Measure FITTING time only
        # -----------------------------------------
        start = perf_counter()

        model.fit(
            train[feat_cols],
            train[label_col],
        )

        training_time = (
            perf_counter() - start
        )

        # -----------------------------------------
        # Prediction
        # -----------------------------------------
        proba = model.predict_proba(
            test[feat_cols]
        )[:, 1]

        pred = (
            proba >= 0.5
        ).astype(int)

        # -----------------------------------------
        # Evaluation
        # -----------------------------------------
        m = evaluate(
            test[label_col],
            pred,
            proba,
        )

        m[
            "training_time_seconds"
        ] = training_time

        m[
            "bootstrap_ci95"
        ] = bootstrap_ci_multi(
            test[
                label_col
            ].to_numpy(),
            pred,
            proba,
            N_BOOTSTRAP,
            rng,
        )

        row[name] = m

        print(
            f"{label_col} | "
            f"{name} | "
            f"training time: "
            f"{training_time:.4f} seconds"
        )

    return row


def run_lopo(
    frame: pd.DataFrame,
    label_col: str,
    numeric_features,
    categorical_features,
) -> dict:

    feat_cols = (
        numeric_features
        + categorical_features
    )

    lopo = {}

    for held_out in frame[
        "project"
    ].unique():

        train = frame[
            frame["project"]
            != held_out
        ]

        test = frame[
            frame["project"]
            == held_out
        ]

        if (
            len(train) == 0
            or len(test) == 0
            or test[
                label_col
            ].nunique() < 1
        ):
            continue

        row = {}

        row["heuristic"] = evaluate(
            test[label_col],
            heuristic_predict(test),
        )

        for name, factory in [
            ("logreg", _make_logreg),
            ("random_forest", _make_rf),
        ]:

            model = factory(
                numeric_features,
                categorical_features,
            )

            model.fit(
                train[feat_cols],
                train[label_col],
            )

            proba = (
                model.predict_proba(
                    test[feat_cols]
                )[:, 1]
            )

            pred = (
                proba >= 0.5
            ).astype(int)

            row[name] = evaluate(
                test[label_col],
                pred,
                proba,
            )

        lopo[
            held_out
        ] = row

    return lopo


def main() -> None:

    sprints = (
        load_reconstructed_sprints()
    )

    labeled = add_labels(
        sprints
    )

    featured = add_features(
        labeled
    )

    full_frame = build_model_frame(
        featured,
        "label_spillover",
    )

    n_total = len(
        full_frame
    )

    n_zero_scope = int(
        (
            full_frame[
                "committed_issue_count"
            ]
            == 0
        ).sum()
    )

    # -----------------------------------------
    # Sanity checks
    # -----------------------------------------
    zero_rows = full_frame[
        full_frame[
            "committed_issue_count"
        ]
        == 0
    ]

    assert (
        zero_rows[
            "committed_story_points"
        ]
        == 0
    ).all(), (
        "committed_story_points is no longer "
        "always 0 when committed_issue_count == 0."
    )

    assert zero_rows[
        "label_spillover"
    ].eq(0).all(), (
        "Not every zero-scope sprint is "
        "spillover-negative."
    )

    rng = np.random.default_rng(
        RNG_SEED
    )

    print(
        f"Total modeled sprints: "
        f"{n_total}; "
        f"zero-committed-scope sprints: "
        f"{n_zero_scope} "
        f"({100 * n_zero_scope / n_total:.1f}%)\n"
    )

    # =====================================================
    # TASK A
    # =====================================================

    task_a_frame = (
        full_frame.copy()
    )

    task_a_frame[
        "has_committed_issue"
    ] = (
        task_a_frame[
            "committed_issue_count"
        ]
        > 0
    ).astype(int)

    train_a, test_a = (
        time_ordered_split(
            task_a_frame
        )
    )

    task_a_time_ordered = (
        run_time_ordered(
            train_a,
            test_a,
            "has_committed_issue",
            TASK_A_FEATURES,
            TASK_A_CATEGORICAL,
            rng,
        )
    )

    task_a_lopo = run_lopo(
        task_a_frame,
        "has_committed_issue",
        TASK_A_FEATURES,
        TASK_A_CATEGORICAL,
    )

    print(
        "\n=== Task A: predicting "
        "has_committed_issue "
        "(n={}, positive rate={:.3f}) ===".format(
            len(task_a_frame),
            task_a_frame[
                "has_committed_issue"
            ].mean(),
        )
    )

    for name, m in (
        task_a_time_ordered.items()
    ):

        print(
            f"  {name:15s} "
            f"f1={m['f1']:.3f}  "
            f"roc_auc={m['roc_auc']:.3f}  "
            f"pr_auc={m['pr_auc']:.3f}  "
            f"training_time="
            f"{m['training_time_seconds']:.4f}s"
        )

    # =====================================================
    # TASK B
    # Conditional spillover:
    # committed_issue_count > 0
    # =====================================================

    nonempty_frame = (
        full_frame[
            full_frame[
                "committed_issue_count"
            ]
            > 0
        ]
        .reset_index(
            drop=True
        )
    )

    train_b, test_b = (
        time_ordered_split(
            nonempty_frame
        )
    )

    task_b_time_ordered = (
        run_time_ordered(
            train_b,
            test_b,
            "label_spillover",
            ALL_FEATURES[:-1],
            CATEGORICAL_FEATURES,
            rng,
        )
    )

    task_b_lopo = run_lopo(
        nonempty_frame,
        "label_spillover",
        ALL_FEATURES[:-1],
        CATEGORICAL_FEATURES,
    )

    print(
        "\n=== Task B: predicting "
        "label_spillover among "
        "committed_issue_count > 0 sprints "
        "(n={}, positive rate={:.3f}) ===".format(
            len(nonempty_frame),
            nonempty_frame[
                "label_spillover"
            ].mean(),
        )
    )

    for name, m in (
        task_b_time_ordered.items()
    ):

        print(
            f"  {name:15s} "
            f"f1={m['f1']:.3f}  "
            f"roc_auc={m['roc_auc']:.3f}  "
            f"pr_auc={m['pr_auc']:.3f}  "
            f"training_time="
            f"{m['training_time_seconds']:.4f}s"
        )

    # =====================================================
    # ORIGINAL FULL-POPULATION SPILLOVER
    # =====================================================

    train_full, test_full = (
        time_ordered_split(
            full_frame
        )
    )

    full_scope_time_ordered = (
        run_time_ordered(
            train_full,
            test_full,
            "label_spillover",
            ALL_FEATURES[:-1],
            CATEGORICAL_FEATURES,
            rng,
        )
    )

    print(
        "\n=== For comparison: original "
        "full-population spillover "
        "(n={}, positive rate={:.3f}) ===".format(
            len(full_frame),
            full_frame[
                "label_spillover"
            ].mean(),
        )
    )

    for name, m in (
        full_scope_time_ordered.items()
    ):

        print(
            f"  {name:15s} "
            f"f1={m['f1']:.3f}  "
            f"roc_auc={m['roc_auc']:.3f}  "
            f"pr_auc={m['pr_auc']:.3f}  "
            f"training_time="
            f"{m['training_time_seconds']:.4f}s"
        )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    payload = {
        "note": (
            "Round-5 review addition. "
            "Task A = predicting has_committed_issue "
            "(full population, n={n_total}); "
            "Task B = predicting label_spillover "
            "restricted to committed_issue_count > 0 "
            "(n={n_nonempty}). "
            "Training time measures model.fit() only. "
            "XGBoost is excluded from this decomposition "
            "script by design."
        ).format(
            n_total=n_total,
            n_nonempty=len(
                nonempty_frame
            ),
        ),

        "n_total_modeled":
            n_total,

        "n_zero_committed_scope":
            n_zero_scope,

        "pct_zero_committed_scope":
            round(
                100
                * n_zero_scope
                / n_total,
                1,
            ),

        "n_nonempty_committed_scope":
            len(
                nonempty_frame
            ),

        "task_a_has_committed_issue": {
            "features":
                TASK_A_FEATURES
                + TASK_A_CATEGORICAL,

            "n":
                len(
                    task_a_frame
                ),

            "positive_rate":
                float(
                    task_a_frame[
                        "has_committed_issue"
                    ].mean()
                ),

            "time_ordered":
                task_a_time_ordered,

            "leave_one_project_out":
                task_a_lopo,
        },

        "task_b_conditional_spillover": {
            "features":
                ALL_FEATURES,

            "n":
                len(
                    nonempty_frame
                ),

            "positive_rate":
                float(
                    nonempty_frame[
                        "label_spillover"
                    ].mean()
                ),

            "time_ordered":
                task_b_time_ordered,

            "leave_one_project_out":
                task_b_lopo,
        },

        "full_population_spillover_for_comparison": {
            "features":
                ALL_FEATURES,

            "n":
                len(
                    full_frame
                ),

            "positive_rate":
                float(
                    full_frame[
                        "label_spillover"
                    ].mean()
                ),

            "time_ordered":
                full_scope_time_ordered,
        },
    }

    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    out_path = (
        RESULTS_DIR
        / "scope_decomposition.json"
    )

    out_path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=float,
        )
    )

    print(
        f"\nWrote results to "
        f"{out_path}"
    )

    # =====================================================
    # TASK B PER-PROJECT LOPO SUMMARY
    # =====================================================

    rows = []

    for held_out, per_model in (
        task_b_lopo.items()
    ):

        for model_name, m in (
            per_model.items()
        ):

            rows.append(
                {
                    "project":
                        held_out,

                    "model":
                        model_name,

                    "n_test":
                        m["n"],

                    "positive_rate":
                        round(
                            m[
                                "positive_rate"
                            ],
                            3,
                        ),

                    "f1":
                        round(
                            m["f1"],
                            3,
                        ),

                    "roc_auc":
                        (
                            round(
                                m[
                                    "roc_auc"
                                ],
                                3,
                            )
                            if m[
                                "roc_auc"
                            ]
                            == m[
                                "roc_auc"
                            ]
                            else None
                        ),

                    "pr_auc":
                        (
                            round(
                                m[
                                    "pr_auc"
                                ],
                                3,
                            )
                            if m[
                                "pr_auc"
                            ]
                            == m[
                                "pr_auc"
                            ]
                            else None
                        ),

                    "brier":
                        (
                            round(
                                m[
                                    "brier"
                                ],
                                3,
                            )
                            if m[
                                "brier"
                            ]
                            == m[
                                "brier"
                            ]
                            else None
                        ),
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        RESULTS_DIR
        / "scope_decomposition_task_b_per_project.csv",
        index=False,
    )

    print(
        "Wrote per-project Task B summary to "
        f"{RESULTS_DIR / 'scope_decomposition_task_b_per_project.csv'}"
    )


if __name__ == "__main__":
    main()
