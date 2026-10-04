"""Additional analyses requested during review.

Run from the project root with:
    python -m scripts.run_validation_sensitivity_analyses

Outputs are written to results/.
"""
from __future__ import annotations

import json
import platform
import re
from time import perf_counter
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.run_baseline import time_ordered_split
from src.data.load_tawos import (
    load_sprints,
    load_issues,
    load_change_log,
    load_reconstructed_sprints,
)
from src.data.planning_snapshot import (
    _field_kind,
    _project_maps,
    _parse_change_sprint_set,
    _story_after,
    _story_before,
)
from src.features import (
    ALL_FEATURES,
    PLANNING_TIME_FEATURES,
    add_features,
    build_model_frame,
)
from src.labels import add_labels


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "tawos"
RESULTS_DIR = ROOT / "results"
RNG_SEED = 42


def metric_row(y, pred, proba=None):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)

    row = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }

    if proba is not None and len(np.unique(y)) > 1:
        row.update(
            {
                "roc_auc": float(roc_auc_score(y, proba)),
                "pr_auc": float(average_precision_score(y, proba)),
                "brier": float(brier_score_loss(y, proba)),
            }
        )
    else:
        row.update(
            {
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "brier": np.nan,
            }
        )

    return row


def preprocessor(features):
    num = [f for f in features if f != "project"]
    cats = [f for f in features if f == "project"]

    blocks = [
        ("num", StandardScaler(), num)
    ]

    if cats:
        blocks.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cats,
            )
        )

    return ColumnTransformer(blocks)


def make_model(name, features):
    if name in {
        "scope_logreg",
        "count_only_logreg",
        "logreg",
    }:
        return Pipeline(
            [
                ("prep", preprocessor(features)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    if name == "random_forest":
        return Pipeline(
            [
                ("prep", preprocessor(features)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RNG_SEED,
                    ),
                ),
            ]
        )

    if name == "xgboost":
        from xgboost import XGBClassifier

        return Pipeline(
            [
                ("prep", preprocessor(features)),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        eval_metric="logloss",
                        random_state=RNG_SEED,
                    ),
                ),
            ]
        )

    raise ValueError(name)


def fit_eval(train, test, label, features, model_name):
    model = make_model(model_name, features)

    # Measure model fitting/training time only.
    start = perf_counter()

    model.fit(
        train[features],
        train[label],
    )

    training_time = perf_counter() - start

    proba = model.predict_proba(
        test[features]
    )[:, 1]

    pred = (proba >= 0.5).astype(int)

    metrics = metric_row(
        test[label],
        pred,
        proba,
    )

    metrics["training_time_seconds"] = float(training_time)

    print(
        f"{label} | {model_name} | "
        f"training time: {training_time:.4f} seconds"
    )

    return model, proba, pred, metrics


def load_frames():
    sprints = load_reconstructed_sprints()
    labeled = add_labels(sprints)
    featured = add_features(labeled)

    return {
        label: build_model_frame(featured, label)
        for label in [
            "label_delay",
            "label_spillover",
        ]
    }


def write(name, df):
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(name, df.shape)


def scope_and_baselines(frames):
    rows = []

    for label, frame in frames.items():
        variants = [("full", frame)]

        if label == "label_spillover":
            variants.append(
                (
                    "nonempty",
                    frame[
                        frame.committed_issue_count > 0
                    ].copy(),
                )
            )

        for pop, vf in variants:
            train, test = time_ordered_split(vf)

            y = test[label].astype(int).to_numpy()

            for name, pred in [
                ("always_negative", np.zeros_like(y)),
                ("always_positive", np.ones_like(y)),
            ]:
                baseline_metrics = metric_row(
                    y,
                    pred,
                    pred,
                )

                baseline_metrics[
                    "training_time_seconds"
                ] = 0.0

                rows.append(
                    {
                        "label": label,
                        "population": pop,
                        "model": name,
                        **baseline_metrics,
                    }
                )

            models = [
                (
                    "count_only_logreg",
                    ["committed_issue_count"],
                ),
                (
                    "scope_logreg",
                    PLANNING_TIME_FEATURES,
                ),
                (
                    "random_forest",
                    ALL_FEATURES,
                ),
            ]

            for model_name, feats in models:
                _, _, _, m = fit_eval(
                    train,
                    test,
                    label,
                    feats,
                    model_name,
                )

                rows.append(
                    {
                        "label": label,
                        "population": pop,
                        "model": model_name,
                        **m,
                    }
                )

    return pd.DataFrame(rows)


def lopo_denominators_from_existing(frames):
    with open(
        RESULTS_DIR / "baseline_results.json",
        "r",
        encoding="utf-8",
    ) as f:
        base = json.load(f)

    rows = []

    for label, frame in frames.items():
        for project, res in base[
            label
        ]["leave_one_project_out"].items():

            test = frame[
                frame.project == project
            ]

            for model in [
                "heuristic",
                "logreg",
                "random_forest",
                "xgboost",
            ]:
                m = res[model]

                rows.append(
                    {
                        "label": label,
                        "population": "full",
                        "project": project,
                        "model": model,
                        "n": len(test),
                        "positives": int(
                            test[label].sum()
                        ),
                        "positive_rate": float(
                            test[label].mean()
                        ),
                        **{
                            k: m.get(k, np.nan)
                            for k in [
                                "precision",
                                "recall",
                                "f1",
                                "roc_auc",
                                "pr_auc",
                                "brier",
                            ]
                        },
                    }
                )

    # Conditional nonempty spillover with fewer refits.
    frame = frames["label_spillover"]

    nonempty = frame[
        frame.committed_issue_count > 0
    ].copy()

    for project in sorted(
        nonempty.project.unique()
    ):
        train = nonempty[
            nonempty.project != project
        ]

        test = nonempty[
            nonempty.project == project
        ]

        for model_name, feats in [
            (
                "scope_logreg",
                PLANNING_TIME_FEATURES,
            ),
            (
                "random_forest",
                ALL_FEATURES,
            ),
        ]:
            _, _, _, m = fit_eval(
                train,
                test,
                "label_spillover",
                feats,
                model_name,
            )

            rows.append(
                {
                    "label": "label_spillover",
                    "population": "nonempty",
                    "project": project,
                    "model": model_name,
                    **m,
                }
            )

    return pd.DataFrame(rows)


def cluster_uncertainty(frames, n_boot=500):
    tasks = [
        (
            "label_delay",
            "full",
            frames["label_delay"],
            "random_forest",
            ALL_FEATURES,
        ),
        (
            "label_spillover",
            "nonempty",
            frames["label_spillover"][
                frames["label_spillover"].committed_issue_count > 0
            ].copy(),
            "random_forest",
            ALL_FEATURES,
        ),
        (
            "label_spillover",
            "nonempty",
            frames["label_spillover"][
                frames["label_spillover"].committed_issue_count > 0
            ].copy(),
            "scope_logreg",
            PLANNING_TIME_FEATURES,
        ),
    ]

    rng = np.random.default_rng(
        RNG_SEED
    )

    out = []

    for (
        label,
        pop,
        frame,
        model_name,
        feats,
    ) in tasks:

        train, test = time_ordered_split(
            frame
        )

        _, proba, pred, m = fit_eval(
            train,
            test,
            label,
            feats,
            model_name,
        )

        tmp = test[
            ["project", label]
        ].copy()

        tmp["proba"] = proba
        tmp["pred"] = pred

        by_proj = {
            p: g
            for p, g in tmp.groupby("project")
        }

        vals = defaultdict(list)
        projects = np.array(
            list(by_proj)
        )

        for _ in range(n_boot):
            smp = pd.concat(
                [
                    by_proj[p]
                    for p in rng.choice(
                        projects,
                        len(projects),
                        replace=True,
                    )
                ],
                ignore_index=True,
            )

            y = smp[label].to_numpy()

            if len(np.unique(y)) < 2:
                continue

            vals["f1"].append(
                f1_score(
                    y,
                    smp.pred,
                    zero_division=0,
                )
            )

            vals["roc_auc"].append(
                roc_auc_score(
                    y,
                    smp.proba,
                )
            )

            vals["pr_auc"].append(
                average_precision_score(
                    y,
                    smp.proba,
                )
            )

            vals["brier"].append(
                brier_score_loss(
                    y,
                    smp.proba,
                )
            )

        row = {
            "label": label,
            "population": pop,
            "model": model_name,
            **m,
        }

        for k, v in vals.items():
            row[
                f"{k}_cluster_ci_low"
            ] = float(
                np.quantile(v, 0.025)
            )

            row[
                f"{k}_cluster_ci_high"
            ] = float(
                np.quantile(v, 0.975)
            )

        out.append(row)

    return pd.DataFrame(out)


def rolling_splits(frames):
    rows = []

    for label, frame in frames.items():
        variants = [
            ("full", frame)
        ] + (
            [
                (
                    "nonempty",
                    frame[
                        frame.committed_issue_count > 0
                    ].copy(),
                )
            ]
            if label == "label_spillover"
            else []
        )

        for pop, vf in variants:
            for frac in [
                0.5,
                0.6,
                0.7,
                0.8,
            ]:
                tr = []
                te = []

                for _, g in vf.groupby(
                    "project"
                ):
                    g = g.sort_values(
                        "start_date"
                    )

                    cut = int(
                        len(g) * frac
                    )

                    if 0 < cut < len(g):
                        tr.append(
                            g.iloc[:cut]
                        )

                        te.append(
                            g.iloc[cut:]
                        )

                train = pd.concat(tr)
                test = pd.concat(te)

                for model_name, feats in [
                    (
                        "scope_logreg",
                        PLANNING_TIME_FEATURES,
                    ),
                    (
                        "random_forest",
                        ALL_FEATURES,
                    ),
                ]:
                    _, _, _, m = fit_eval(
                        train,
                        test,
                        label,
                        feats,
                        model_name,
                    )

                    rows.append(
                        {
                            "label": label,
                            "population": pop,
                            "train_fraction": frac,
                            "model": model_name,
                            **m,
                        }
                    )

    return pd.DataFrame(rows)


def project_flow(frames):
    raw = load_sprints(
        RAW_DIR,
        modeling_only=False,
    )

    eligible = load_sprints(
        RAW_DIR,
        modeling_only=True,
    )

    frame = frames["label_delay"]

    rows = []

    for p in sorted(
        raw.project.unique()
    ):
        r = raw[
            raw.project == p
        ]

        e = eligible[
            eligible.project == p
        ]

        m = frame[
            frame.project == p
        ]

        rows.append(
            {
                "project": p,
                "raw_sprints": len(r),
                "closed": int(
                    (
                        r.sprint_state
                        .astype(str)
                        .str.upper()
                        == "CLOSED"
                    ).sum()
                ),
                "future": int(
                    (
                        r.sprint_state
                        .astype(str)
                        .str.upper()
                        == "FUTURE"
                    ).sum()
                ),
                "active": int(
                    (
                        r.sprint_state
                        .astype(str)
                        .str.upper()
                        == "ACTIVE"
                    ).sum()
                ),
                "invalid_closed_timestamps": int(
                    (
                        r.sprint_state
                        .astype(str)
                        .str.upper()
                        == "CLOSED"
                    ).sum()
                    - len(e)
                ),
                "eligible_closed": len(e),
                "modeled": len(m),
                "zero_scope": int(
                    (
                        m.committed_issue_count
                        == 0
                    ).sum()
                ),
                "nonempty": int(
                    (
                        m.committed_issue_count
                        > 0
                    ).sum()
                ),
                "late_positive": int(
                    m.label_delay.sum()
                ),
                "spillover_positive": int(
                    frames[
                        "label_spillover"
                    ][
                        frames[
                            "label_spillover"
                        ].project
                        == p
                    ].label_spillover.sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def coverage_by_project(frames):
    issues = load_issues(
        RAW_DIR
    )

    change = load_change_log(
        RAW_DIR,
        required=True,
    )

    sprint_ids = set(
        change[
            change.field
            .astype(str)
            .str.lower()
            .str.contains(
                "sprint",
                na=False,
            )
        ].raw_issue_id
    )

    story_ids = set(
        change[
            change.field
            .astype(str)
            .str.lower()
            .str.replace(
                " ",
                "",
            )
            .str.contains(
                "storypoints",
                na=False,
            )
        ].raw_issue_id
    )

    issues = issues.assign(
        has_sprint_history=issues.raw_issue_id.isin(
            sprint_ids
        ),
        has_story_point_history=issues.raw_issue_id.isin(
            story_ids
        ),
    )

    cov = (
        issues.groupby("project")
        .agg(
            issues=(
                "raw_issue_id",
                "count",
            ),
            issues_with_sprint_history=(
                "has_sprint_history",
                "sum",
            ),
            issues_with_story_point_history=(
                "has_story_point_history",
                "sum",
            ),
        )
        .reset_index()
    )

    cov[
        "sprint_history_rate"
    ] = (
        cov.issues_with_sprint_history
        / cov.issues
    )

    cov[
        "story_point_history_rate"
    ] = (
        cov.issues_with_story_point_history
        / cov.issues
    )

    sprint = (
        frames[
            "label_spillover"
        ]
        .groupby("project")
        .agg(
            modeled_sprints=(
                "sprint_id",
                "count",
            ),
            zero_scope_sprints=(
                "committed_issue_count",
                lambda s: int(
                    (s == 0).sum()
                ),
            ),
            nonempty_sprints=(
                "committed_issue_count",
                lambda s: int(
                    (s > 0).sum()
                ),
            ),
            spillover_positives=(
                "label_spillover",
                "sum",
            ),
        )
        .reset_index()
    )

    return (
        cov.merge(
            sprint,
            on="project",
        ),
        issues,
    )


def missing_history_sensitivity(
    frames,
    issues_cov,
):
    rates = (
        issues_cov.groupby(
            "project"
        ).has_sprint_history.mean()
    )

    rows = []

    for cutoff in [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
    ]:
        keep = list(
            rates[
                rates >= cutoff
            ].index
        )

        frame = frames[
            "label_spillover"
        ]

        subset = frame[
            (
                frame.project.isin(
                    keep
                )
            )
            & (
                frame.committed_issue_count
                > 0
            )
        ].copy()

        if (
            subset.project.nunique() < 2
            or len(subset) < 50
        ):
            continue

        train, test = (
            time_ordered_split(
                subset
            )
        )

        for model_name, feats in [
            (
                "scope_logreg",
                PLANNING_TIME_FEATURES,
            ),
            (
                "random_forest",
                ALL_FEATURES,
            ),
        ]:
            _, _, _, m = fit_eval(
                train,
                test,
                "label_spillover",
                feats,
                model_name,
            )

            rows.append(
                {
                    "cutoff_project_sprint_history_rate":
                        cutoff,
                    "projects_kept":
                        len(keep),
                    "model":
                        model_name,
                    **m,
                }
            )

    return pd.DataFrame(rows)


def invalid_status_sensitivity(frames):
    issues = load_issues(
        RAW_DIR
    )

    bad = (
        issues.status
        .astype(str)
        .str.lower()
        .str.contains(
            "invalid|won't fix|not being considered",
            regex=True,
            na=False,
        )
    )

    rows = [
        {
            "removed_status_rule":
                "Invalid, Won't Fix, Not Being Considered",
            "matching_issues":
                int(bad.sum()),
            "share_of_all_issues":
                float(bad.mean()),
        }
    ]

    return pd.DataFrame(rows)


def perm_importance(frames):
    rows = []

    tasks = [
        (
            "label_delay",
            "full",
            frames[
                "label_delay"
            ],
        ),
        (
            "label_spillover",
            "nonempty",
            frames[
                "label_spillover"
            ][
                frames[
                    "label_spillover"
                ].committed_issue_count
                > 0
            ].copy(),
        ),
    ]

    for label, pop, frame in tasks:
        train, test = (
            time_ordered_split(
                frame
            )
        )

        model, _, _, _ = fit_eval(
            train,
            test,
            label,
            ALL_FEATURES,
            "random_forest",
        )

        pi = permutation_importance(
            model,
            test[ALL_FEATURES],
            test[label],
            scoring="roc_auc",
            n_repeats=10,
            random_state=RNG_SEED,
        )

        for f, mean, std in zip(
            ALL_FEATURES,
            pi.importances_mean,
            pi.importances_std,
        ):
            rows.append(
                {
                    "label": label,
                    "population": pop,
                    "model": "random_forest",
                    "feature": f,
                    "auc_drop_mean":
                        float(mean),
                    "auc_drop_std":
                        float(std),
                }
            )

    return pd.DataFrame(rows)


def calibration_bins(frames):
    rows = []

    tasks = [
        (
            "label_delay",
            "full",
            frames[
                "label_delay"
            ],
        ),
        (
            "label_spillover",
            "nonempty",
            frames[
                "label_spillover"
            ][
                frames[
                    "label_spillover"
                ].committed_issue_count
                > 0
            ].copy(),
        ),
    ]

    for label, pop, frame in tasks:
        train, test = (
            time_ordered_split(
                frame
            )
        )

        for model_name in [
            "random_forest"
        ]:
            _, proba, _, _ = fit_eval(
                train,
                test,
                label,
                ALL_FEATURES,
                model_name,
            )

            tmp = pd.DataFrame(
                {
                    "y":
                        test[
                            label
                        ].to_numpy(),
                    "proba":
                        proba,
                }
            )

            tmp["bin"] = pd.cut(
                tmp.proba,
                bins=np.linspace(
                    0,
                    1,
                    11,
                ),
                include_lowest=True,
            )

            for b, g in tmp.groupby(
                "bin",
                observed=False,
            ):
                rows.append(
                    {
                        "label":
                            label,
                        "population":
                            pop,
                        "model":
                            model_name,
                        "bin":
                            str(b),
                        "n":
                            len(g),
                        "mean_predicted":
                            (
                                float(
                                    g.proba.mean()
                                )
                                if len(g)
                                else np.nan
                            ),
                        "observed_rate":
                            (
                                float(
                                    g.y.mean()
                                )
                                if len(g)
                                else np.nan
                            ),
                    }
                )

    return pd.DataFrame(rows)


def environment():
    import pandas
    import sklearn
    import numpy

    try:
        import xgboost
        xgb = xgboost.__version__
    except Exception as e:
        xgb = str(e)

    return {
        "python":
            platform.python_version(),
        "platform":
            platform.platform(),
        "numpy":
            numpy.__version__,
        "pandas":
            pandas.__version__,
        "scikit_learn":
            sklearn.__version__,
        "xgboost":
            xgb,
    }


def main():
    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    frames = load_frames()

    analyses = []

    print("stage scope")
    analyses.append(
        (
            "scope_only_baselines",
            scope_and_baselines(
                frames
            ),
        )
    )

    print("stage lopo")
    analyses.append(
        (
            "lopo_with_denominators",
            lopo_denominators_from_existing(
                frames
            ),
        )
    )

    print("stage uncertainty")
    analyses.append(
        (
            "cluster_uncertainty",
            cluster_uncertainty(
                frames,
                n_boot=2000,
            ),
        )
    )

    print("stage rolling")
    analyses.append(
        (
            "rolling_temporal_splits",
            rolling_splits(
                frames
            ),
        )
    )

    print("stage coverage")
    cov, issue_cov = (
        coverage_by_project(
            frames
        )
    )

    analyses.append(
        (
            "reconstruction_coverage_by_project",
            cov,
        )
    )

    analyses.append(
        (
            "missing_history_sensitivity",
            missing_history_sensitivity(
                frames,
                issue_cov,
            ),
        )
    )

    analyses.append(
        (
            "invalid_status_sensitivity",
            invalid_status_sensitivity(
                frames
            ),
        )
    )

    analyses.append(
        (
            "project_flow_table",
            project_flow(
                frames
            ),
        )
    )

    print("stage perm")
    analyses.append(
        (
            "permutation_importance",
            perm_importance(
                frames
            ),
        )
    )

    print("stage calibration")
    analyses.append(
        (
            "calibration_bins",
            calibration_bins(
                frames
            ),
        )
    )

    compact = {}

    for name, df in analyses:
        write(
            name,
            df,
        )

        compact[name] = (
            df.head(30)
            .to_dict(
                orient="records"
            )
        )

    env = environment()
    compact["environment"] = env

    (
        RESULTS_DIR
        / "environment_versions.json"
    ).write_text(
        json.dumps(
            env,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        ROOT
        / "requirements-lock.txt"
    ).write_text(
        "\n".join(
            [
                f"numpy=={env['numpy']}",
                f"pandas=={env['pandas']}",
                f"scikit-learn=={env['scikit_learn']}",
                f"xgboost=={env['xgboost']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (
        RESULTS_DIR
        / "validation_sensitivity_results.json"
    ).write_text(
        json.dumps(
            compact,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("done")


if __name__ == "__main__":
    main()
