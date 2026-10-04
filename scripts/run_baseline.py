"""
Orchestrates the Week 10 working baseline: load data -> build labels and
features -> train/evaluate heuristic, logistic regression, and random
forest under both validation schemes from the proposal -> write results.

Usage:
    python -m scripts.run_baseline
"""
from __future__ import annotations

import json
from time import perf_counter
from pathlib import Path

import pandas as pd

from scripts.log_experiment import log_run
from src.baselines import MODEL_FACTORIES, heuristic_predict
from src.data.load_tawos import load_reconstructed_sprints
from src.evaluate import evaluate
from src.features import ALL_FEATURES, add_features, build_model_frame
from src.labels import add_labels

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LABELS = ["label_delay", "label_spillover"]
TIME_SPLIT_TRAIN_FRACTION = 0.7


def time_ordered_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-project time-ordered split, combined across projects."""
    train_parts, test_parts = [], []

    for _, g in frame.groupby("project"):
        g = g.sort_values("start_date")
        cut = max(1, int(len(g) * TIME_SPLIT_TRAIN_FRACTION))
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])

    return pd.concat(train_parts), pd.concat(test_parts)


def run_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str,
) -> dict:
    row = {}

    # Heuristic needs no fitting.
    row["heuristic"] = evaluate(
        test[label_col],
        heuristic_predict(test),
    )
    row["heuristic"]["training_time_seconds"] = 0.0

    for name, factory in MODEL_FACTORIES.items():
        model = factory()

        # Measure model fitting time only.
        start = perf_counter()

        model.fit(
            train[ALL_FEATURES],
            train[label_col],
        )

        training_time = perf_counter() - start

        proba = model.predict_proba(
            test[ALL_FEATURES]
        )[:, 1]

        pred = (proba >= 0.5).astype(int)

        row[name] = evaluate(
            test[label_col],
            pred,
            proba,
        )

        # Store measured training time.
        row[name]["training_time_seconds"] = training_time

        # Print training time for easy checking.
        print(
            f"{label_col} | {name} | "
            f"training time: {training_time:.4f} seconds"
        )

    return row


def main() -> None:
    sprints = load_reconstructed_sprints()
    labeled = add_labels(sprints)
    featured = add_features(labeled)

    print(
        f"Loaded {len(sprints)} sprints across "
        f"{sprints['project'].nunique()} projects."
    )

    print(
        "Label prevalence "
        "(fraction of sprints flagged at-risk):"
    )

    print(
        labeled[
            [
                "label_delay",
                "label_spillover",
                "at_risk",
            ]
        ]
        .mean()
        .round(3)
        .to_string()
    )

    all_results = {}

    for label_col in LABELS:
        frame = build_model_frame(
            featured,
            label_col,
        )

        all_results[label_col] = {}

        # -------------------------------------------------
        # (a) Time-ordered per-project split
        # -------------------------------------------------
        train, test = time_ordered_split(frame)

        all_results[label_col]["time_ordered"] = run_one(
            train,
            test,
            label_col,
        )

        # -------------------------------------------------
        # (b) Leave-one-project-out
        # -------------------------------------------------
        lopo = {}

        for held_out in frame["project"].unique():
            train = frame[
                frame["project"] != held_out
            ]

            test = frame[
                frame["project"] == held_out
            ]

            if (
                len(train) == 0
                or len(test) == 0
                or test[label_col].nunique() < 1
            ):
                continue

            lopo[held_out] = run_one(
                train,
                test,
                label_col,
            )

        all_results[label_col][
            "leave_one_project_out"
        ] = lopo

    # -------------------------------------------------
    # Save complete results
    # -------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)

    out_path = (
        RESULTS_DIR
        / "baseline_results.json"
    )

    out_path.write_text(
        json.dumps(
            all_results,
            indent=2,
            default=float,
        )
    )

    print(
        f"\nWrote results to {out_path}"
    )

    # -------------------------------------------------
    # Experiment tracking
    # -------------------------------------------------
    log_path = log_run(
        all_results,
        dataset="tawos",
        n_projects=sprints[
            "project"
        ].nunique(),
        n_sprints=len(sprints),
        notes="python -m scripts.run_baseline",
    )

    print(
        f"Appended this run to {log_path}"
    )

    # -------------------------------------------------
    # Human-readable time-ordered summary
    # -------------------------------------------------
    print(
        "\n=== Time-ordered split summary ==="
    )

    rows = []

    for label_col in LABELS:
        for model_name, m in all_results[
            label_col
        ]["time_ordered"].items():

            rows.append(
                {
                    "label": label_col,
                    "model": model_name,
                    "n_test": m["n"],
                    "precision": round(
                        m["precision"], 3
                    ),
                    "recall": round(
                        m["recall"], 3
                    ),
                    "f1": round(
                        m["f1"], 3
                    ),
                    "roc_auc": (
                        round(
                            m["roc_auc"], 3
                        )
                        if m["roc_auc"]
                        == m["roc_auc"]
                        else None
                    ),
                    "pr_auc": (
                        round(
                            m["pr_auc"], 3
                        )
                        if m["pr_auc"]
                        == m["pr_auc"]
                        else None
                    ),
                    "training_time_seconds": round(
                        m["training_time_seconds"],
                        4,
                    ),
                }
            )

    summary = pd.DataFrame(rows)

    print(
        summary.to_string(index=False)
    )

    summary.to_csv(
        RESULTS_DIR
        / "time_ordered_summary.csv",
        index=False,
    )

    # -------------------------------------------------
    # Leave-one-project-out summary
    # -------------------------------------------------
    print(
        "\n=== Leave-one-project-out summary "
        "(mean over held-out projects) ==="
    )

    lopo_rows = []

    for label_col in LABELS:
        per_project = all_results[
            label_col
        ]["leave_one_project_out"]

        for model_name in (
            ["heuristic"]
            + list(MODEL_FACTORIES.keys())
        ):
            f1s = [
                m[model_name]["f1"]
                for m in per_project.values()
            ]

            recalls = [
                m[model_name]["recall"]
                for m in per_project.values()
            ]

            precisions = [
                m[model_name]["precision"]
                for m in per_project.values()
            ]

            training_times = [
                m[model_name][
                    "training_time_seconds"
                ]
                for m in per_project.values()
            ]

            lopo_rows.append(
                {
                    "label": label_col,
                    "model": model_name,
                    "n_held_out_projects": len(
                        per_project
                    ),
                    "mean_precision": (
                        round(
                            sum(precisions)
                            / len(precisions),
                            3,
                        )
                        if precisions
                        else None
                    ),
                    "mean_recall": (
                        round(
                            sum(recalls)
                            / len(recalls),
                            3,
                        )
                        if recalls
                        else None
                    ),
                    "mean_f1": (
                        round(
                            sum(f1s)
                            / len(f1s),
                            3,
                        )
                        if f1s
                        else None
                    ),
                    "mean_training_time_seconds": (
                        round(
                            sum(training_times)
                            / len(training_times),
                            4,
                        )
                        if training_times
                        else None
                    ),
                }
            )

    lopo_summary = pd.DataFrame(
        lopo_rows
    )

    print(
        lopo_summary.to_string(index=False)
    )

    lopo_summary.to_csv(
        RESULTS_DIR
        / "leave_one_project_out_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
