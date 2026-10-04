"""
Runs both leakage case studies (src/leakage_case_studies.py) on this
project's own data and prints/saves a before/after comparison -- the
concrete evidence behind the "contribution" framing: published high
accuracy in this literature may partly reflect avoidable methodological
mistakes rather than the task being genuinely easy.

Usage:
    python -m scripts.run_leakage_case_studies
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.load_tawos import load_reconstructed_sprints
from src.features import add_features, build_model_frame
from src.labels import add_labels
from src.leakage_case_studies import case_study_a_post_outcome_feature, case_study_b_oversample_order

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def build_frame_with_ratio(label_col: str) -> pd.DataFrame:
    sprints = load_reconstructed_sprints()
    labeled = add_labels(sprints)
    featured = add_features(labeled)
    frame = build_model_frame(featured, label_col)
    # build_model_frame drops completion_ratio; case study A needs it back.
    frame = frame.merge(
        featured[["sprint_id", "completion_ratio"]], on="sprint_id", how="left"
    )
    # Case study A adds completion_ratio as a feature. Sprints with zero
    # committed issues have an undefined ratio (NaN in labels.py by design),
    # which no classifier can use as a feature -- exclude those rows from
    # this comparison only (the leakage-safe feature set never needed them).
    frame = frame.dropna(subset=["completion_ratio"]).reset_index(drop=True)
    return frame


def time_ordered_indices(frame: pd.DataFrame, train_fraction: float = 0.7):
    train_idx, test_idx = [], []
    for _, g in frame.groupby("project"):
        g = g.sort_values("start_date")
        cut = max(1, int(len(g) * train_fraction))
        train_idx += list(g.index[:cut])
        test_idx += list(g.index[cut:])
    return train_idx, test_idx


def fmt(metrics: dict) -> dict:
    return {
        k: (round(v, 3) if isinstance(v, float) else v)
        for k, v in metrics.items()
        if k in ("precision", "recall", "f1", "roc_auc", "pr_auc")
    }


def main() -> None:
    all_results = {}

    for label_col in ["label_delay", "label_spillover"]:
        frame = build_frame_with_ratio(label_col)
        train_idx, test_idx = time_ordered_indices(frame)

        print(f"\n=== Case study A ({label_col}): post-outcome feature ===")
        a = case_study_a_post_outcome_feature(frame, label_col, train_idx, test_idx)
        for name, metrics in a.items():
            print(f"  {name}: {fmt(metrics)}")

        print(f"\n=== Case study B ({label_col}): oversample order ===")
        b = case_study_b_oversample_order(frame, label_col, train_idx, test_idx)
        for name, metrics in b.items():
            print(f"  {name}: {fmt(metrics)}")

        all_results[label_col] = {
            "case_study_a_post_outcome_feature": {k: fmt(v) for k, v in a.items()},
            "case_study_b_oversample_order": {k: fmt(v) for k, v in b.items()},
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "leakage_case_studies.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
