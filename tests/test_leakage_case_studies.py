"""
Smoke tests for the leakage case studies: not re-verifying the ML result
(that's the point of results/leakage_case_studies.json, not a unit test),
just guarding that both demos run end-to-end and return well-formed
metrics for both labels, so a future refactor can't silently break them.
"""
from scripts.run_leakage_case_studies import build_frame_with_ratio, time_ordered_indices
from src.leakage_case_studies import case_study_a_post_outcome_feature, case_study_b_oversample_order

EXPECTED_METRIC_KEYS = {"precision", "recall", "f1", "roc_auc", "pr_auc"}


def test_case_study_a_runs_for_both_labels():
    for label_col in ["label_delay", "label_spillover"]:
        frame = build_frame_with_ratio(label_col)
        train_idx, test_idx = time_ordered_indices(frame)
        results = case_study_a_post_outcome_feature(frame, label_col, train_idx, test_idx)
        assert len(results) == 2
        for metrics in results.values():
            assert EXPECTED_METRIC_KEYS.issubset(metrics.keys())
            assert 0.0 <= metrics["precision"] <= 1.0
            assert 0.0 <= metrics["recall"] <= 1.0


def test_case_study_b_runs_for_both_labels():
    for label_col in ["label_delay", "label_spillover"]:
        frame = build_frame_with_ratio(label_col)
        train_idx, test_idx = time_ordered_indices(frame)
        results = case_study_b_oversample_order(frame, label_col, train_idx, test_idx)
        assert len(results) == 2
        for metrics in results.values():
            assert EXPECTED_METRIC_KEYS.issubset(metrics.keys())


def test_completion_ratio_is_reattached_for_case_study_a():
    # build_model_frame() deliberately drops completion_ratio (it's an
    # outcome column, excluded from ALL_FEATURES) -- case study A needs it
    # back specifically to demonstrate what happens if it leaks in, so
    # confirm the merge in build_frame_with_ratio actually restores it.
    frame = build_frame_with_ratio("label_spillover")
    assert "completion_ratio" in frame.columns
    assert frame["completion_ratio"].notna().all()
