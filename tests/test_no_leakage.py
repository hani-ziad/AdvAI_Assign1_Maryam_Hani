"""
Guards the core scientific integrity requirement from the Week 8 proposal:
features must be computable at (or before) sprint start, so none of them
may be derived from the current sprint's own outcome.

This test doesn't just check today's feature list -- it fails loudly if
someone adds an outcome column to ALL_FEATURES in the future.
"""
from src.features import ALL_FEATURES

OUTCOME_COLUMNS = {
    "completed_issue_count",
    "not_completed_count",
    "punted_count",
    "delivered_story_points",
    "completion_ratio",
    "complete_date",
    "label_delay",
    "label_spillover",
    "at_risk",
}


def test_no_outcome_columns_in_feature_list():
    leaked = OUTCOME_COLUMNS.intersection(ALL_FEATURES)
    assert not leaked, f"Leakage: outcome columns present in feature list: {leaked}"


def test_historical_features_are_named_as_prior():
    # A light naming convention check: every "historical" feature should be
    # explicit about referring to prior sprints, so the distinction remains explicit and a
    # same-sprint feature by name alone.
    from src.features import HISTORICAL_FEATURES

    for name in HISTORICAL_FEATURES:
        assert "prior" in name, f"{name} does not look like a prior-sprint feature"
