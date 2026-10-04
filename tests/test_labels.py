import pandas as pd

from src.labels import add_labels


def _sprint_row(**overrides):
    row = {
        "project": "p",
        "sprint_id": "p__1",
        "start_date": pd.Timestamp("2020-01-01"),
        "end_date": pd.Timestamp("2020-01-14"),
        "complete_date": pd.Timestamp("2020-01-14"),
        "committed_issue_count": 10,
        "completed_issue_count": 10,
        "not_completed_count": 0,
        "delivered_story_points": 20,
    }
    row.update(overrides)
    return row


def test_on_time_full_completion_is_not_at_risk():
    df = pd.DataFrame([_sprint_row()])
    labeled = add_labels(df)
    assert labeled.loc[0, "label_delay"] == 0
    assert labeled.loc[0, "label_spillover"] == 0
    assert labeled.loc[0, "at_risk"] == 0


def test_late_completion_triggers_delay_label():
    df = pd.DataFrame([_sprint_row(complete_date=pd.Timestamp("2020-01-20"))])
    labeled = add_labels(df)
    assert labeled.loc[0, "label_delay"] == 1
    assert labeled.loc[0, "at_risk"] == 1


def test_one_day_late_is_within_threshold():
    # DELAY_THRESHOLD_DAYS is 1, so exactly 1 day late is NOT flagged.
    df = pd.DataFrame([_sprint_row(complete_date=pd.Timestamp("2020-01-15"))])
    labeled = add_labels(df)
    assert labeled.loc[0, "label_delay"] == 0


def test_carryover_triggers_spillover_label():
    df = pd.DataFrame(
        [_sprint_row(completed_issue_count=8, not_completed_count=2)]
    )
    labeled = add_labels(df)
    assert labeled.loc[0, "label_spillover"] == 1
    assert labeled.loc[0, "at_risk"] == 1


def test_spillover_uses_derived_incomplete_issue_count():
    # The current binding definition is "at least one incomplete committed
    # issue". In production not_completed_count is derived from committed -
    # completed, so an inconsistent hand-built row should follow that explicit
    # target column rather than revive the obsolete 0.8 ratio rule.
    df = pd.DataFrame([_sprint_row(completed_issue_count=7, not_completed_count=0)])
    labeled = add_labels(df)
    assert labeled.loc[0, "label_spillover"] == 0


def test_zero_committed_issues_does_not_crash():
    df = pd.DataFrame([_sprint_row(committed_issue_count=0, completed_issue_count=0)])
    labeled = add_labels(df)
    # completion_ratio is undefined (0/0); should not raise and should not
    # spuriously flag spillover via the ratio branch.
    assert labeled.loc[0, "completion_ratio"] != labeled.loc[0, "completion_ratio"]  # NaN
