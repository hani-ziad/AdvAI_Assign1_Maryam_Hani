import pandas as pd

from src.data.planning_snapshot import derive_planning_time_aggregates


def test_issue_added_after_sprint_start_is_not_committed():
    sprints = pd.DataFrame([
        {
            "raw_sprint_id": 1,
            "jira_sprint_id": 101,
            "sprint_id": "P__1",
            "sprint_name": "S1",
            "project": "P",
            "start_date": pd.Timestamp("2020-01-10"),
            "end_date": pd.Timestamp("2020-01-20"),
            "complete_date": pd.Timestamp("2020-01-20"),
        }
    ])
    issues = pd.DataFrame([
        {
            "raw_issue_id": 7,
            "issue_key": "P-7",
            "project": "P",
            "raw_sprint_id": 1,  # final state says it is in S1
            "creation_date": pd.Timestamp("2020-01-01"),
            "resolution_date": pd.Timestamp("2020-01-19"),
            "story_point": 5.0,
        }
    ])
    # Issue was added to sprint after it started. Reversing this event at
    # 2020-01-10 must yield empty membership.
    changes = pd.DataFrame([
        {
            "raw_issue_id": 7,
            "field": "Sprint",
            "from_value": None,
            "to_value": "101",
            "from_string": None,
            "to_string": "S1",
            "change_date": pd.Timestamp("2020-01-12"),
        }
    ])
    agg, audit = derive_planning_time_aggregates(sprints, issues, changes)
    assert agg.loc[0, "committed_issue_count"] == 0


def test_issue_removed_after_start_remains_in_committed_scope():
    sprints = pd.DataFrame([
        {
            "raw_sprint_id": 1,
            "jira_sprint_id": 101,
            "sprint_id": "P__1",
            "sprint_name": "S1",
            "project": "P",
            "start_date": pd.Timestamp("2020-01-10"),
            "end_date": pd.Timestamp("2020-01-20"),
            "complete_date": pd.Timestamp("2020-01-20"),
        }
    ])
    issues = pd.DataFrame([
        {
            "raw_issue_id": 8,
            "issue_key": "P-8",
            "project": "P",
            "raw_sprint_id": None,  # final state: removed from sprint
            "creation_date": pd.Timestamp("2020-01-01"),
            "resolution_date": pd.NaT,
            "story_point": 3.0,
        }
    ])
    changes = pd.DataFrame([
        {
            "raw_issue_id": 8,
            "field": "Sprint",
            "from_value": "101",
            "to_value": None,
            "from_string": "S1",
            "to_string": None,
            "change_date": pd.Timestamp("2020-01-15"),
        }
    ])
    agg, audit = derive_planning_time_aggregates(sprints, issues, changes)
    assert agg.loc[0, "committed_issue_count"] == 1
    assert agg.loc[0, "not_completed_count"] == 1
