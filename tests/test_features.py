import pandas as pd

from src.features import add_features, build_model_frame


def _sprints_for_one_project(n=4):
    rows = []
    for i in range(n):
        rows.append(
            {
                "project": "p",
                "sprint_id": f"p__{i}",
                "start_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=14 * i),
                "end_date": pd.Timestamp("2020-01-14") + pd.Timedelta(days=14 * i),
                "complete_date": pd.Timestamp("2020-01-14") + pd.Timedelta(days=14 * i),
                "committed_issue_count": 10,
                "committed_story_points": 20,
                "num_developers": 5,
                "sprint_length_days": 14,
                "completed_issue_count": 8 + i % 2,
                "not_completed_count": 2 - i % 2,
                "delivered_story_points": 15 + i,
                "completion_ratio": (8 + i % 2) / 10,
                "label_delay": 0,
                "label_spillover": 0,
                "at_risk": 0,
            }
        )
    return pd.DataFrame(rows)


def test_first_sprint_has_no_history_and_is_dropped():
    df = _sprints_for_one_project(4)
    featured = add_features(df)
    frame = build_model_frame(featured, "label_delay")
    # 4 sprints in, 1 (the first) has no prior history -> 3 remain.
    assert len(frame) == 3
    assert "p__0" not in frame["sprint_id"].values


def test_rolling_features_use_only_strictly_prior_sprints():
    df = _sprints_for_one_project(4)
    featured = add_features(df)
    row2 = featured[featured["sprint_id"] == "p__2"].iloc[0]
    # prior_avg_delivered_sp_3 for sprint index 2 should average sprints 0,1
    # (delivered_story_points 15, 16), NOT include sprint 2's own 17.
    expected = pd.Series([15, 16]).mean()
    assert row2["prior_avg_delivered_sp_3"] == expected


def test_projects_do_not_leak_history_into_each_other():
    df_a = _sprints_for_one_project(3)
    df_b = _sprints_for_one_project(3)
    df_b["project"] = "q"
    df_b["sprint_id"] = df_b["sprint_id"].str.replace("p__", "q__")
    combined = pd.concat([df_a, df_b], ignore_index=True)

    featured = add_features(combined)
    first_of_q = featured[featured["sprint_id"] == "q__0"].iloc[0]
    assert pd.isna(first_of_q["prior_sprint_completion_ratio"])
