import pandas as pd
import numpy as np

from src.features import add_features, build_model_frame


def _row(i, committed, completed):
    return {
        "project": "p",
        "sprint_id": f"p__{i}",
        "start_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=14*i),
        "committed_issue_count": committed,
        "committed_story_points": float(committed),
        "num_developers": 0 if committed == 0 else 1,
        "sprint_length_days": 14,
        "completed_issue_count": completed,
        "not_completed_count": committed-completed,
        "delivered_story_points": float(completed),
        "completion_ratio": np.nan if committed == 0 else completed/committed,
        "label_delay": 0,
        "label_spillover": int(committed-completed > 0),
    }


def test_zero_scope_previous_sprint_does_not_delete_next_sprint():
    # Old code dropped p__2 because p__1 had completion_ratio NaN.
    df = pd.DataFrame([_row(0, 10, 10), _row(1, 0, 0), _row(2, 8, 7)])
    featured = add_features(df)
    frame = build_model_frame(featured, "label_delay")
    assert list(frame["sprint_id"]) == ["p__1", "p__2"]
    assert frame.loc[frame["sprint_id"] == "p__2", "prior_sprint_completion_ratio"].iloc[0] == 1.0
