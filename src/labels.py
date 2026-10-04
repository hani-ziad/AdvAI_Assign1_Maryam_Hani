"""Label construction for sprint-risk prediction.

Delay is operationalized as recorded sprint closure >1 day after planned end.
Spillover is operationalized as at least one committed issue not completed by
planned sprint end, matching the implementation used for the main
results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DELAY_THRESHOLD_DAYS = 1


def add_labels(sprints: pd.DataFrame) -> pd.DataFrame:
    df = sprints.copy()

    delay_days = (df["complete_date"] - df["end_date"]).dt.total_seconds() / 86400.0
    df["label_delay"] = (delay_days > DELAY_THRESHOLD_DAYS).astype(int)

    safe_denominator = df["committed_issue_count"].replace(0, np.nan)
    df["completion_ratio"] = df["completed_issue_count"] / safe_denominator

    # Binding definition: any committed issue unfinished by sprint end.
    df["label_spillover"] = (df["not_completed_count"] > 0).astype(int)
    df["at_risk"] = ((df["label_delay"] == 1) | (df["label_spillover"] == 1)).astype(int)
    return df
