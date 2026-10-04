"""
Loader and aggregation helpers for the TAWOS sprint-risk study.

Corrections in this version
---------------------------
1. Only completed/closed sprints with coherent timestamps enter modeling.
   ACTIVE and FUTURE sprints are excluded before labels/features are built.
2. The loader keeps optional temporal columns (Issue.Creation_Date,
   Sprint.JiraID/Activated_Date, Issue.ID) when the export contains them.
3. The current two-file export can still be reproduced in ``static`` scope
   mode, but that mode is explicitly marked as *not* a verified planning-time
   snapshot. A fully leakage-safe planning-time scope requires the TAWOS
   Change_Log export and is handled by ``src.data.planning_snapshot``.

Expected files in data/raw/tawos:
  - tawos_sprint.csv
  - tawos_issue.csv
Optional for the planning-time reconstruction:
  - tawos_change_log.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_SPRINT_STATES = {"CLOSED"}


def _read_flexible_csv(path: Path) -> pd.DataFrame:
    """Read comma- or tab-separated exports produced by MySQL/Workbench."""
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def load_sprints(raw_dir: Path, *, modeling_only: bool = True) -> pd.DataFrame:
    path = Path(raw_dir) / "tawos_sprint.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See scripts/export_tawos_subset.sql for the export query."
        )

    df = _read_flexible_csv(path)
    df = df.rename(
        columns={
            "ID": "raw_sprint_id",
            "JiraID": "jira_sprint_id",
            "Name": "sprint_name",
            "State": "sprint_state",
            "Start_Date": "start_date",
            "Activated_Date": "activated_date",
            "End_Date": "end_date",
            "Complete_Date": "complete_date",
            "Project_Name": "project",
        }
    )
    raw_sid = pd.to_numeric(df["raw_sprint_id"], errors="coerce").astype("Int64").astype(str)
    df["sprint_id"] = df["project"].astype(str) + "__" + raw_sid
    for c in ["start_date", "activated_date", "end_date", "complete_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if modeling_only:
        # A delay/late-closure label is only defined after a sprint has closed.
        # Also remove impossible timestamp rows (e.g., closure before start).
        state = df["sprint_state"].astype(str).str.upper()
        mask = state.isin(REQUIRED_SPRINT_STATES)
        mask &= df["start_date"].notna() & df["end_date"].notna() & df["complete_date"].notna()
        mask &= df["end_date"] >= df["start_date"]
        mask &= df["complete_date"] >= df["start_date"]
        df = df.loc[mask].copy()

    return df.reset_index(drop=True)


def load_issues(raw_dir: Path) -> pd.DataFrame:
    path = Path(raw_dir) / "tawos_issue.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See scripts/export_tawos_subset.sql for the export query."
        )

    df = _read_flexible_csv(path)
    df = df.rename(
        columns={
            "ID": "raw_issue_id",
            "Issue_Key": "issue_key",
            "Type": "issue_type",
            "Status": "status",
            "Story_Point": "story_point",
            "Priority": "priority",
            "Creation_Date": "creation_date",
            "Sprint_ID": "raw_sprint_id",
            "Project_Name": "project",
            "Resolution_Date": "resolution_date",
            "Assignee_ID": "assignee",
        }
    )
    raw_sid = pd.to_numeric(df["raw_sprint_id"], errors="coerce").astype("Int64").astype(str)
    df["sprint_id"] = df["project"].astype(str) + "__" + raw_sid
    for c in ["creation_date", "resolution_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_change_log(raw_dir: Path, *, required: bool = False) -> pd.DataFrame | None:
    path = Path(raw_dir) / "tawos_change_log.csv"
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"{path} not found. A true planning-time sprint-scope reconstruction "
                "requires the Change_Log export. Run the updated SQL in "
                "scripts/export_tawos_subset.sql against the official TAWOS database."
            )
        return None

    df = _read_flexible_csv(path)
    df = df.rename(
        columns={
            "ID": "change_id",
            "Issue_ID": "raw_issue_id",
            "Field": "field",
            "From_Value": "from_value",
            "To_Value": "to_value",
            "From_String": "from_string",
            "To_String": "to_string",
            "Change_Type": "change_type",
            "Creation_Date": "change_date",
            "Project_Name": "project",
        }
    )
    df["change_date"] = pd.to_datetime(df["change_date"], errors="coerce")
    return df


def derive_sprint_aggregates(sprints: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the legacy/static TAWOS sprint aggregation on the *eligible*
    (closed and timestamp-valid) sprint set.

    IMPORTANT: Issue.Sprint_ID is a present-day/static foreign key. Therefore
    these aggregates are suitable for reproducing the old analysis and for the
    interim Issues-1/2 rerun, but they are not sufficient to claim that scope
    was exactly known at sprint start. For the current analysis use
    ``derive_planning_time_aggregates`` from ``src.data.planning_snapshot``.
    """
    merged = issues.merge(
        sprints[["sprint_id", "start_date", "end_date"]], on="sprint_id", how="inner"
    )
    merged["is_completed"] = (
        merged["resolution_date"].notna() & (merged["resolution_date"] <= merged["end_date"])
    )

    agg = merged.groupby("sprint_id").agg(
        committed_issue_count=("issue_key", "count"),
        committed_story_points=("story_point", "sum"),
        completed_issue_count=("is_completed", "sum"),
        num_developers=("assignee", "nunique")
        if "assignee" in merged.columns
        else ("issue_key", "count"),
    )
    agg["not_completed_count"] = agg["committed_issue_count"] - agg["completed_issue_count"]
    agg["delivered_story_points"] = (
        merged[merged["is_completed"]].groupby("sprint_id")["story_point"].sum()
    ).reindex(agg.index).fillna(0)

    result = sprints.merge(agg.reset_index(), on="sprint_id", how="left")
    result["sprint_length_days"] = (result["end_date"] - result["start_date"]).dt.total_seconds() / 86400.0

    agg_cols = [
        "committed_issue_count",
        "committed_story_points",
        "completed_issue_count",
        "num_developers",
        "not_completed_count",
        "delivered_story_points",
    ]
    result[agg_cols] = result[agg_cols].fillna(0)
    result["scope_source"] = "static_issue_sprint_id"
    return result


def load_dataset(
    raw_dir: Path | str | None = None, *, modeling_only: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "tawos"
    raw_dir = Path(raw_dir)
    return load_sprints(raw_dir, modeling_only=modeling_only), load_issues(raw_dir)


if __name__ == "__main__":
    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "tawos"
    raw_sprints = load_sprints(raw_dir, modeling_only=False)
    sprints, issues = load_dataset(raw_dir, modeling_only=True)
    agg = derive_sprint_aggregates(sprints, issues)
    print(f"Raw sprint rows: {len(raw_sprints)}")
    print(f"Eligible CLOSED/timestamp-valid sprints: {len(sprints)}")
    print(f"Projects: {sprints['project'].nunique()}")
    print(f"Static issue rows: {len(issues)}")
    print(f"Aggregated eligible sprints: {len(agg)}")


def load_reconstructed_sprints(
    raw_dir: Path | str | None = None,
    *,
    return_audit: bool = False,
):
    """Load the TAWOS data and reconstruct sprint-start scope.

    This is the binding loader for the current analysis. It excludes
    ACTIVE/FUTURE/invalid sprints, uses the complete selected-project issue
    export, and derives sprint scope/story points from Change_Log at each
    sprint's own start timestamp.
    """
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "tawos"
    raw_dir = Path(raw_dir)
    sprints = load_sprints(raw_dir, modeling_only=True)
    issues = load_issues(raw_dir)
    change_log = load_change_log(raw_dir, required=True)
    from src.data.planning_snapshot import derive_planning_time_aggregates

    reconstructed, audit = derive_planning_time_aggregates(sprints, issues, change_log)
    if return_audit:
        return reconstructed, audit
    return reconstructed
