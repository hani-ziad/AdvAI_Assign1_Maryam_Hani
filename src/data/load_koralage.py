"""
Loader for the Koralage "Agile Scrum Sprint Velocity" dataset
(https://github.com/RandulaKoralage/AgileScrumSprintVelocityDataSet, commit
544b07e3f36e9c42040609effae5b68a13fdefc9), used as the working baseline
dataset for the Week 10 stage. See docs/data_access_and_versioning.md
for why this stands in for the proposal's primary dataset (TAWOS) here.

Normalizes the four projects' raw CSVs into the canonical schema used
throughout the pipeline (see src/data/schema.py).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# project name -> (sprints_csv_glob, issues_csv_glob)
_PROJECTS = {
    "usergrid": ("Usergrid Sprints *.csv", "Usergrid Issues [0-9]*.csv"),
    "aurora": ("Aurora Sprints *.csv", "Aurora Issues [0-9]*.csv"),
    "meso": ("MESO Sprint *.csv", "Mesos Stories *.csv"),
    "spring_xd": ("Spring XD Sprints *.csv", "Spring XD Issues [0-9]*.csv"),
}

_FOLDER_BY_PROJECT = {
    "usergrid": "Finalized Datasets for UserGrid",
    "aurora": "Finalized Datasets for Aurora Project",
    "meso": "Finalized Datasets for Meso Project",
    "spring_xd": "Finalized Datasets for Spring XD Project by Randula",
}


def _read_one_csv(folder: Path, pattern: str) -> pd.DataFrame:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {folder}")
    return pd.read_csv(matches[0], encoding="utf-8-sig")


def load_sprints(raw_dir: Path) -> pd.DataFrame:
    """Load and normalize the Sprints table for all four projects."""
    frames = []
    for project, (sprints_glob, _) in _PROJECTS.items():
        folder = raw_dir / _FOLDER_BY_PROJECT[project]
        df = _read_one_csv(folder, sprints_glob)
        if "totalNumberOfIssues" in df.columns and "total" not in df.columns:
            df = df.rename(columns={"totalNumberOfIssues": "total"})
        df = df.rename(
            columns={
                "sprintId": "raw_sprint_id",
                "sprintName": "sprint_name",
                "sprintState": "sprint_state",
                "sprintStartDate": "start_date",
                "sprintEndDate": "end_date",
                "sprintCompleteDate": "complete_date",
                "total": "committed_issue_count",
                "completedIssuesCount": "completed_issue_count",
                "issuesNotCompletedInCurrentSprint": "not_completed_count",
                "puntedIssues": "punted_count",
                "issueKeysAddedDuringSprint": "added_during_sprint_count",
                "NoOfDevelopers": "num_developers",
                "SprintLength": "sprint_length_days",
            }
        )
        for col in [
            "completedIssuesInitialEstimateSum",
            "issuesNotCompletedInitialEstimateSum",
            "puntedIssuesInitialEstimateSum",
            "issuesCompletedInAnotherSprintInitialEstimateSum",
            "completedIssuesEstimateSum",
        ]:
            if col not in df.columns:
                df[col] = 0

        # Total story points ever attached to the sprint, across every
        # outcome bucket. This is a best-effort proxy for "committed scope"
        # -- see docs/data_access_and_versioning.md ("known limitations")
        # for why this is not a perfectly leakage-free planning-time figure
        # in this dataset (TAWOS's issue-level timestamps allow a cleaner
        # cut and are the intended fix for the full-scale run).
        df["committed_story_points"] = (
            df["completedIssuesInitialEstimateSum"]
            + df["issuesNotCompletedInitialEstimateSum"]
            + df["puntedIssuesInitialEstimateSum"]
            + df["issuesCompletedInAnotherSprintInitialEstimateSum"]
        )
        df["delivered_story_points"] = df["completedIssuesEstimateSum"]

        df["project"] = project
        df["sprint_id"] = project + "__" + df["raw_sprint_id"].astype(str)

        for date_col in ["start_date", "end_date", "complete_date"]:
            df[date_col] = pd.to_datetime(df[date_col], format="%d-%b-%y", errors="coerce")

        keep = [
            "project",
            "sprint_id",
            "raw_sprint_id",
            "sprint_name",
            "sprint_state",
            "start_date",
            "end_date",
            "complete_date",
            "committed_issue_count",
            "completed_issue_count",
            "not_completed_count",
            "punted_count",
            "added_during_sprint_count",
            "committed_story_points",
            "delivered_story_points",
            "num_developers",
            "sprint_length_days",
        ]
        frames.append(df[keep])

    sprints = pd.concat(frames, ignore_index=True)
    sprints = sprints.dropna(subset=["start_date", "end_date"])
    sprints = sprints.sort_values(["project", "start_date"]).reset_index(drop=True)
    return sprints


def load_issues(raw_dir: Path) -> pd.DataFrame:
    """Load and normalize the Issues table for all four projects."""
    frames = []
    for project, (_, issues_glob) in _PROJECTS.items():
        folder = raw_dir / _FOLDER_BY_PROJECT[project]
        df = _read_one_csv(folder, issues_glob)
        df = df.rename(
            columns={
                "key": "issue_key",
                "sprint": "raw_sprint_id",
                "storyPoint": "story_point",
            }
        )
        df["project"] = project
        df["sprint_id"] = project + "__" + df["raw_sprint_id"].astype(str)
        keep = [c for c in ["project", "sprint_id", "issue_key", "issueType", "status",
                             "story_point", "priority", "assignee"] if c in df.columns]
        frames.append(df[keep].rename(columns={"issueType": "issue_type"}))

    issues = pd.concat(frames, ignore_index=True)
    return issues


def load_dataset(raw_dir: Path | str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience entry point returning (sprints, issues)."""
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "agile_scrum_sprint_velocity"
    raw_dir = Path(raw_dir)
    return load_sprints(raw_dir), load_issues(raw_dir)


if __name__ == "__main__":
    sprints, issues = load_dataset()
    print(f"Loaded {len(sprints)} sprints and {len(issues)} issues across "
          f"{sprints['project'].nunique()} projects.")
    print(sprints.groupby("project").size())
