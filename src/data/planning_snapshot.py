"""Reconstruct TAWOS sprint scope exactly as far as the timestamped export allows.

The static ``Issue.Sprint_ID`` column is not sufficient for a planning-time
snapshot because Jira's Sprint field can change after a sprint starts.  This
module rolls the Sprint and Story Points fields backwards through Change_Log
so each sprint is evaluated using the issue membership and story-point value
that existed at the sprint's own Start_Date.

Important identifier detail
---------------------------
TAWOS ``Issue.Sprint_ID`` uses the database Sprint.ID, while Sprint changes in
``Change_Log`` store Jira sprint identifiers (Sprint.JiraID).  The two are not
interchangeable.  Parsing is therefore project-scoped and uses the correct
identifier namespace for each source.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import pandas as pd


SPRINT_FIELD_NAMES = {"sprint", "sprints"}
STORY_POINT_FIELD_NAMES = {
    "storypoint",
    "storypoints",
    "storypointestimate",
    "storypointsestimate",
}


def _norm_field(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _field_kind(field) -> str | None:
    n = _norm_field(field)
    if n in SPRINT_FIELD_NAMES or n.endswith("sprint"):
        return "sprint"
    if n in STORY_POINT_FIELD_NAMES or ("story" in n and "point" in n):
        return "story_point"
    return None


def _as_number(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        return float(m.group(0)) if m else None


def _project_maps(sprints: pd.DataFrame):
    jira_to_canonical: dict[str, dict[str, str]] = {}
    raw_to_canonical: dict[str, dict[str, str]] = {}
    name_to_canonical: dict[str, dict[str, str]] = {}
    for project, g in sprints.groupby("project"):
        jmap, rmap, nmap = {}, {}, {}
        for row in g.itertuples(index=False):
            canonical = row.sprint_id
            if not pd.isna(row.raw_sprint_id):
                rmap[str(int(float(row.raw_sprint_id)))] = canonical
            if hasattr(row, "jira_sprint_id") and not pd.isna(row.jira_sprint_id):
                jmap[str(int(float(row.jira_sprint_id)))] = canonical
            if hasattr(row, "sprint_name") and not pd.isna(row.sprint_name):
                nmap[str(row.sprint_name).strip().lower()] = canonical
        jira_to_canonical[project] = jmap
        raw_to_canonical[project] = rmap
        name_to_canonical[project] = nmap
    return jira_to_canonical, raw_to_canonical, name_to_canonical


def _parse_change_sprint_set(value, string_value, *, jira_map, name_map):
    """Parse a Change_Log Sprint value into target canonical sprint ids.

    Change_Log numeric values are Jira sprint ids. Unknown ids are legitimate:
    an issue may have belonged to a sprint outside the selected/retained sprint
    set, so unknown tokens are ignored rather than treated as parse failures.
    String values are used only as a fallback when the numeric value is absent;
    we match exact sprint names and never extract arbitrary numbers from names.
    """
    if value is not None and not pd.isna(value):
        text = str(value).strip()
        if not text or text.lower() in {"none", "null", "nan", "[]"}:
            return set(), 0
        tokens = re.findall(r"\d+", text)
        known = {jira_map[t] for t in tokens if t in jira_map}
        unknown = sum(t not in jira_map for t in tokens)
        return known, unknown

    if string_value is None or pd.isna(string_value):
        return set(), 0
    text = str(string_value).strip().lower()
    if not text or text in {"none", "null", "nan", "[]"}:
        return set(), 0
    # Jira normally provides the numeric value. This fallback handles test
    # fixtures or unusual exports where only the exact sprint name is present.
    if text in name_map:
        return {name_map[text]}, 0
    # A comma-separated name representation can contain several exact names.
    known = {canonical for name, canonical in name_map.items() if name and name in text}
    return known, 0 if known else 1


def _story_before(ev):
    v = _as_number(ev.from_value)
    return v if v is not None else _as_number(ev.from_string)


def _story_after(ev):
    v = _as_number(ev.to_value)
    return v if v is not None else _as_number(ev.to_string)


def _validate_inputs(sprints: pd.DataFrame, issues: pd.DataFrame, change_log: pd.DataFrame) -> None:
    required_s = {"sprint_id", "raw_sprint_id", "jira_sprint_id", "project", "start_date", "end_date"}
    required_i = {"raw_issue_id", "issue_key", "project", "raw_sprint_id", "creation_date", "resolution_date", "story_point"}
    required_c = {"raw_issue_id", "field", "from_value", "to_value", "from_string", "to_string", "change_date"}
    missing = {
        "sprints": sorted(required_s - set(sprints.columns)),
        "issues": sorted(required_i - set(issues.columns)),
        "change_log": sorted(required_c - set(change_log.columns)),
    }
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise ValueError(f"Planning-time reconstruction missing required columns: {missing}")


def derive_planning_time_aggregates(
    sprints: pd.DataFrame,
    issues: pd.DataFrame,
    change_log: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Reconstruct committed scope and story points at each sprint Start_Date.

    For issues with Sprint/Story-Point history, the most recent Change_Log
    after-state is used as the terminal state and events are reversed as we
    walk sprint starts backwards in time.  For issues with no history for a
    field, the static Issue value is used as the unchanged value.
    """
    _validate_inputs(sprints, issues, change_log)
    sprints = sprints.sort_values(["project", "start_date", "sprint_id"]).copy()
    issues = issues.copy()
    change_log = change_log.copy()

    jira_maps, raw_maps, name_maps = _project_maps(sprints)
    sprint_rows_by_project = {
        p: list(g.sort_values("start_date", ascending=False).itertuples(index=False))
        for p, g in sprints.groupby("project")
    }

    relevant = change_log[change_log["field"].map(_field_kind).notna()].copy()
    relevant = relevant.dropna(subset=["raw_issue_id", "change_date"])
    relevant = relevant.sort_values(
        ["raw_issue_id", "change_date", "change_id" if "change_id" in relevant.columns else "raw_issue_id"],
        ascending=[True, False, False],
    )
    changes_by_issue = defaultdict(list)
    for row in relevant.itertuples(index=False):
        changes_by_issue[row.raw_issue_id].append(row)

    committed = defaultdict(list)
    audit = {
        "issues_seen": int(len(issues)),
        "relevant_change_events": int(len(relevant)),
        "sprint_change_events": int((relevant["field"].map(_field_kind) == "sprint").sum()),
        "story_point_change_events": int((relevant["field"].map(_field_kind) == "story_point").sum()),
        "issues_with_sprint_history": 0,
        "issues_with_story_point_history": 0,
        "unknown_sprint_id_tokens_ignored": 0,
        "issue_sprint_memberships_at_start": 0,
        "memberships_excluded_issue_created_after_start": 0,
        "memberships_excluded_resolved_before_start": 0,
        "latest_sprint_history_vs_static_disagreements": 0,
        "latest_story_history_vs_static_disagreements": 0,
    }

    for issue in issues.itertuples(index=False):
        project = issue.project
        if project not in sprint_rows_by_project:
            continue
        issue_id = issue.raw_issue_id
        events = changes_by_issue.get(issue_id, [])
        jira_map = jira_maps[project]
        raw_map = raw_maps[project]
        name_map = name_maps[project]

        sprint_events = [e for e in events if _field_kind(e.field) == "sprint"]
        story_events = [e for e in events if _field_kind(e.field) == "story_point"]

        # Terminal Sprint state: prefer the timestamped latest Change_Log event.
        if sprint_events:
            audit["issues_with_sprint_history"] += 1
            membership, unknown = _parse_change_sprint_set(
                sprint_events[0].to_value,
                sprint_events[0].to_string,
                jira_map=jira_map,
                name_map=name_map,
            )
            audit["unknown_sprint_id_tokens_ignored"] += unknown

            static_membership = set()
            if not pd.isna(issue.raw_sprint_id):
                key = str(int(float(issue.raw_sprint_id)))
                if key in raw_map:
                    static_membership = {raw_map[key]}
            if static_membership and not static_membership.issubset(membership):
                audit["latest_sprint_history_vs_static_disagreements"] += 1
        else:
            membership = set()
            if not pd.isna(issue.raw_sprint_id):
                key = str(int(float(issue.raw_sprint_id)))
                if key in raw_map:
                    membership = {raw_map[key]}

        # Terminal Story Point state: likewise prefer latest timestamped event.
        if story_events:
            audit["issues_with_story_point_history"] += 1
            story_point = _story_after(story_events[0])
            static_sp = None if pd.isna(issue.story_point) else float(issue.story_point)
            if story_point is not None and static_sp is not None and abs(story_point - static_sp) > 1e-6:
                audit["latest_story_history_vs_static_disagreements"] += 1
        else:
            story_point = None if pd.isna(issue.story_point) else float(issue.story_point)

        created = issue.creation_date
        resolved = issue.resolution_date
        idx = 0
        for sprint in sprint_rows_by_project[project]:
            t = sprint.start_date

            # Reverse every field change that occurred strictly after sprint start.
            # A change exactly at Start_Date is treated as available at start.
            while idx < len(events) and events[idx].change_date > t:
                ev = events[idx]
                kind = _field_kind(ev.field)
                if kind == "sprint":
                    membership, unknown = _parse_change_sprint_set(
                        ev.from_value,
                        ev.from_string,
                        jira_map=jira_map,
                        name_map=name_map,
                    )
                    audit["unknown_sprint_id_tokens_ignored"] += unknown
                elif kind == "story_point":
                    story_point = _story_before(ev)
                idx += 1

            if sprint.sprint_id not in membership:
                continue
            if created is not None and not pd.isna(created) and created > t:
                audit["memberships_excluded_issue_created_after_start"] += 1
                continue
            # A resolved issue is not an unresolved planning commitment. Keeping
            # it would let a pre-start outcome leak into a nominal planning scope.
            if resolved is not None and not pd.isna(resolved) and resolved < t:
                audit["memberships_excluded_resolved_before_start"] += 1
                continue

            is_completed = bool(
                resolved is not None
                and not pd.isna(resolved)
                and resolved <= sprint.end_date
            )
            committed[sprint.sprint_id].append(
                {
                    "issue_key": issue.issue_key,
                    "story_point_at_start": story_point,
                    "is_completed": is_completed,
                }
            )
            audit["issue_sprint_memberships_at_start"] += 1

    rows = []
    for sprint in sprints.itertuples(index=False):
        items = committed.get(sprint.sprint_id, [])
        committed_count = len(items)
        completed_count = sum(int(x["is_completed"]) for x in items)
        story_points = sum((x["story_point_at_start"] or 0.0) for x in items)
        delivered_points = sum(
            (x["story_point_at_start"] or 0.0) for x in items if x["is_completed"]
        )
        rows.append(
            {
                "sprint_id": sprint.sprint_id,
                "committed_issue_count": committed_count,
                "committed_story_points": story_points,
                "completed_issue_count": completed_count,
                "not_completed_count": committed_count - completed_count,
                "delivered_story_points": delivered_points,
            }
        )

    agg = pd.DataFrame(rows)
    result = sprints.merge(agg, on="sprint_id", how="left")
    result["sprint_length_days"] = (
        result["end_date"] - result["start_date"]
    ).dt.total_seconds() / 86400.0
    result["scope_source"] = "change_log_reconstructed_at_sprint_start"

    audit["sprints_reconstructed"] = int(len(result))
    audit["zero_scope_sprints"] = int((result["committed_issue_count"] == 0).sum())
    audit["nonzero_scope_sprints"] = int((result["committed_issue_count"] > 0).sum())
    return result, audit
