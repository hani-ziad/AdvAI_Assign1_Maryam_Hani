"""Guards the current zero-scope and cold-start behavior."""
from src.data.load_tawos import load_reconstructed_sprints
from src.labels import add_labels
from src.features import add_features, build_model_frame


def _frame():
    sprints = load_reconstructed_sprints()
    return build_model_frame(add_features(add_labels(sprints)), "label_spillover")


def test_committed_story_points_zero_when_no_committed_issues():
    zero_rows = _frame()[lambda x: x["committed_issue_count"] == 0]
    assert len(zero_rows) > 0
    assert (zero_rows["committed_story_points"] == 0).all()


def test_zero_committed_scope_sprints_are_spillover_negative():
    zero_rows = _frame()[lambda x: x["committed_issue_count"] == 0]
    assert zero_rows["label_spillover"].eq(0).all()


def test_current_cold_start_drops_only_one_eligible_sprint_per_project():
    sprints = load_reconstructed_sprints()
    frame = build_model_frame(add_features(add_labels(sprints)), "label_spillover")
    assert len(sprints) == 2610
    assert sprints["project"].nunique() == 8
    assert len(frame) == 2602
