import pytest

import non_claude_judge_review as judge
import ablate_clustering_airline as clustering
from fedskill.models.skill import Skill
from fedskill.utils.run_manifest import atomic_write_json


def test_judge_requires_explicit_complete_artifacts(tmp_path):
    with pytest.raises(ValueError, match="Missing artifact"):
        judge.build_pairs([tmp_path / "missing"], {1}, 42)
    parent = Skill("p", "d", "w", "p", id="parent", source_task="t", score=0)
    child = Skill("c", "d", "w", "p", id="child", level=1, parent_ids=["parent"], score=5)
    atomic_write_json(tmp_path / "level0_all.json", [parent.to_dict()])
    atomic_write_json(tmp_path / "round1_skills.json", [child.to_dict()])
    atomic_write_json(tmp_path / "sessions.json", [dict(task_description="task", metadata={"domain": "retail"})])
    pairs = judge.build_pairs([tmp_path], {1}, 42)
    assert pairs[0].domain == "retail"
    assert pairs[0].claude_parent_score == 0
    atomic_write_json(tmp_path / "level0_all.json", [])
    with pytest.raises(ValueError, match="Missing parent"):
        judge.build_pairs([tmp_path], {1}, 42)


def test_judge_scores_full_requested_task_set():
    assert judge.score_from_per_task([dict(relevance=8, applicability=5)], 1) == 8
    with pytest.raises(ValueError, match="exactly"):
        judge.score_from_per_task([], 1)
    with pytest.raises(ValueError, match="finite"):
        judge.score_from_per_task([dict(relevance=10, applicability=True)], 1)


def test_clustering_audit_reuses_saved_membership(tmp_path, monkeypatch):
    monkeypatch.setattr(clustering, "OUT_DIR", tmp_path)
    monkeypatch.setattr(clustering, "PER_CLUSTER_CSV", tmp_path / "per_cluster.csv")
    skill = Skill("p", "d", "w", "p", id="parent")
    class Strategy:
        calls = 0
        def cluster(self, skills):
            self.calls += 1
            return [skills]
    strategy = Strategy()
    monkeypatch.setattr(clustering, "get_clustering_strategy", lambda *a, **k: strategy)
    spec = clustering.StrategySpec("test", "test", "test")
    clustering.run_strategy(spec, [skill], ["task"], set())
    clustering.run_strategy(spec, [skill], ["task"], set())
    assert strategy.calls == 1
    with pytest.raises(ValueError, match="mismatch"):
        clustering.run_strategy(spec, [skill], ["changed task"], set())
