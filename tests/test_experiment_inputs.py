import json
from types import SimpleNamespace

import pytest

import build_awm_skills
import run_awm_baseline
import ablate_gates_r1
from fedskill.models.skill import Skill


@pytest.fixture
def source_artifacts(tmp_path):
    folder = tmp_path / "source" / "skills_from_train"
    folder.mkdir(parents=True)
    a = Skill(name="success", description="d", when_to_use="t", procedure="p",
              id="original-success", source_task="a")
    b = Skill(name="failure", description="d", when_to_use="t", procedure="p",
              id="original-failure", source_task="b")
    (folder / "sessions.json").write_text(json.dumps([
        {"task_id": "a", "session_id": "session-a", "outcome": "success"},
        {"task_id": "b", "session_id": "session-b", "outcome": "failure"},
    ]))
    (folder / "level0_all.json").write_text(json.dumps([a.to_dict(), b.to_dict(), a.to_dict()]))
    (folder / "evolved_all.json").write_text(json.dumps([
        Skill(name="merged", description="d", when_to_use="t", procedure="p",
              level=1, parent_ids=[a.id, b.id]).to_dict(),
    ]))
    return folder.parent


@pytest.mark.parametrize("entrypoint", ["standalone", "runner"])
def test_awm_keeps_successful_original_skills(source_artifacts, tmp_path, entrypoint):
    """[defect-probing] 合成吸收父技能不会改变 AWM 候选。"""
    destination = tmp_path / "awm.json"
    if entrypoint == "standalone":
        build_awm_skills.build(source_artifacts, destination)
    else:
        run_awm_baseline.build_skills(SimpleNamespace(
            train_dir=source_artifacts, skills_out=destination, domain="airline",
        ))
    data = json.loads(destination.read_text())
    assert [skill["id"] for skill in data] == ["original-success"]


def test_gate_ablation_uses_all_saved_candidate_clusters(tmp_path, monkeypatch):
    parents = [
        Skill(name=str(i), description="d", when_to_use="t", procedure="p",
              id=f"parent-{i}", source_task=f"task-{i}").to_dict()
        for i in range(5)
    ]
    (tmp_path / "round1_clusters.json").write_text(json.dumps([
        parents[:2], parents[2:4], parents[4:],
    ]))
    monkeypatch.setattr(ablate_gates_r1, "ARTIFACT_DIR", tmp_path)
    clusters, identifiers = ablate_gates_r1.load_clusters_from_artifact()
    assert [[skill.id for skill in cluster] for cluster in clusters] == [
        ["parent-0", "parent-1"], ["parent-2", "parent-3"],
    ]
    assert len(set(identifiers)) == 2


def test_awm_rejects_ambiguous_multi_session_provenance(source_artifacts):
    path = source_artifacts / "skills_from_train" / "sessions.json"
    sessions = json.loads(path.read_text())
    sessions.append({"task_id": "a", "session_id": "session-a2", "outcome": "failure"})
    path.write_text(json.dumps(sessions))
    with pytest.raises(ValueError, match="source_session_id"):
        build_awm_skills.success_only_l0(source_artifacts)
