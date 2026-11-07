from argparse import Namespace
import json

import pytest

import run_tau2_comparison as comparison
from fedskill.models.skill import Skill
from fedskill.server import aggregator
from fedskill.utils.results import valid_simulation_coverage
from fedskill.utils.run_manifest import atomic_write_json


def skill(identity):
    return Skill(identity, "description", "condition", "procedure", id=identity)


def test_model_change_has_distinct_run_identity():
    """BUG_MAP run_identity: a model change must not reuse completed results."""
    common = dict(domain="airline", num_trials=1, user_model="simulator", seed=42)
    first = Namespace(**common, agent_model="consumer-a")
    second = Namespace(**common, agent_model="consumer-b")
    assert comparison.run_prefix(first, ["0", "1"]) != comparison.run_prefix(second, ["0", "1"])


def test_task_order_changes_identity():
    args = Namespace(domain="airline", num_trials=1)
    assert comparison.run_prefix(args, ["0", "1"]) != comparison.run_prefix(args, ["1", "0"])


def test_completion_requires_unique_task_trial_reward_and_status():
    sims = [
        dict(id="a", task_id="0", trial=0, reward_info={"reward": 0}, termination_reason="max_steps"),
        dict(id="b", task_id="0", trial=1, reward_info={"reward": 1}, termination_reason="agent_stop"),
    ]
    assert valid_simulation_coverage(sims, ["0"], 2)
    assert not valid_simulation_coverage([sims[0], sims[0]], ["0"], 2)
    assert not valid_simulation_coverage(sims[:1], ["0"], 2)
    assert not valid_simulation_coverage(sims, ["0", "1"], 1)
    assert not valid_simulation_coverage([{**sims[0], "reward_info": {}}], ["0"], 1)


def test_evolution_resumes_same_clusters_and_rejects_changed_input(tmp_path, monkeypatch):
    calls = []
    def evolve(cluster, **kwargs):
        calls.append(cluster[0].id)
        if cluster[0].id == "c" and len(calls) == 2:
            raise RuntimeError("interrupted")
        return None
    monkeypatch.setattr(aggregator, "evolve_skill_cluster", evolve)
    server = aggregator.FedSkillServer(clustering_strategy="random")
    clusters = [[skill("a"), skill("b")], [skill("c"), skill("d")]]
    path = str(tmp_path / "checkpoint.json")
    with pytest.raises(RuntimeError):
        server.evolve(clusters, checkpoint_path=path)
    assert json.loads((tmp_path / "checkpoint.json").read_text())["next_cluster"] == 1
    result = server.evolve(clusters, checkpoint_path=path)
    assert [s.id for s in result] == ["a", "b", "c", "d"]
    assert calls == ["a", "c", "c"]
    with pytest.raises(ValueError, match="mismatched checkpoint"):
        server.evolve(list(reversed(clusters)), checkpoint_path=path)


def test_corrupt_checkpoint_is_preserved(tmp_path):
    path = tmp_path / "checkpoint.json"
    atomic_write_json(path, {"next_cluster": 999, "skills": []})
    before = path.read_bytes()
    server = aggregator.FedSkillServer(clustering_strategy="random")
    with pytest.raises(ValueError):
        server.evolve([[skill("a")]], checkpoint_path=str(path))
    assert path.read_bytes() == before


def test_round_resume_uses_saved_membership(tmp_path, monkeypatch):
    import run_tau2_phase1 as phase1
    calls = []
    def cluster(self):
        calls.append("cluster")
        return [self.all_local_skills]
    monkeypatch.setattr(aggregator.FedSkillServer, "cluster_skills", cluster)
    monkeypatch.setattr(aggregator, "evolve_skill_cluster", lambda *a, **k: None)
    original = [skill("a"), skill("b")]
    first = phase1.evolve_skills(original, ["task"], tmp_path, 1, clustering_strategy="random")
    second = phase1.evolve_skills(original, ["task"], tmp_path, 1, clustering_strategy="random")
    assert [s.id for s in first] == [s.id for s in second]
    assert calls == ["cluster"]


def test_interrupted_round_reloads_snapshot_instead_of_reclustering(tmp_path, monkeypatch):
    import run_tau2_phase1 as phase1
    calls = []
    def cluster(self):
        calls.append("cluster")
        return [self.all_local_skills]
    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted")
    monkeypatch.setattr(aggregator.FedSkillServer, "cluster_skills", cluster)
    monkeypatch.setattr(aggregator, "evolve_skill_cluster", interrupted)
    original = [skill("a"), skill("b")]
    with pytest.raises(RuntimeError):
        phase1.evolve_skills(original, ["task"], tmp_path, 1, clustering_strategy="random")
    monkeypatch.setattr(aggregator, "evolve_skill_cluster", lambda *a, **k: None)
    result = phase1.evolve_skills(original, ["task"], tmp_path, 1, clustering_strategy="random")
    assert [s.id for s in result] == ["a", "b"]
    assert calls == ["cluster"]


def test_force_run_uses_new_attempt(tmp_path, monkeypatch):
    args = Namespace(tau2_root=str(tmp_path), domain="airline", num_trials=1, force_runs=True,
                     task_set_name=None, task_split_name="base", max_concurrency=1,
                     agent_model="a", user_model="u", max_steps=30, openai_base_url="",
                     openai_api_key="")
    names = []
    def run(cmd, *a):
        name = cmd[cmd.index("--save-to") + 1]
        names.append(name)
        atomic_write_json(comparison.tau2_result_path(args, name), {"simulations": [
            dict(id=name, task_id="0", trial=0, reward_info={"reward": 0}, termination_reason="max_steps")
        ]})
    monkeypatch.setattr(comparison, "run_logged", run)
    first = comparison.tau2_run(args, run_name="test", ids=["0"], agent="llm_agent", log_path=tmp_path / "log")
    second = comparison.tau2_run(args, run_name="test", ids=["0"], agent="llm_agent", log_path=tmp_path / "log")
    assert first != second
    assert len(names) == 2
