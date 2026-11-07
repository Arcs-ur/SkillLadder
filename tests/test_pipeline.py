import json
import sys
from pathlib import Path

import pytest

import run_tau2_comparison as comparison
from fedskill.backends.base import TaskSpec
from fedskill.backends.tau2bench_backend import Tau2BenchBackend
from fedskill.models.skill import Skill
from fedskill.utils.run_manifest import atomic_write_json


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source"
    original = Skill("local", "d", "w", "p", id="original", source_task="private-task-17")
    evolved = Skill("general", "d", "w", "p", id="evolved", level=1, parent_ids=["original"],
                    quality_history=[{"task_id": "private-task-17"}])
    atomic_write_json(path / "level0_all.json", [original.to_dict()])
    atomic_write_json(path / "evolved_all.json", [original.to_dict(), evolved.to_dict()])
    atomic_write_json(path / "sessions.json", [dict(task_id="private-task-17", outcome="success")])
    return path


def test_config_cli_override_reaches_runner_and_producer(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "compare", "--config", str(comparison.REPO_ROOT / "configs/paper.yaml"),
        "--tau2-root", "/offline", "--agent-model", "consumer", "--user-model", "simulator",
        "--domain", "airline", "--max-steps", "7", "--synthesis-temperature", "0.8",
        "--task-ids", "custom", "--openai-api-key", "SECRET",
    ])
    args = comparison.parse_args()
    assert args.max_steps == 7
    assert args.evolve_rounds == 3
    assert args.strict_embeddings is True
    assert comparison.task_ids(args) == ["custom"]
    cmd = comparison.tau2_command(args, run_name="offline", ids=["custom"], agent="llm_agent")
    assert cmd[cmd.index("--max-steps") + 1] == "7"
    cmd = comparison.phase1_command(args, "offline", Path("/offline/skills"), 1)
    assert cmd[cmd.index("--synthesis-temperature") + 1] == "0.8"
    assert "SECRET" not in json.dumps(comparison.public_parameters(args))


@pytest.mark.parametrize("domain,count", [
    ("airline", 50), ("banking_knowledge", 97), ("retail", 114), ("telecom", 200),
])
@pytest.mark.parametrize("consumer", ["sonnet", "qwen3-32b", "qwen3-8b"])
def test_four_domains_three_consumers_dry_run_create_no_artifacts(
    domain, count, consumer, source, tmp_path, monkeypatch, capsys,
):
    output = tmp_path / "output"
    monkeypatch.setattr(comparison, "validate_tau2_checkout", lambda *a: None)
    monkeypatch.setattr(sys, "argv", [
        "compare", "--config", str(comparison.REPO_ROOT / "configs/paper.yaml"),
        "--tau2-root", str(tmp_path), "--agent-model", consumer, "--user-model", "sonnet",
        "--source-model", "sonnet", "--source-artifacts-dir", str(source),
        "--domain", domain, "--output-dir", str(output),
        "--conditions", "baseline,l0-static,l1plus-static,full-static,full-dynamic", "--dry-run",
    ])
    comparison.main()
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["task_ids"]) == count
    assert plan["generate_skills"] is False
    assert plan["fixed_library"] is True
    assert plan["parameters"]["task_set_name"] == ("telecom" if domain == "telecom" else None)
    assert plan["parameters"]["task_split_name"] == ("full" if domain == "telecom" else "base")
    assert len(plan["commands"]) == 5
    assert not output.exists()


def test_l0_and_shared_condition_snapshots_are_distinct_and_stable(source, tmp_path):
    full = source / "evolved_all.json"
    l0 = comparison.prepare_condition_library("l0-static", source, full, tmp_path / "run")
    shared = comparison.prepare_condition_library("l1plus-static", source, full, tmp_path / "run")
    assert [s.id for s in comparison.load_skills(l0)] == ["original"]
    payload = shared.read_text()
    public = json.loads(payload)
    assert len(public) == 1 and public[0]["level"] == 1
    assert public[0]["id"] != "evolved"
    assert public[0]["parent_ids"] == public[0]["quality_history"] == []
    assert "private-task-17" not in payload
    before = shared.read_bytes()
    comparison.prepare_condition_library("l1plus-static", source, full, tmp_path / "run")
    assert shared.read_bytes() == before


def test_transfer_runs_consumer_without_extracting_again(source, tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "validate_tau2_checkout", lambda *a: None)
    def no_producer(*a, **k):
        raise AssertionError("Transfer must not run extraction/evolution")
    monkeypatch.setattr(comparison, "run_logged", no_producer)
    called = []
    def run(args, *, run_name, **kwargs):
        called.append((args.agent_model, args.user_model, kwargs.get("injection_mode")))
        atomic_write_json(comparison.tau2_result_path(args, run_name), {"simulations": [
            dict(id=run_name, task_id="0", trial=0, reward_info={"reward": 1},
                 termination_reason="agent_stop")
        ]})
        return run_name
    monkeypatch.setattr(comparison, "tau2_run", run)
    monkeypatch.setattr(sys, "argv", [
        "compare", "--tau2-root", str(tmp_path), "--agent-model", "qwen3-8b", "--user-model", "sonnet",
        "--source-model", "sonnet", "--source-artifacts-dir", str(source),
        "--task-ids", "0", "--output-dir", str(tmp_path / "out"), "--disable-quality-update",
    ])
    comparison.main()
    assert called == [("qwen3-8b", "sonnet", None),
                      ("qwen3-8b", "sonnet", "static"), ("qwen3-8b", "sonnet", "dynamic")]


def test_backend_injects_real_candidate_file_and_reports_only_logged_selection(tmp_path, monkeypatch):
    agent_path = tmp_path / "src/tau2/agent/llm_agent.py"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_text("def create_fedskill_llm_agent(): pass")
    backend = Tau2BenchBackend(config={"tau2_root": str(tmp_path), "agent_llm": "consumer", "user_llm": "sim"})
    def run(cmd, *, env, **kwargs):
        assert cmd[cmd.index("--agent") + 1] == "fedskill_llm_agent"
        assert len(json.loads(Path(env["FEDSKILL_TAU2_SKILLS_PATH"]).read_text())) == 2
        name = cmd[cmd.index("--save-to") + 1]
        atomic_write_json(tmp_path / "data/simulations" / name / "results.json", {
            "simulations": [dict(id="sim", task_id="t", trial=0, reward_info={"reward": 1}, messages=[])]
        })
        Path(env["FEDSKILL_TAU2_SKILL_LOG_PATH"]).write_text(
            json.dumps(dict(task_id="t", selected_skills=[dict(id="selected")])) + "\n"
        )
    monkeypatch.setattr("fedskill.backends.tau2bench_backend.subprocess.run", run)
    backend.inject_skills([dict(id="selected", name="used"), dict(id="other", name="unused")])
    session = backend.run_task(TaskSpec(task_id="t", description="task"))
    assert session.skills_used == ["selected"]


def test_backend_missing_patch_explains_failure(tmp_path):
    backend = Tau2BenchBackend(config={"tau2_root": str(tmp_path)})
    with pytest.raises(RuntimeError, match="integration.patch"):
        backend.run_task(TaskSpec(task_id="t", description="task"), skills=[dict(id="s")])
