"""Verify the actual patched tau2 agent offline, without model calls."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from fedskill.utils.run_manifest import TAU2_REVISION, atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", required=True)
    args = parser.parse_args()
    root = Path(args.tau2_root).resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != TAU2_REVISION:
        raise ValueError(f"Expected tau2 {TAU2_REVISION}, got {revision}")
    patch = Path(__file__).resolve().parents[1] / "integrations/tau2/fedskill_tau2_integration.patch"
    subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=root, check=True)

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TAU2_DATA_DIR"] = str(root / "data")
    os.environ["FEDSKILL_TAU2_INJECTION_MODE"] = "static"
    os.environ["FEDSKILL_TAU2_MAX_SKILLS"] = "10"
    os.environ["FEDSKILL_TAU2_SCORE_MODE"] = "token_overlap"
    sys.path.insert(0, str(root / "src"))
    def forbid_network(*args, **kwargs):
        raise RuntimeError("Network disabled during offline integration check")
    socket.socket.connect = forbid_network
    socket.create_connection = forbid_network

    from tau2.agent import llm_agent
    from tau2.data_model.tasks import Task
    from tau2.registry import registry
    from tau2.runner.helpers import get_tasks

    llm_agent.generate = forbid_network
    factory = registry.get_agent_factory("fedskill_llm_agent")
    split_dir = Path(__file__).resolve().parents[1] / "configs/task_splits"
    for domain, filename in (
        ("airline", "airline_50.json"), ("banking_knowledge", "banking_knowledge_97.json"),
        ("retail", "retail_114.json"), ("telecom", "telecom_200.json"),
    ):
        ids = json.loads((split_dir / filename).read_text())
        tasks = get_tasks(task_set_name=domain, task_split_name="full" if domain == "telecom" else "base",
                          task_ids=ids)
        assert {task.id for task in tasks} == set(ids), f"Invalid pinned task IDs for {domain}"
    with tempfile.TemporaryDirectory() as temporary:
        skill_path = Path(temporary) / "skills.json"
        log_path = Path(temporary) / "usage.jsonl"
        os.environ["FEDSKILL_TAU2_SKILLS_PATH"] = str(skill_path)
        os.environ["FEDSKILL_TAU2_SKILL_LOG_PATH"] = str(log_path)
        retired = dict(id="retired", name="retired-name", procedure="RETIRED_PROCEDURE", retired=True)
        active = dict(id="active", name="active-name", procedure="ACTIVE_PROCEDURE",
                      level=1, score=1, quality_score=0, quality_state_status="observed")
        atomic_write_json(skill_path, [retired])
        task = Task(id="offline-task", user_scenario={"instructions": "offline scenario"})
        agent = factory(tools=[], domain_policy="policy", llm="offline", task=task)
        assert agent._select_skills() == []
        assert "RETIRED_PROCEDURE" not in agent.system_prompt

        atomic_write_json(skill_path, [retired, active, active])
        agent = factory(tools=[], domain_policy="policy", llm="offline", task=task)
        assert agent._select_skills() == [active]
        assert agent._skill_quality(active) == 0
        assert agent._skill_quality({"score": 1}) == 0.1
        assert "ACTIVE_PROCEDURE" in agent.system_prompt
        agent.system_prompt  # Re-reading a prompt must not log another injection.
        events = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(events) == 2
        assert events[-1]["task_id"] == "offline-task"
        assert [s["id"] for s in events[-1]["selected_skills"]] == ["active"]
        os.environ["FEDSKILL_TAU2_MAX_SKILLS"] = "0"
        assert agent._select_skills() == []
    print("Pinned tau2: four task splits, agent import, registry, retired/dedup/zero-quality selection and actual usage logging passed.")


if __name__ == "__main__":
    main()
