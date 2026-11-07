"""tau2-bench backend for SkillLadder.

This adapter converts tau2-bench simulation results into FedSkill's common
``SessionData`` format so the existing skill extractor and evolver can be reused.

Config keys:
    tau2_root:       Path to the tau2-bench repository root.
    run_name:        Optional single run name under data/simulations to collect.
    results_file:    Optional explicit path to a tau2 results.json file.
    domain:          tau2 domain for active runs/task loading (default: airline).
    python_cmd:      Python executable for tau2 CLI runs.
    agent_llm:       LiteLLM model name for active runs.
    user_llm:        LiteLLM model name for active runs.
    openai_base_url: Optional OpenAI-compatible base URL for LiteLLM.
    openai_api_key:  Optional API key for LiteLLM.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from fedskill.backends.base import ExecutionBackend, SessionData, TaskSpec
from fedskill.backends.registry import register_backend
from fedskill.utils.run_manifest import atomic_write_json, bind_state, fingerprint


class Tau2BenchBackend(ExecutionBackend):
    """Backend that adapts tau2-bench runs to SkillLadder sessions."""

    def __init__(self, name: str = "tau2bench", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        tau2_root = self.config.get("tau2_root") or os.getenv("TAU2_ROOT")
        if not tau2_root:
            raise ValueError("tau2_root is required (or set TAU2_ROOT)")
        self._tau2_root = Path(tau2_root).expanduser().resolve()
        if not self._tau2_root.exists():
            raise ValueError(f"tau2_root does not exist: {self._tau2_root}")

        self._src_dir = self._tau2_root / "src"
        self._simulations_dir = self._tau2_root / "data" / "simulations"
        self._run_name = self.config.get("run_name")
        self._results_file = self.config.get("results_file")
        self._domain = self.config.get("domain", "airline")
        self._task_split_name = self.config.get("task_split_name", "base")
        self._python_cmd = self.config.get("python_cmd", sys.executable)
        self._agent_llm = self.config.get("agent_llm", "openai/gpt-4.1")
        self._user_llm = self.config.get("user_llm", self._agent_llm)
        self._max_steps = int(self.config.get("max_steps", 30))
        self._max_retries = int(self.config.get("max_retries", 0))
        self._max_concurrency = int(self.config.get("max_concurrency", 1))
        self._openai_base_url = self.config.get("openai_base_url")
        self._openai_api_key = self.config.get("openai_api_key")
        self._session_history: list[SessionData] = []
        self._injected_skills: list[dict] = []

    # ------------------------------------------------------------------
    # ExecutionBackend interface
    # ------------------------------------------------------------------

    def run_task(self, task: TaskSpec, skills: list[dict] | None = None) -> SessionData:
        """Run a real patched agent; report only selections recorded by that agent."""
        if skills is not None:
            self._injected_skills = deepcopy(skills)
        candidates = self._injected_skills
        agent = "fedskill_llm_agent" if candidates else "llm_agent"
        if candidates:
            agent_path = self._src_dir / "tau2/agent/llm_agent.py"
            if not agent_path.exists() or "def create_fedskill_llm_agent(" not in agent_path.read_text():
                raise RuntimeError("Skill injection requires integrations/tau2/fedskill_tau2_integration.patch.")
        identity = {
            "domain": self._domain, "task_id": task.task_id, "split": self._task_split_name,
            "agent": agent, "agent_llm": self._agent_llm, "user_llm": self._user_llm,
            "max_steps": self._max_steps, "seed": self.config.get("seed", 42),
            "skills": candidates, "injection_mode": self.config.get("injection_mode", "static"),
            "max_skills": self.config.get("max_skills", 10),
        }
        prefix = self.config.get("active_run_name") or f"fedskill_tau2_{self._domain}_{task.task_id}_{uuid.uuid4().hex[:6]}"
        run_name = f"{prefix}_{fingerprint(identity)[:12]}"
        run_dir = self._simulations_dir / run_name
        bind_state(run_dir / "skillladder_manifest.json", identity)
        selection_log = run_dir / "skill_usage.jsonl"
        cmd = [
            self._python_cmd,
            "-m",
            "tau2.cli",
            "run",
            "--domain",
            self._domain,
            "--task-split-name",
            self._task_split_name,
            "--agent",
            agent,
            "--task-ids",
            task.task_id,
            "--num-trials",
            "1",
            "--max-concurrency",
            str(self._max_concurrency),
            "--agent-llm",
            self._agent_llm,
            "--user-llm",
            self._user_llm,
            "--agent-llm-args",
            json.dumps(self.config.get("agent_llm_args", {"temperature": 0.2})),
            "--user-llm-args",
            json.dumps(self.config.get("user_llm_args", {"temperature": 0.2})),
            "--max-steps",
            str(self._max_steps),
            "--max-retries",
            str(self._max_retries),
            "--seed",
            str(self.config.get("seed", 42)),
            "--auto-resume",
            "--save-to",
            run_name,
            "--log-level",
            self.config.get("log_level", "INFO"),
        ]

        env = self._tau2_env()
        if candidates:
            skills_file = run_dir / "skillladder_skills.json"
            atomic_write_json(skills_file, candidates)
            env["FEDSKILL_TAU2_SKILLS_PATH"] = str(skills_file)
            env["FEDSKILL_TAU2_INJECTION_MODE"] = self.config.get("injection_mode", "static")
            env["FEDSKILL_TAU2_MAX_SKILLS"] = str(self.config.get("max_skills", 10))
            env["FEDSKILL_TAU2_SKILL_LOG_PATH"] = str(selection_log)
            for config_key, variable in (
                ("dynamic_top_k", "DYNAMIC_TOP_K"), ("retrieval_candidates", "RETRIEVAL_CANDIDATES"),
                ("librarian_max_rounds", "LIBRARIAN_MAX_ROUNDS"), ("skill_min_overlap", "MIN_SKILL_OVERLAP"),
            ):
                if config_key in self.config:
                    env[f"FEDSKILL_TAU2_{variable}"] = str(self.config[config_key])
        try:
            subprocess.run(
                cmd,
                cwd=str(self._tau2_root),
                env=env,
                check=True,
                text=True,
                timeout=int(self.config.get("timeout", 900)),
            )
        except Exception as exc:
            return SessionData(
                task_id=task.task_id,
                task_description=task.description,
                outcome="failure",
                metadata={"error": str(exc), "run_name": run_name},
            )

        results_path = self._simulations_dir / run_name / "results.json"
        sessions = self._parse_results_file(results_path)
        session = sessions[0] if sessions else SessionData(
            task_id=task.task_id,
            task_description=task.description,
            outcome="failure",
            metadata={"error": f"No simulation found in {results_path}", "run_name": run_name},
        )
        if candidates:
            used = set()
            if selection_log.exists():
                for line in selection_log.read_text().splitlines():
                    event = json.loads(line)
                    if str(event.get("task_id")) == task.task_id:
                        used.update(str(s.get("id") or s.get("name")) for s in event.get("selected_skills", [])
                                    if s.get("id") or s.get("name"))
            else:
                session.metadata["selection_log_status"] = "missing; usage unavailable"
            session.skills_used = sorted(used)
            session.metadata["selection_log_path"] = str(selection_log)
        self._session_history.append(session)
        return session

    def collect_sessions(self, limit: int = 100) -> list[SessionData]:
        """Collect sessions from configured tau2 result files."""
        sessions = list(self._session_history)
        seen = {s.session_id for s in sessions}

        for results_file in self._result_files():
            for session in self._parse_results_file(results_file):
                if session.session_id not in seen:
                    sessions.append(session)
                    seen.add(session.session_id)

        return sessions[-limit:]

    def inject_skills(self, skills: list[dict]) -> bool:
        """Set the candidate library used by subsequent run_task calls."""
        self._injected_skills = deepcopy(skills)
        return True

    def health_check(self) -> bool:
        return self._tau2_root.exists() and (self._src_dir / "tau2" / "cli.py").exists()

    # ------------------------------------------------------------------
    # Optional task loading helper
    # ------------------------------------------------------------------

    def get_task_specs(self, num_tasks: int | None = None) -> list[TaskSpec]:
        """Load tau2 tasks as FedSkill TaskSpec objects."""
        if str(self._src_dir) not in sys.path:
            sys.path.insert(0, str(self._src_dir))
        from tau2.runner.helpers import get_tasks

        tasks = get_tasks(
            self._domain,
            task_split_name=self._task_split_name,
            num_tasks=num_tasks,
        )
        return [
            TaskSpec(
                task_id=str(task.id),
                description=self._task_description(task.model_dump(mode="json")),
                metadata={"domain": self._domain, "native_task": task.model_dump(mode="json")},
            )
            for task in tasks
        ]

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _tau2_env(self) -> dict[str, str]:
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("FEDSKILL_TAU2_"):
                del env[key]
        env["PYTHONPATH"] = str(self._src_dir)
        env["TAU2_DATA_DIR"] = str(self._tau2_root / "data")
        if self._openai_base_url:
            env["OPENAI_API_BASE"] = self._openai_base_url
            env["OPENAI_BASE_URL"] = self._openai_base_url
        if self._openai_api_key:
            env["OPENAI_API_KEY"] = self._openai_api_key
        return env

    def _result_files(self) -> list[Path]:
        if self._results_file:
            return [Path(self._results_file)]
        if self._run_name:
            return [self._simulations_dir / self._run_name / "results.json"]
        if not self._simulations_dir.exists():
            return []
        return sorted(self._simulations_dir.glob("*/results.json"))

    def _parse_results_file(self, path: Path) -> list[SessionData]:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tasks_by_id = {str(t.get("id", "")): t for t in data.get("tasks", [])}
        info = data.get("info", {})
        sessions = []

        for sim in data.get("simulations", []):
            task_id = str(sim.get("task_id", ""))
            task = tasks_by_id.get(task_id, {})
            reward_info = sim.get("reward_info") or {}
            score = float(reward_info.get("reward") or 0.0)
            traces, tools = self._format_messages(sim.get("messages", []))

            sessions.append(SessionData(
                session_id=str(sim.get("id") or uuid.uuid4().hex[:12]),
                task_id=task_id,
                task_description=self._task_description(task),
                traces=traces,
                tools_called=tools,
                outcome="success" if score > 0.5 else "failure",
                score=score,
                num_turns=len(sim.get("messages", [])),
                metadata={
                    "benchmark": "tau2-bench",
                    "domain": info.get("environment_info", {}).get("domain_name") or self._domain,
                    "trial": sim.get("trial"),
                    "seed": sim.get("seed"),
                    "termination_reason": sim.get("termination_reason"),
                    "reward_info": reward_info,
                    "results_file": str(path),
                    "task": task,
                },
            ))

        return sessions

    def _task_description(self, task: dict[str, Any]) -> str:
        parts = []
        for key in ("description", "user_scenario", "ticket"):
            value = task.get(key)
            if value:
                parts.append(f"{key}: {value}")
        criteria = task.get("evaluation_criteria")
        if criteria:
            parts.append("evaluation_criteria: " + json.dumps(criteria, ensure_ascii=False))
        return "\n".join(parts) or str(task.get("id", ""))

    def _format_messages(self, messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        traces: list[str] = []
        tools: list[str] = []

        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content")
            if content:
                traces.append(f"Turn {i} {role}: {content}")

            for tool_call in message.get("tool_calls") or []:
                name = tool_call.get("name") or tool_call.get("function", {}).get("name", "unknown_tool")
                arguments = tool_call.get("arguments") or tool_call.get("function", {}).get("arguments", {})
                tools.append(name)
                traces.append(
                    f"Turn {i} {role} ToolCall: {name}({json.dumps(arguments, ensure_ascii=False)})"
                )

            if role == "tool" and content:
                traces.append(f"Turn {i} Observation: {content}")

        return traces, tools


# Auto-register under both names for convenience.
register_backend("tau2bench", Tau2BenchBackend)
register_backend("tau2", Tau2BenchBackend)
