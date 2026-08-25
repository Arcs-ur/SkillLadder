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
from pathlib import Path
from typing import Any

from fedskill.backends.base import ExecutionBackend, SessionData, TaskSpec
from fedskill.backends.registry import register_backend


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
        """Run one tau2 task via the tau2 CLI and return its session.

        Skill injection for tau2 is implemented in a later phase; for now the
        optional skills are recorded in metadata and can be used by a custom
        tau2 agent once registered.
        """
        if skills:
            self._injected_skills = skills

        run_name = self.config.get("active_run_name") or f"fedskill_tau2_{self._domain}_{task.task_id}_{uuid.uuid4().hex[:6]}"
        cmd = [
            self._python_cmd,
            "-m",
            "tau2.cli",
            "run",
            "--domain",
            self._domain,
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
            "--auto-resume",
            "--save-to",
            run_name,
            "--log-level",
            self.config.get("log_level", "INFO"),
        ]

        env = self._tau2_env()
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
        if skills:
            session.skills_used = [s.get("name", s.get("id", "skill")) for s in skills]
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
        """Store skills for a future tau2 FedSkill agent implementation."""
        self._injected_skills = skills
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
        env["PYTHONPATH"] = str(self._src_dir)
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
