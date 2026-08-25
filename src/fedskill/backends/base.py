"""Abstract base class for execution backends.

An execution backend represents an environment where agents run tasks and
produce session data (traces, tool calls, outcomes). FedSkill collects this
data, evolves skills, and pushes them back.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


def _ts_id() -> str:
    """Generate a timestamp-based task ID: YYYYMMDD-HHMMSS-xxxx."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


@dataclass
class TaskSpec:
    """Specification for a task to be executed by a backend."""

    task_id: str = field(default_factory=_ts_id)
    description: str = ""
    input_data: str = ""
    expected_criteria: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionData:
    """Standardised session data collected from any backend.

    This is the common exchange format between backends and the evolution engine.
    Different backends produce this by adapting their native session format.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    task_description: str = ""
    traces: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    skills_used: list[str] = field(default_factory=list)
    outcome: str = ""  # "success", "partial", "failure"
    score: float = 0.0  # 0-1 overall score
    num_turns: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionData":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ExecutionBackend(ABC):
    """Abstract execution backend.

    Subclasses implement how to:
    - Run tasks in a specific environment
    - Collect session data from completed runs
    - Inject evolved skills into agents
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def run_task(self, task: TaskSpec, skills: list[dict] | None = None) -> SessionData:
        """Execute a task and return session data.

        Args:
            task: Task specification.
            skills: Optional list of skill dicts to inject into the agent.

        Returns:
            SessionData from the execution.
        """

    @abstractmethod
    def collect_sessions(self, limit: int = 100) -> list[SessionData]:
        """Collect recent session data from the backend.

        This is for backends that accumulate sessions independently
        (e.g., a running agent service). For backends where FedSkill
        triggers execution, run_task already returns the data.

        Args:
            limit: Maximum number of sessions to collect.

        Returns:
            List of SessionData from recent executions.
        """

    @abstractmethod
    def inject_skills(self, skills: list[dict]) -> bool:
        """Push evolved skills to the backend's agent(s).

        Args:
            skills: List of skill dicts (Skill.to_dict() format).

        Returns:
            True if injection succeeded.
        """

    def health_check(self) -> bool:
        """Check if the backend is reachable and ready."""
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
