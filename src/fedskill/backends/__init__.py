"""Pluggable execution backends for FedSkill middleware.

Built-in backend types:
    local_sim      — LLM-simulated agent (no real execution)
    local_exec     — Real ReAct agent with sandboxed local tool execution
    docker_bench   — Generic Docker container execution
    http           — Remote agent service over REST API
    wildclawbench  — WildClawBench + SkillClaw executor (Docker + OpenClaw cluster)
"""

from fedskill.backends.base import ExecutionBackend, SessionData, TaskSpec
from fedskill.backends.registry import get_backend, register_backend

__all__ = [
    "ExecutionBackend",
    "SessionData",
    "TaskSpec",
    "get_backend",
    "register_backend",
]
