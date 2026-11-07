"""Simulation-level validation and statistics; missing rewards stay missing."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean, stdev


def reward_value(sim):
    value = (sim.get("reward_info") or {}).get("reward")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def valid_simulation_coverage(sims, ids, trials):
    expected = Counter((str(task), trial) for task in ids for trial in range(trials))
    actual = Counter()
    seen = set()
    for sim in sims:
        if not isinstance(sim, dict):
            return False
        sim_id = sim.get("id")
        trial = sim.get("trial")
        if not sim_id or sim_id in seen or type(trial) is not int:
            return False
        if reward_value(sim) is None or sim.get("termination_reason") not in {
            "user_stop", "agent_stop", "max_steps", "timeout", "too_many_errors",
            "agent_error", "user_error", "infrastructure_error", "context_window_exceeded", "unexpected_error",
        }:
            return False
        seen.add(sim_id)
        actual[(str(sim.get("task_id")), trial)] += 1
    return actual == expected


def mean_se(values):
    values = [value for value in values if value is not None]
    return {
        "n": len(values), "mean": mean(values) if values else None,
        "se": stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None,
    }


def simulation_rows(data):
    return [
        {
            "id": sim.get("id"), "task_id": str(sim.get("task_id")),
            "trial": sim.get("trial"), "seed": sim.get("seed"),
            "reward": reward_value(sim), "status": sim.get("termination_reason"),
            "duration": sim.get("duration"), "messages": len(sim.get("messages") or []),
            "tool_calls": sum(len(m.get("tool_calls") or []) for m in sim.get("messages") or []),
        }
        for sim in data.get("simulations", [])
    ]
