"""Replay evaluation — runs baseline vs skill-augmented agent and compares results.

This evaluator actually "replays" tasks through an agent, comparing:
1. Baseline: agent solves tasks WITHOUT any learned skills
2. Skill-augmented: agent solves the SAME tasks WITH evolved skills injected

Metrics:
- Task Success Rate: fraction of output criteria met (per mode)
- Skill Gain: success_augmented - success_baseline
- Step Efficiency: ratio of steps (fewer = more efficient)
- Completion Rate: fraction of tasks the agent reports as completed
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from fedskill.models.skill import Skill
from fedskill.eval.agent_runner import run_agent, AgentResult


# ── Result data classes ──────────────────────────────────────────────

@dataclass
class TestCaseComparison:
    """Side-by-side comparison of baseline vs augmented for one test case."""
    test_id: str
    task_description: str

    # Baseline metrics
    baseline_success_rate: float = 0.0
    baseline_num_steps: int = 0
    baseline_completed: bool = False
    baseline_criteria: dict[str, bool] = field(default_factory=dict)

    # Augmented metrics
    augmented_success_rate: float = 0.0
    augmented_num_steps: int = 0
    augmented_completed: bool = False
    augmented_criteria: dict[str, bool] = field(default_factory=dict)

    # Deltas
    skill_gain: float = 0.0           # augmented_success - baseline_success
    step_efficiency: float = 0.0      # 1 - (augmented_steps / baseline_steps); >0 = fewer steps

    # Criteria-level diff
    criteria_improved: list[str] = field(default_factory=list)   # failed→passed
    criteria_regressed: list[str] = field(default_factory=list)  # passed→failed

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReplayEvalReport:
    """Aggregate report comparing baseline vs skill-augmented runs."""
    num_testcases: int = 0

    # Baseline aggregate
    avg_baseline_success: float = 0.0
    baseline_completion_rate: float = 0.0
    avg_baseline_steps: float = 0.0

    # Augmented aggregate
    avg_augmented_success: float = 0.0
    augmented_completion_rate: float = 0.0
    avg_augmented_steps: float = 0.0

    # Deltas
    avg_skill_gain: float = 0.0
    avg_step_efficiency: float = 0.0

    per_testcase: list[TestCaseComparison] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "num_testcases": self.num_testcases,
            "baseline": {
                "avg_success_rate": round(self.avg_baseline_success, 4),
                "completion_rate": round(self.baseline_completion_rate, 4),
                "avg_steps": round(self.avg_baseline_steps, 2),
            },
            "augmented": {
                "avg_success_rate": round(self.avg_augmented_success, 4),
                "completion_rate": round(self.augmented_completion_rate, 4),
                "avg_steps": round(self.avg_augmented_steps, 2),
            },
            "skill_gain": round(self.avg_skill_gain, 4),
            "step_efficiency": round(self.avg_step_efficiency, 4),
            "per_testcase": [tc.to_dict() for tc in self.per_testcase],
        }


# ── Core evaluation logic ───────────────────────────────────────────

def _compute_success_rate(criteria: dict[str, bool]) -> float:
    """Fraction of criteria that are True."""
    if not criteria:
        return 0.0
    return sum(1 for v in criteria.values() if v) / len(criteria)


def evaluate_single_testcase(
    testcase: dict,
    skills: list[Skill],
    progress_callback=None,
) -> TestCaseComparison:
    """Run baseline and augmented agent on one test case, compare results.

    Args:
        testcase: A test case dict from eval_testcases.py.
        skills: Evolved skills to inject in the augmented run.
        progress_callback: Optional callable(message: str) for progress updates.
    """
    tc_id = testcase["id"]
    task_desc = testcase["task_description"]
    test_input = testcase["test_input"]
    expected_output = testcase["expected_output"]

    # ── Run baseline (no skills) ─────────────────────────────────
    if progress_callback:
        progress_callback(f"  [{tc_id}] Running baseline (no skills) ...")
    baseline: AgentResult = run_agent(
        task_id=tc_id,
        task_description=task_desc,
        test_input=test_input,
        output_criteria=expected_output,
        skills=None,
        mode="baseline",
    )

    # ── Run augmented (with skills) ──────────────────────────────
    if progress_callback:
        progress_callback(f"  [{tc_id}] Running skill-augmented ...")
    augmented: AgentResult = run_agent(
        task_id=tc_id,
        task_description=task_desc,
        test_input=test_input,
        output_criteria=expected_output,
        skills=skills,
        mode="skill_augmented",
    )

    # ── Compute metrics ──────────────────────────────────────────
    baseline_sr = _compute_success_rate(baseline.output_criteria)
    augmented_sr = _compute_success_rate(augmented.output_criteria)

    skill_gain = augmented_sr - baseline_sr

    # Step efficiency: positive means fewer steps with skills
    if baseline.num_steps > 0 and augmented.num_steps > 0:
        step_eff = 1.0 - (augmented.num_steps / baseline.num_steps)
    else:
        step_eff = 0.0

    # Find criteria that improved or regressed
    improved = []
    regressed = []
    for k in expected_output:
        b_val = baseline.output_criteria.get(k, False)
        a_val = augmented.output_criteria.get(k, False)
        if not b_val and a_val:
            improved.append(k)
        elif b_val and not a_val:
            regressed.append(k)

    return TestCaseComparison(
        test_id=tc_id,
        task_description=task_desc,
        baseline_success_rate=round(baseline_sr, 4),
        baseline_num_steps=baseline.num_steps,
        baseline_completed=baseline.completed,
        baseline_criteria=baseline.output_criteria,
        augmented_success_rate=round(augmented_sr, 4),
        augmented_num_steps=augmented.num_steps,
        augmented_completed=augmented.completed,
        augmented_criteria=augmented.output_criteria,
        skill_gain=round(skill_gain, 4),
        step_efficiency=round(step_eff, 4),
        criteria_improved=improved,
        criteria_regressed=regressed,
    )


def run_replay_eval(
    skills: list[Skill],
    testcases: list[dict],
    progress_callback=None,
) -> ReplayEvalReport:
    """Run the full replay evaluation: baseline vs skill-augmented across all test cases.

    Args:
        skills: Evolved skills to inject in augmented runs.
        testcases: List of test case dicts (from eval_testcases.py).
        progress_callback: Optional callable(message: str) for progress updates.

    Returns:
        ReplayEvalReport with per-testcase comparisons and aggregate metrics.
    """
    comparisons: list[TestCaseComparison] = []

    for tc in testcases:
        comp = evaluate_single_testcase(tc, skills, progress_callback)
        comparisons.append(comp)

    n = len(comparisons)
    if n == 0:
        return ReplayEvalReport()

    report = ReplayEvalReport(
        num_testcases=n,
        avg_baseline_success=round(sum(c.baseline_success_rate for c in comparisons) / n, 4),
        baseline_completion_rate=round(sum(1 for c in comparisons if c.baseline_completed) / n, 4),
        avg_baseline_steps=round(sum(c.baseline_num_steps for c in comparisons) / n, 2),
        avg_augmented_success=round(sum(c.augmented_success_rate for c in comparisons) / n, 4),
        augmented_completion_rate=round(sum(1 for c in comparisons if c.augmented_completed) / n, 4),
        avg_augmented_steps=round(sum(c.augmented_num_steps for c in comparisons) / n, 2),
        avg_skill_gain=round(sum(c.skill_gain for c in comparisons) / n, 4),
        avg_step_efficiency=round(sum(c.step_efficiency for c in comparisons) / n, 4),
        per_testcase=comparisons,
    )
    return report
