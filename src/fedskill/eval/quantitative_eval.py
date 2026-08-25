"""Quantitative skill evaluator — measures skills against test cases with ground truth.

Metrics computed:
- Task Success Rate: fraction of expected output criteria met per test case
- Skill Recall: fraction of ground-truth skills covered by the evolved skill set
- Skill Precision: fraction of evolved skills that match at least one ground-truth skill
- Step Coverage: fraction of expected agent steps addressed by skill procedures
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from fedskill.models.skill import Skill
from fedskill.utils.llm import llm_call


# ── Data classes for evaluation results ──────────────────────────────

@dataclass
class TestCaseResult:
    """Result of evaluating a skill set against one test case."""
    test_id: str
    task_description: str
    success_rate: float = 0.0          # fraction of expected_output criteria met
    skill_recall: float = 0.0          # fraction of ground-truth skills covered
    skill_precision: float = 0.0       # fraction of matched skills / retrieved skills
    step_coverage: float = 0.0         # fraction of expected steps covered
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    covered_steps: list[str] = field(default_factory=list)
    uncovered_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalReport:
    """Aggregate evaluation report across all test cases."""
    num_testcases: int = 0
    avg_success_rate: float = 0.0
    avg_skill_recall: float = 0.0
    avg_skill_precision: float = 0.0
    avg_step_coverage: float = 0.0
    per_testcase: list[TestCaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "num_testcases": self.num_testcases,
            "avg_success_rate": round(self.avg_success_rate, 4),
            "avg_skill_recall": round(self.avg_skill_recall, 4),
            "avg_skill_precision": round(self.avg_skill_precision, 4),
            "avg_step_coverage": round(self.avg_step_coverage, 4),
            "per_testcase": [tc.to_dict() for tc in self.per_testcase],
        }


# ── Prompts ──────────────────────────────────────────────────────────

MATCH_SYSTEM = """You are a skill evaluator.
Given a set of evolved skills and a test case, determine:

1. **skill_matches**: For each ground-truth skill tag, which evolved skill (if any) covers it?
   Return a mapping: {ground_truth_tag: matched_skill_name or null}

2. **step_coverage**: For each expected step, is it covered by the procedure of any evolved skill?
   Return a mapping: {step: true/false}

3. **task_success**: Given the skills available, assess whether each expected output criterion
   could be achieved. Return a mapping: {criterion: true/false}

Return a JSON object with keys: "skill_matches", "step_coverage", "task_success".
Only JSON, no extra text."""

MATCH_USER = """### Evolved Skills Available
{skills_json}

### Test Case
**Task**: {task_description}
**Input**: {test_input}

### Ground-Truth Skill Tags (with descriptions)
{ground_truth_text}

### Expected Steps
{steps_text}

### Expected Output Criteria
{output_criteria}

Evaluate how well the evolved skills cover this test case."""


# ── Core evaluation logic ────────────────────────────────────────────

def evaluate_testcase(
    skills: list[Skill],
    testcase: dict,
    skill_tag_descriptions: dict[str, str],
) -> TestCaseResult:
    """Evaluate a set of evolved skills against a single test case.

    Uses LLM to judge skill-to-ground-truth matching, step coverage, and
    task success — then computes quantitative metrics from the judgement.
    """
    skills_json = json.dumps(
        [{"name": s.name, "description": s.description,
          "when_to_use": s.when_to_use, "procedure": s.procedure}
         for s in skills],
        indent=2, ensure_ascii=False,
    )

    gt_tags = testcase["ground_truth_skills"]
    ground_truth_text = "\n".join(
        f"- {tag}: {skill_tag_descriptions.get(tag, tag)}" for tag in gt_tags
    )
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(testcase["expected_steps"]))
    output_criteria = "\n".join(
        f"- {k}: {v}" for k, v in testcase["expected_output"].items()
    )

    prompt = MATCH_USER.format(
        skills_json=skills_json,
        task_description=testcase["task_description"],
        test_input=testcase["test_input"],
        ground_truth_text=ground_truth_text,
        steps_text=steps_text,
        output_criteria=output_criteria,
    )

    raw = llm_call(prompt, system_prompt=MATCH_SYSTEM)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: can't parse — return zeros
        return TestCaseResult(
            test_id=testcase["id"],
            task_description=testcase["task_description"],
        )

    # --- Compute Skill Recall & Precision ---
    skill_matches = result.get("skill_matches", {})
    matched_tags = [tag for tag, match in skill_matches.items() if match]
    missing_tags = [tag for tag, match in skill_matches.items() if not match]

    skill_recall = len(matched_tags) / len(gt_tags) if gt_tags else 0.0

    # Precision: how many distinct evolved skills are actually useful (matched at least one tag)
    matched_skill_names = set(v for v in skill_matches.values() if v)
    skill_precision = len(matched_skill_names) / len(skills) if skills else 0.0

    # --- Compute Step Coverage ---
    step_cov = result.get("step_coverage", {})
    covered = [s for s, ok in step_cov.items() if ok]
    uncovered = [s for s, ok in step_cov.items() if not ok]
    step_coverage = len(covered) / len(testcase["expected_steps"]) if testcase["expected_steps"] else 0.0

    # --- Compute Task Success Rate ---
    task_success = result.get("task_success", {})
    success_count = sum(1 for v in task_success.values() if v)
    success_rate = success_count / len(testcase["expected_output"]) if testcase["expected_output"] else 0.0

    return TestCaseResult(
        test_id=testcase["id"],
        task_description=testcase["task_description"],
        success_rate=round(success_rate, 4),
        skill_recall=round(skill_recall, 4),
        skill_precision=round(skill_precision, 4),
        step_coverage=round(step_coverage, 4),
        matched_skills=list(matched_skill_names),
        missing_skills=missing_tags,
        covered_steps=covered,
        uncovered_steps=uncovered,
    )


def run_quantitative_eval(
    skills: list[Skill],
    testcases: list[dict],
    skill_tag_descriptions: dict[str, str],
) -> EvalReport:
    """Run quantitative evaluation across all test cases.

    Returns an EvalReport with per-testcase results and aggregated metrics.
    """
    results: list[TestCaseResult] = []

    for tc in testcases:
        tc_result = evaluate_testcase(skills, tc, skill_tag_descriptions)
        results.append(tc_result)

    n = len(results)
    report = EvalReport(
        num_testcases=n,
        avg_success_rate=round(sum(r.success_rate for r in results) / n, 4) if n else 0.0,
        avg_skill_recall=round(sum(r.skill_recall for r in results) / n, 4) if n else 0.0,
        avg_skill_precision=round(sum(r.skill_precision for r in results) / n, 4) if n else 0.0,
        avg_step_coverage=round(sum(r.step_coverage for r in results) / n, 4) if n else 0.0,
        per_testcase=results,
    )
    return report
