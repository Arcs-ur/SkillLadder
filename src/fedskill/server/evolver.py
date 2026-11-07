"""Skill evolver — merges / generalises a cluster of similar skills into a higher-level skill."""

from __future__ import annotations

import json
import logging
import math
import random
from fedskill.models.skill import Skill
from fedskill.utils.llm import llm_call


logger = logging.getLogger(__name__)


class InvalidJudgeResponse(ValueError):
    """A model response cannot be used as a valid gate or quality score."""


def _json_object(raw: str) -> dict:
    if not isinstance(raw, str):
        raise InvalidJudgeResponse("Expected a JSON response string")
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise InvalidJudgeResponse("Incomplete fenced JSON response")
        text = "\n".join(lines[1:-1])
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise InvalidJudgeResponse("Invalid JSON response") from exc
    if not isinstance(data, dict):
        raise InvalidJudgeResponse("Expected a JSON object")
    return data


def _candidate_data(raw: str) -> dict:
    data = _json_object(raw)
    for name in ("name", "description", "when_to_use", "procedure"):
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise InvalidJudgeResponse(f"Candidate requires a non-empty {name}")
    examples = data.get("examples", [])
    if not isinstance(examples, list) or any(not isinstance(item, str) for item in examples):
        raise InvalidJudgeResponse("Candidate examples must be a string list")
    scripts = data.get("scripts", [])
    if not isinstance(scripts, list):
        raise InvalidJudgeResponse("Candidate scripts must be a list")
    for script in scripts:
        if not isinstance(script, dict) or any(
            not isinstance(script.get(name), str)
            for name in ("name", "language", "code", "description")
        ):
            raise InvalidJudgeResponse("Candidate script fields must be strings")
    return data


EVOLVE_SYSTEM = """You are a skill-evolution engine.
You receive a cluster of similar skills extracted from different but related tasks.
Your job is to *merge and generalise* them into ONE higher-level skill that:
1. Retains the common core strategy shared across the input skills.
2. Abstracts away task-specific details so the result is broadly applicable.
3. Preserves concrete procedural steps — do NOT over-abstract into vague advice.
4. Merges and generalises executable scripts from the input skills when present.

Return a single JSON object with keys:
- name (str)
- description (str)
- when_to_use (str)
- procedure (str)
- examples (list[str]): 1-2 generalised examples
- scripts (list[object]): merged/generalised executable scripts. Each script has:
    - name (str): filename
    - language (str): "python" or "bash"
    - code (str): generalised, parameterised code
    - description (str): one-line explanation
  If the input skills have no scripts, set scripts to [].
  When merging scripts: generalise hardcoded values into parameters, combine
  common patterns, remove task-specific logic, keep the reusable core.

Return ONLY the JSON object, no extra text."""

EVOLVE_USER = """Below are {n} related skills extracted from different tasks.
Please evolve them into ONE generalised higher-level skill.

### Input Skills
{skills_text}
"""

CONSISTENCY_SYSTEM = """You are a skill quality checker.
Given an evolved skill and the original skills it was derived from, verify:
1. The evolved skill preserves the CORE strategy from the originals (not hallucinated).
2. The evolved skill is more general (not just copying one original).
3. The procedure steps are concrete and actionable (not vague platitudes).

Return a JSON object:
{"consistent": true/false, "reason": "brief explanation"}
Only JSON, no extra text."""

SCORE_SYSTEM = """You are a skill evaluator assessing generalizability.
Given a skill and a set of task descriptions, evaluate the skill on EACH task across two dimensions:

1. relevance (0-10): How related is this skill's strategy to the task?
2. applicability (0-10): Would an agent actually USE this skill when doing this task?
   (A skill can be relevant but not applicable if the agent wouldn't need it in practice.)

Return a JSON object:
{
  "per_task": [
    {"relevance": int, "applicability": int},
    ...
  ]
}
One entry per task, in order. Return ONLY the JSON object."""

SCORE_USER = """### Skill
{skill_json}

### Tasks to evaluate on
{tasks_text}
"""

STEP_PRESERVATION_SYSTEM = """You are a skill quality auditor.
Given an evolved skill and the best parent skill it was derived from, check whether
the evolved skill RETAINS all critical procedural steps from the best parent.

Generalization (e.g. replacing specific paths with generic descriptions) is fine.
But DROPPING concrete, actionable steps entirely is NOT acceptable.

Return a JSON object:
{"preserved": true/false, "lost_steps": ["step description", ...]}
Only JSON, no extra text."""

# Default thresholds
MIN_EVOLUTION_SCORE = 1.0   # Minimum generalizability score to keep an evolved skill
MAX_EVOLVE_RETRIES = 2      # Retry evolution if quality check fails
APPLICABILITY_THRESHOLD = 3  # Minimum applicability to count as "covered"
SCORE_GAP_RATIO = 0.3        # Gate 0: demote members scoring < ratio * best to context-only


def evolve_skill_cluster(
    skills: list[Skill],
    task_descriptions: list[str] | None = None,
    min_score: float = MIN_EVOLUTION_SCORE,
    max_retries: int = MAX_EVOLVE_RETRIES,
    *,
    disable_score_gap: bool = False,
    disable_consistency: bool = False,
    disable_nondegradation: bool = False,
    disable_critical_step: bool = False,
    rng: random.Random | None = None,
    score_gap_ratio: float = SCORE_GAP_RATIO,
    synthesis_temperature: float = 0.5,
    gate_temperature: float = 0.2,
) -> Skill | None:
    """Merge a cluster of similar skills into one higher-level generalised skill.

    Includes quality gates (each can be individually disabled for ablation):
    0. Score-gap demotion: members scoring < 30% of the best are demoted to
       context-only (included in the prompt as supplementary but not as equal
       members), preventing low-quality skills from diluting high-quality ones.
    1. Consistency check: verify evolved skill is faithful to originals
    2. Score gate: evolved skill must score >= min_score
    3. Non-degradation: evolved skill must score >= max(parent scores)
    4. Critical-step preservation: verify that the evolved skill retains the
       key procedural steps of the best parent skill.
    5. Retry on failure

    Args:
        skills: A list of similar skills from different clients / tasks.
        task_descriptions: Optional task descriptions for scoring.
        min_score: Minimum score threshold (default: 1.0).
        max_retries: Number of retries if quality check fails.
        disable_score_gap: Skip gate 0 (score-gap demotion).
        disable_consistency: Skip gate 1 (consistency check).
        disable_nondegradation: Skip gate 3 (non-degradation).
        disable_critical_step: Skip gate 4 (critical-step preservation).

    Returns:
        A new evolved Skill, or None if evolution fails quality checks.
    """
    # Enforce permutation symmetry of synthesis (Privacy Analysis, Definition 1):
    # randomly shuffle cluster members before prompt construction so that the
    # output distribution is invariant to member ordering. This makes the
    # symmetric-synthesis assumption a protocol-level guarantee rather than a
    # property we hope the LLM exhibits.
    if len(skills) < 2:
        raise ValueError("Synthesis requires at least two parent skills")
    if max_retries < 0 or not 0 <= score_gap_ratio <= 1:
        raise ValueError("Invalid retry budget or score-gap ratio")
    skills = list(skills)
    (rng or random).shuffle(skills)

    # --- Gate 0: Score-gap demotion (Solution B) ---
    # If some members score far below the best, demote them to "context-only"
    # so the LLM treats the best skill as the primary source and the weak ones
    # as supplementary perspective only.
    parent_max_score = max((s.score for s in skills), default=0.0)
    if not disable_score_gap and parent_max_score > 0:
        threshold = score_gap_ratio * parent_max_score
        primary = [s for s in skills if s.score >= threshold]
        context_only = [s for s in skills if s.score < threshold]
    else:
        primary = skills
        context_only = []

    # Build prompt with asymmetric roles
    parts = []
    for s in primary:
        parts.append(f"#### Skill [PRIMARY, score={s.score:.2f}] from [{s.source_task or 'unknown'}]\n{s.to_json()}")
    for s in context_only:
        parts.append(f"#### Skill [CONTEXT-ONLY, score={s.score:.2f}] from [{s.source_task or 'unknown'}]\n"
                     f"(This skill scored much lower than the best. Use it for supplementary perspective only. "
                     f"Do NOT let it dilute the procedural detail of the primary skills.)\n{s.to_json()}")
    skills_text = "\n\n".join(parts)
    prompt = EVOLVE_USER.format(n=len(skills), skills_text=skills_text)

    max_level = max(s.level for s in skills)
    best_parent = max(skills, key=lambda s: s.score)

    for attempt in range(1 + max_retries):
        # Generate evolved skill
        raw = llm_call(prompt, system_prompt=EVOLVE_SYSTEM, temperature=synthesis_temperature)
        try:
            data = _candidate_data(raw)
        except InvalidJudgeResponse as exc:
            logger.warning("Synthesis attempt %d rejected: %s", attempt + 1, exc)
            continue

        evolved = Skill(
            name=data.get("name", "Evolved Skill"),
            description=data.get("description", ""),
            when_to_use=data.get("when_to_use", ""),
            procedure=data.get("procedure", ""),
            examples=data.get("examples", []),
            scripts=data.get("scripts", []),
            source_task=None,
            level=max_level + 1,
            parent_ids=[s.id for s in skills],
        )

        # Quality gate 1: Consistency check
        if not disable_consistency and not _check_consistency(
            evolved, skills, temperature=gate_temperature
        ):
            if attempt < max_retries:
                continue  # Retry
            else:
                return None  # All retries exhausted, don't force a bad evolution

        # Quality gate 2: Score evaluation
        if task_descriptions:
            try:
                evolved.score = evaluate_skill(
                    evolved, task_descriptions, temperature=gate_temperature
                )
            except InvalidJudgeResponse as exc:
                logger.warning("Score gate attempt %d rejected: %s", attempt + 1, exc)
                continue

            # Gate 2a: Minimum score
            if evolved.score < min_score and attempt < max_retries:
                continue  # Retry

            # Gate 2b: Non-degradation (evolved should be >= best parent)
            if not disable_nondegradation and evolved.score < parent_max_score and attempt < max_retries:
                continue  # Retry — evolved is worse than parents

            # Final attempt but still below thresholds → reject
            if attempt == max_retries and (evolved.score < min_score or (not disable_nondegradation and evolved.score < parent_max_score)):
                return None

        # Quality gate 4: Critical-step preservation (Solution C)
        # Verify evolved skill retains key steps from the best parent
        if not disable_critical_step and not _check_step_preservation(
            evolved, best_parent, temperature=gate_temperature
        ):
            if attempt < max_retries:
                continue  # Retry
            else:
                return None  # Critical steps lost, reject

        return evolved

    return None  # All retries exhausted without passing quality gates


def _check_consistency(
    evolved: Skill, originals: list[Skill], *, temperature: float = 0.2
) -> bool:
    """Verify evolved skill is consistent with its source skills."""
    originals_summary = "\n".join(
        f"- {s.name}: {s.description[:100]}" for s in originals
    )
    prompt = (
        f"### Evolved Skill\n{evolved.to_json()}\n\n"
        f"### Original Skills\n{originals_summary}"
    )

    raw = llm_call(prompt, system_prompt=CONSISTENCY_SYSTEM, temperature=temperature)
    try:
        result = _json_object(raw)
        if type(result.get("consistent")) is not bool:
            raise InvalidJudgeResponse("Consistency verdict must be a boolean")
        return result["consistent"]
    except InvalidJudgeResponse as exc:
        logger.warning("Consistency check rejected: %s", exc)
        return False


def _check_step_preservation(
    evolved: Skill, best_parent: Skill, *, temperature: float = 0.2
) -> bool:
    """Verify evolved skill retains critical procedural steps from the best parent.

    Gate 4 (Solution C): After synthesis, compare the evolved skill against the
    best-scoring parent to ensure that concrete procedural steps were not lost
    during generalization. Generalization (e.g. replacing specific paths with
    generic descriptions) is acceptable; dropping steps entirely is not.

    Returns:
        True if critical steps are preserved, False if key steps were lost.
    """
    if not best_parent.procedure:
        return True  # Nothing to compare against

    prompt = (
        f"### Evolved Skill\n{evolved.to_json()}\n\n"
        f"### Best Parent Skill (score={best_parent.score:.2f})\n{best_parent.to_json()}"
    )

    raw = llm_call(prompt, system_prompt=STEP_PRESERVATION_SYSTEM, temperature=temperature)
    try:
        result = _json_object(raw)
        if type(result.get("preserved")) is not bool:
            raise InvalidJudgeResponse("Step-preservation verdict must be a boolean")
        return result["preserved"]
    except InvalidJudgeResponse as exc:
        logger.warning("Step-preservation check rejected: %s", exc)
        return False


def evaluate_skill(
    skill: Skill, task_descriptions: list[str], *, temperature: float = 0.2
) -> float:
    """Evaluate a skill's generalizability across multiple task descriptions.

    Scores the skill on two dimensions per task:
    - relevance: How related is the skill's strategy to the task?
    - applicability: Would an agent actually use this skill for this task?

    Returns a generalizability score that combines coverage (fraction of tasks
    where the skill is applicable) with mean relevance of covered tasks.

    generalizability = coverage × mean_relevance_of_applicable_tasks

    This distinguishes "broadly applicable but moderately relevant" (high)
    from "narrowly applicable but highly relevant" (lower).
    """
    if not task_descriptions:
        return 0.0

    tasks_text = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(task_descriptions))
    prompt = SCORE_USER.format(skill_json=skill.to_json(), tasks_text=tasks_text)

    raw = llm_call(prompt, system_prompt=SCORE_SYSTEM, temperature=temperature)
    result = _json_object(raw)
    per_task = result.get("per_task")
    if not isinstance(per_task, list) or len(per_task) != len(task_descriptions):
        raise InvalidJudgeResponse("Score response must contain exactly one item per task")
    for item in per_task:
        if not isinstance(item, dict):
            raise InvalidJudgeResponse("Each task score must be an object")
        for name in ("relevance", "applicability"):
            value = item.get(name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 10
            ):
                raise InvalidJudgeResponse(f"{name} must be a finite number in [0, 10]")
    applicable = [
        item for item in per_task if item["applicability"] >= APPLICABILITY_THRESHOLD
    ]
    # Algebraically identical to coverage times mean covered-task relevance.
    return round(sum(item["relevance"] for item in applicable) / len(task_descriptions), 2)
