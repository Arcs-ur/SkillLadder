"""Privacy evaluation — compute Entity Retention (ER) and Source Attribution (SA) metrics.

Implements the two empirical privacy metrics defined in §3.7 of the paper:
  1. ER(ℓ): fraction of source-trace entities retained in level-ℓ skills
  2. SA(ℓ): LLM adversary accuracy in 5-choice source attribution game

Also fits the independent sieving model (Proposition 2):
  ER(ℓ) ≈ p_ext · p_evo^ℓ
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from fedskill.models.skill import Skill
from fedskill.utils.llm import llm_call

# ---------------------------------------------------------------------------
# Entity extraction (regex-based)
# ---------------------------------------------------------------------------

# Patterns for sensitive entities
_ENTITY_PATTERNS = {
    "file_path": re.compile(
        r'(?:/[\w.\-]+){2,}|(?:[A-Z]:\\[\w.\-\\]+)', re.IGNORECASE
    ),
    "url": re.compile(
        r'https?://[^\s\'"<>)\],]+', re.IGNORECASE
    ),
    "ip_address": re.compile(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ),
    "api_key": re.compile(
        r'(?:sk-|api[_-]?key|token|bearer)\s*[=:]\s*["\']?[\w\-]{20,}',
        re.IGNORECASE,
    ),
    "email": re.compile(
        r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b'
    ),
    "username": re.compile(
        r'(?:user(?:name)?|login)\s*[=:]\s*["\']?[\w.\-]+',
        re.IGNORECASE,
    ),
}

# Common false positives to filter out
_IGNORE_PATHS = {
    "/bin/bash", "/bin/sh", "/usr/bin/env", "/dev/null",
    "/tmp", "/etc/hosts", "/usr/local",
}


def extract_entities(text: str) -> set[str]:
    """Extract sensitive entities from text via regex patterns."""
    entities: set[str] = set()
    for _kind, pattern in _ENTITY_PATTERNS.items():
        for match in pattern.finditer(text):
            val = match.group(0).strip().rstrip(".,;:)")
            if val not in _IGNORE_PATHS and len(val) > 5:
                entities.add(val)
    return entities


# ---------------------------------------------------------------------------
# Entity Retention Rate (ER)
# ---------------------------------------------------------------------------

@dataclass
class ERResult:
    level: int
    num_skills: int
    mean_er: float
    per_skill: list[float]


def compute_entity_retention(
    skills: list[Skill],
    trace_texts: dict[str, str],
) -> dict[int, ERResult]:
    """Compute entity retention rate per evolution level.

    Args:
        skills: All skills with parent_ids tracing back to source tasks.
        trace_texts: Mapping from source_task identifier to trace text.

    Returns:
        Dict mapping level -> ERResult.
    """
    # Build lookup: skill_id -> skill
    skill_map = {s.id: s for s in skills}

    # For each skill, recursively find source traces
    def get_source_tasks(skill: Skill) -> set[str]:
        if skill.level == 0 and skill.source_task:
            return {skill.source_task}
        sources: set[str] = set()
        for pid in skill.parent_ids:
            parent = skill_map.get(pid)
            if parent:
                sources |= get_source_tasks(parent)
        return sources

    # Group by level
    by_level: dict[int, list[Skill]] = defaultdict(list)
    for s in skills:
        by_level[s.level].append(s)

    results: dict[int, ERResult] = {}
    for level in sorted(by_level):
        level_skills = by_level[level]
        per_skill_er: list[float] = []
        for s in level_skills:
            source_tasks = get_source_tasks(s)
            # Collect source entities
            source_ents: set[str] = set()
            for st in source_tasks:
                if st in trace_texts:
                    source_ents |= extract_entities(trace_texts[st])
            if not source_ents:
                continue
            # Skill entities
            skill_text = f"{s.name} {s.description} {s.when_to_use} {s.procedure} {' '.join(s.examples)}"
            for script in s.scripts:
                skill_text += f" {script.get('code', '')} {script.get('description', '')}"
            skill_ents = extract_entities(skill_text)
            # Compute retention
            retained = skill_ents & source_ents
            er = len(retained) / len(source_ents)
            per_skill_er.append(er)

        mean_er = sum(per_skill_er) / len(per_skill_er) if per_skill_er else 0.0
        results[level] = ERResult(
            level=level,
            num_skills=len(level_skills),
            mean_er=mean_er,
            per_skill=per_skill_er,
        )
    return results


# ---------------------------------------------------------------------------
# Source Attribution (SA) game
# ---------------------------------------------------------------------------

_SA_SYSTEM = """You are evaluating whether you can determine the source of a skill.
You will be given an evolved skill and {m} candidate execution traces.
Exactly one of these traces was used (directly or indirectly) to produce the skill.
Your task: identify which trace is the most likely source.

Return ONLY a JSON object: {{"choice": <1-indexed trace number>, "confidence": <0-1>}}
No extra text."""

_SA_USER = """### Evolved Skill (level {level})
{skill_text}

### Candidate Traces
{traces_text}

Which trace most likely contributed to this skill?"""


@dataclass
class SAResult:
    level: int
    num_trials: int
    accuracy: float
    mean_confidence: float
    per_trial: list[dict]


def compute_source_attribution(
    skills: list[Skill],
    trace_texts: dict[str, str],
    m: int = 5,
    max_trials_per_level: int = 30,
    seed: int = 42,
) -> dict[int, SAResult]:
    """Run the source attribution game for skills at each level.

    Args:
        skills: All skills with lineage information.
        trace_texts: Mapping from source_task id to trace text.
        m: Number of candidates in each lineup (1 true + m-1 distractors).
        max_trials_per_level: Max number of trials per level.
        seed: Random seed for distractor sampling.

    Returns:
        Dict mapping level -> SAResult.
    """
    rng = random.Random(seed)
    skill_map = {s.id: s for s in skills}
    all_task_ids = list(trace_texts.keys())

    def get_source_tasks(skill: Skill) -> set[str]:
        if skill.level == 0 and skill.source_task:
            return {skill.source_task}
        sources: set[str] = set()
        for pid in skill.parent_ids:
            parent = skill_map.get(pid)
            if parent:
                sources |= get_source_tasks(parent)
        return sources

    by_level: dict[int, list[Skill]] = defaultdict(list)
    for s in skills:
        by_level[s.level].append(s)

    results: dict[int, SAResult] = {}
    for level in sorted(by_level):
        level_skills = by_level[level]
        rng.shuffle(level_skills)
        trials: list[dict] = []

        for s in level_skills[:max_trials_per_level]:
            source_tasks = get_source_tasks(s)
            source_tasks_in_traces = [t for t in source_tasks if t in trace_texts]
            if not source_tasks_in_traces:
                continue

            # Pick one true source
            true_source = rng.choice(source_tasks_in_traces)

            # Pick m-1 distractors (not in source set)
            distractor_pool = [t for t in all_task_ids if t not in source_tasks]
            if len(distractor_pool) < m - 1:
                continue
            distractors = rng.sample(distractor_pool, m - 1)

            # Build lineup with random position for true source
            lineup = distractors[:]
            true_pos = rng.randint(0, len(lineup))
            lineup.insert(true_pos, true_source)

            # Build prompt
            skill_text = f"Name: {s.name}\nDescription: {s.description}\nWhen to use: {s.when_to_use}\nProcedure: {s.procedure}"
            traces_block = ""
            for i, tid in enumerate(lineup):
                # Truncate traces to ~500 tokens for prompt size
                trace = trace_texts[tid][:2000]
                traces_block += f"\n--- Trace {i+1} (ID: hidden) ---\n{trace}\n"

            prompt = _SA_USER.format(
                level=level, skill_text=skill_text, traces_text=traces_block
            )
            system = _SA_SYSTEM.format(m=m)

            try:
                raw = llm_call(prompt, system_prompt=system, temperature=0.0)
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                result = json.loads(raw)
                chosen = int(result.get("choice", 0)) - 1  # to 0-indexed
                confidence = float(result.get("confidence", 0))
                correct = chosen == true_pos
            except (json.JSONDecodeError, ValueError, KeyError):
                correct = False
                confidence = 0.0

            trials.append({
                "skill_id": s.id,
                "correct": correct,
                "confidence": confidence,
            })

        accuracy = sum(t["correct"] for t in trials) / len(trials) if trials else 0.0
        mean_conf = sum(t["confidence"] for t in trials) / len(trials) if trials else 0.0
        results[level] = SAResult(
            level=level,
            num_trials=len(trials),
            accuracy=accuracy,
            mean_confidence=mean_conf,
            per_trial=trials,
        )
    return results


# ---------------------------------------------------------------------------
# Sieving model fit: ER(ℓ) ≈ p_ext · p_evo^ℓ
# ---------------------------------------------------------------------------

def fit_sieving_model(
    er_results: dict[int, ERResult],
) -> dict[str, float]:
    """Fit the independent sieving model to observed ER data.

    Model: ER(ℓ) = p_ext · p_evo^ℓ
    Taking log: log ER(ℓ) = log p_ext + ℓ · log p_evo

    Returns dict with keys: p_ext, p_evo, r_squared.
    """
    points = [(level, r.mean_er) for level, r in er_results.items() if r.mean_er > 0]
    if len(points) < 2:
        return {"p_ext": 0.0, "p_evo": 0.0, "r_squared": 0.0}

    # Log-linear regression: y = a + b*x where y=log(ER), x=level
    xs = [p[0] for p in points]
    ys = [math.log(p[1]) for p in points]
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)

    denom = n * sum_x2 - sum_x**2
    if abs(denom) < 1e-12:
        return {"p_ext": 0.0, "p_evo": 0.0, "r_squared": 0.0}

    b = (n * sum_xy - sum_x * sum_y) / denom
    a = (sum_y - b * sum_x) / n

    p_ext = math.exp(a)
    p_evo = math.exp(b)

    # R² computation
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"p_ext": p_ext, "p_evo": p_evo, "r_squared": r_squared}


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def run_privacy_evaluation(
    skills: list[Skill],
    trace_texts: dict[str, str],
    run_sa: bool = True,
    output_path: str | None = None,
) -> dict:
    """Run full privacy evaluation and produce a report.

    Args:
        skills: All skills (all levels) with lineage.
        trace_texts: source_task -> trace text mapping.
        run_sa: Whether to run the (expensive) LLM-based SA game.
        output_path: If given, write JSON report to this path.

    Returns:
        Report dict with ER, SA, and model fit results.
    """
    print("=" * 60)
    print("FedSkill Privacy Evaluation")
    print("=" * 60)

    # --- ER ---
    print("\n[1/3] Computing entity retention rate (ER)...")
    er_results = compute_entity_retention(skills, trace_texts)
    for level, er in sorted(er_results.items()):
        print(f"  Level {level}: ER = {er.mean_er:.4f}  (n={er.num_skills} skills)")

    # --- Model fit ---
    print("\n[2/3] Fitting sieving model ER(ℓ) = p_ext · p_evo^ℓ ...")
    model = fit_sieving_model(er_results)
    print(f"  p_ext = {model['p_ext']:.4f}")
    print(f"  p_evo = {model['p_evo']:.4f}")
    print(f"  R²    = {model['r_squared']:.4f}")

    # --- SA ---
    sa_results_dict: dict = {}
    if run_sa:
        print("\n[3/3] Running source attribution game (SA)...")
        sa_results = compute_source_attribution(skills, trace_texts)
        for level, sa in sorted(sa_results.items()):
            print(
                f"  Level {level}: SA = {sa.accuracy:.3f}  "
                f"(n={sa.num_trials} trials, random={1/5:.3f})"
            )
        sa_results_dict = {
            level: {
                "accuracy": sa.accuracy,
                "num_trials": sa.num_trials,
                "mean_confidence": sa.mean_confidence,
            }
            for level, sa in sa_results.items()
        }
    else:
        print("\n[3/3] Skipping SA game (run_sa=False)")

    # --- Mixing degree (measured) ---
    skill_map = {s.id: s for s in skills}

    def get_source_count(s: Skill) -> int:
        if s.level == 0:
            return 1
        return sum(
            get_source_count(skill_map[pid])
            for pid in s.parent_ids
            if pid in skill_map
        )

    by_level: dict[int, list[int]] = defaultdict(list)
    for s in skills:
        by_level[s.level].append(get_source_count(s))

    mixing_report: dict[int, dict] = {}
    for level in sorted(by_level):
        counts = by_level[level]
        mixing_report[level] = {
            "min": min(counts),
            "mean": sum(counts) / len(counts),
            "max": max(counts),
        }

    # --- Report ---
    report = {
        "entity_retention": {
            level: {"mean_er": er.mean_er, "num_skills": er.num_skills}
            for level, er in er_results.items()
        },
        "sieving_model": model,
        "source_attribution": sa_results_dict,
        "mixing_degree": mixing_report,
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Levels evaluated: {sorted(er_results.keys())}")
    print(f"Sieving model: ER(ℓ) ≈ {model['p_ext']:.3f} × {model['p_evo']:.3f}^ℓ  (R²={model['r_squared']:.3f})")
    for level in sorted(by_level):
        md = mixing_report[level]
        er_val = er_results.get(level, ERResult(level, 0, 0.0, []))
        sa_val = sa_results_dict.get(level, {})
        print(
            f"  Level {level}: μ={md['mean']:.1f} (min={md['min']}), "
            f"ER={er_val.mean_er:.4f}, "
            f"SA={sa_val.get('accuracy', 'N/A')}"
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {output_path}")

    return report
