"""Runtime skill quality tracking and feedback updates.

This module gives FedSkill a second loop in addition to extraction/evolution:
skills are not only generalized across rounds, they also accumulate empirical
quality signals from later task executions.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from fedskill.models.skill import Skill


@dataclass
class SkillFeedbackEvent:
    """One outcome signal assigned to one or more selected skills."""

    skill_refs: list[str]
    score: float
    task_id: str = ""
    round_id: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SkillFeedbackEvent":
        return cls(
            skill_refs=[str(item) for item in data.get("skill_refs", []) if item is not None],
            score=float(data.get("score") or 0.0),
            task_id=str(data.get("task_id") or ""),
            round_id=str(data.get("round_id") or ""),
            source=str(data.get("source") or ""),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_feedback_events(
    skills: list[Skill],
    events: Iterable[SkillFeedbackEvent | Mapping[str, Any]],
    *,
    alpha: float = 0.35,
    retire_quality_threshold: float = 0.15,
    retire_min_injections: int = 5,
) -> dict[str, Any]:
    """Apply feedback events to skills and return a compact update summary."""
    before = {skill.id: _quality_metric(skill) for skill in skills}
    by_ref = _index_skills(skills)
    missing_refs: Counter[str] = Counter()
    touched_ids: set[str] = set()
    applied_events = 0

    materialized_events = [
        event if isinstance(event, SkillFeedbackEvent) else SkillFeedbackEvent.from_mapping(event)
        for event in events
    ]

    for event in materialized_events:
        unique_refs = {ref for ref in event.skill_refs if ref}
        for ref in unique_refs:
            skill = by_ref.get(ref)
            if skill is None:
                missing_refs[ref] += 1
                continue
            skill.record_quality_event(
                event.score,
                round_id=event.round_id,
                source=event.source,
                task_id=event.task_id,
                alpha=alpha,
                metadata=event.metadata,
            )
            touched_ids.add(skill.id)
            applied_events += 1

    retired = []
    for skill in skills:
        if skill.retired:
            continue
        if skill.inject_count < retire_min_injections:
            continue
        if skill.quality_score <= retire_quality_threshold and skill.effectiveness <= retire_quality_threshold:
            reason = (
                f"quality_score={skill.quality_score:.3f}, "
                f"effectiveness={skill.effectiveness:.3f}, injections={skill.inject_count}"
            )
            skill.retire(reason)
            retired.append({"id": skill.id, "name": skill.name, "reason": reason})

    after = {skill.id: _quality_metric(skill) for skill in skills}
    touched = [skill for skill in skills if skill.id in touched_ids]
    deltas = [
        {
            "id": skill.id,
            "name": skill.name,
            "level": skill.level,
            "before": round(before.get(skill.id, 0.0), 4),
            "after": round(after.get(skill.id, 0.0), 4),
            "delta": round(after.get(skill.id, 0.0) - before.get(skill.id, 0.0), 4),
            "inject_count": skill.inject_count,
            "effectiveness": round(skill.effectiveness, 4),
            "retired": skill.retired,
        }
        for skill in touched
    ]
    deltas.sort(key=lambda item: item["delta"], reverse=True)

    active_quality = [
        {
            "id": skill.id,
            "name": skill.name,
            "level": skill.level,
            "quality": round(_quality_metric(skill), 4),
            "inject_count": skill.inject_count,
            "effectiveness": round(skill.effectiveness, 4),
        }
        for skill in skills
        if not skill.retired
    ]
    active_quality.sort(key=lambda item: item["quality"], reverse=True)
    quality_values = [item["quality"] for item in active_quality]
    return {
        "feedback_events": len(materialized_events),
        "applied_skill_events": applied_events,
        "touched_skills": len(touched_ids),
        "missing_skill_refs": dict(missing_refs.most_common(20)),
        "retired_skills": retired,
        "avg_active_quality": round(sum(quality_values) / len(quality_values), 4) if quality_values else 0.0,
        "top_quality_skills": active_quality[:10],
        "low_quality_skills": active_quality[-10:],
        "top_improved": deltas[:10],
        "top_declined": sorted(deltas, key=lambda item: item["delta"])[:10],
    }


def feedback_events_from_tau2_dynamic_log(
    results_path: str | Path,
    consultation_log_path: str | Path,
    *,
    source: str = "tau2_dynamic_librarian",
    round_id: str = "",
) -> list[SkillFeedbackEvent]:
    """Build per-task skill feedback events from tau2 dynamic librarian logs."""
    task_scores = _tau2_scores_by_task(results_path)
    usage_by_task: dict[str, set[str]] = defaultdict(set)
    path = Path(consultation_log_path)
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task_id = str(item.get("task_id") or "")
            if not task_id:
                continue
            for skill in item.get("selected_skills") or []:
                skill_ref = _skill_ref(skill)
                if skill_ref:
                    usage_by_task[task_id].add(skill_ref)

    events: list[SkillFeedbackEvent] = []
    for task_id, refs in usage_by_task.items():
        if task_id not in task_scores or not refs:
            continue
        events.append(
            SkillFeedbackEvent(
                skill_refs=sorted(refs),
                score=task_scores[task_id]["score"],
                task_id=task_id,
                round_id=round_id,
                source=source,
                metadata={"termination_reason": task_scores[task_id].get("termination_reason")},
            )
        )
    return events


def feedback_events_from_tau2_static_selection(
    results_path: str | Path,
    skills: list[Skill],
    *,
    max_skills: int,
    source: str = "tau2_static_reinject",
    round_id: str = "",
) -> list[SkillFeedbackEvent]:
    """Approximate static skill use by reproducing tau2's pre-task selector."""
    data = _read_json(results_path)
    if not data:
        return []
    tasks_by_id = {str(task.get("id", "")): task for task in data.get("tasks", []) if isinstance(task, dict)}
    task_scores = _tau2_scores_from_data(data)
    events = []

    for task_id, score_info in task_scores.items():
        task = tasks_by_id.get(task_id) or {"id": task_id}
        refs = [skill.id for skill in select_static_skills_for_task(skills, task, max_skills=max_skills)]
        if not refs:
            continue
        events.append(
            SkillFeedbackEvent(
                skill_refs=refs,
                score=score_info["score"],
                task_id=task_id,
                round_id=round_id,
                source=source,
                metadata={"termination_reason": score_info.get("termination_reason")},
            )
        )
    return events


def select_static_skills_for_task(skills: list[Skill], task: Mapping[str, Any], *, max_skills: int) -> list[Skill]:
    """Replicate tau2's static FedSkill selector using task/skill token overlap."""
    task_tokens = _tokens(json.dumps(task, ensure_ascii=False))
    scored: list[tuple[float, Skill]] = []
    seen_refs: set[str] = set()
    for skill in skills:
        if skill.retired:
            continue
        ref = skill.id or skill.name
        if ref in seen_refs:
            continue
        seen_refs.add(ref)
        skill_tokens = _tokens(_skill_text(skill))
        overlap = len(task_tokens & skill_tokens)
        quality = _quality_metric(skill)
        level_bonus = float(skill.level or 0) * 0.1
        scored.append((overlap + quality + level_bonus, skill))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [skill for _, skill in scored[:max_skills]]


def load_skills(path: str | Path) -> list[Skill]:
    data = _read_json(path)
    if isinstance(data, dict):
        data = data.get("skills", [])
    if not isinstance(data, list):
        return []
    return [Skill.from_dict(item) for item in data if isinstance(item, dict)]


def save_skills(path: str | Path, skills: list[Skill]) -> None:
    _atomic_write_json(path, [skill.to_dict() for skill in skills])


def save_quality_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    _atomic_write_json(path, dict(summary))


def _index_skills(skills: list[Skill]) -> dict[str, Skill]:
    by_ref: dict[str, Skill] = {}
    for skill in skills:
        for ref in {skill.id, skill.name}:
            if ref:
                by_ref[str(ref)] = skill
    return by_ref


def _skill_ref(skill: Mapping[str, Any]) -> str:
    return str(skill.get("id") or skill.get("name") or "")


def _quality_metric(skill: Skill) -> float:
    if skill.quality_score > 0.0:
        return max(0.0, min(1.0, skill.quality_score))
    if skill.effectiveness > 0.0:
        return max(0.0, min(1.0, skill.effectiveness))
    if skill.score > 1.0:
        return max(0.0, min(1.0, skill.score / 10.0))
    return max(0.0, min(1.0, skill.score))


def _skill_text(skill: Skill) -> str:
    return " ".join(
        str(value)
        for value in [skill.name, skill.description, skill.when_to_use, skill.procedure, skill.category]
        if value
    )


def _tokens(text: str) -> set[str]:
    import re

    return {token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())}


def _tau2_scores_by_task(path: str | Path) -> dict[str, dict[str, Any]]:
    data = _read_json(path)
    return _tau2_scores_from_data(data or {})


def _tau2_scores_from_data(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for sim in data.get("simulations", []) or []:
        if not isinstance(sim, dict):
            continue
        task_id = str(sim.get("task_id") or "")
        if not task_id:
            continue
        reward_info = sim.get("reward_info") or {}
        scores[task_id] = {
            "score": float(reward_info.get("reward") or 0.0),
            "termination_reason": sim.get("termination_reason"),
        }
    return scores


def _read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass