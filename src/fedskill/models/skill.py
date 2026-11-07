"""Skill data model — the core unit in FedSkill."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class Skill:
    """Represents a reusable skill extracted or evolved from task traces.

    A skill is a composite structure containing both natural-language guidance
    and optional executable scripts, following the community convention
    established by Voyager and SkillClaw.

    Attributes:
        name: Short, descriptive skill name.
        description: Natural-language description of what the skill does.
        when_to_use: Conditions under which this skill is applicable.
        procedure: Step-by-step procedure or template (natural language).
        examples: Concrete input/output examples demonstrating the skill.
        scripts: Optional executable scripts extracted from traces.
        category: Semantic category discovered during clustering (e.g., "error-recovery").
        source_task: Which task / client this skill was extracted from (None for evolved).
        level: 0 = local (extracted), 1+ = evolved (higher = more general).
        parent_ids: IDs of skills that were merged/evolved to produce this one.
        score: LLM-judged quality/utility score (0-10).
        inject_count: Number of times this skill has been injected into agent prompts.
        positive_count: Injections where agent outcome was positive.
        negative_count: Injections where agent outcome was negative.
        effectiveness: Observed effectiveness = positive_count / inject_count.
        version: Monotonic skill revision version.
        quality_score: Runtime quality estimate in [0, 1], updated from task feedback.
        quality_history: Per-round/per-task quality update events.
        retired: Whether this skill should be ignored by retrieval/injection.
    """

    name: str
    description: str
    when_to_use: str
    procedure: str
    examples: list[str] = field(default_factory=list)
    scripts: list[dict] = field(default_factory=list)
    category: str = ""
    source_task: Optional[str] = None
    level: int = 0
    parent_ids: list[str] = field(default_factory=list)
    score: float = 0.0
    inject_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    effectiveness: float = 0.0
    version: int = 1
    quality_score: float = 0.0
    quality_history: list[dict] = field(default_factory=list)
    last_quality_update: str = ""
    retired: bool = False
    retirement_reason: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    source_session_id: Optional[str] = None
    reward_ema: Optional[float] = None
    quality_state_status: str = "uninitialized"

    # ------------------------------------------------------------------
    # Feedback helpers
    # ------------------------------------------------------------------

    def record_injection(self) -> None:
        """Record that this skill was injected into an agent prompt."""
        self.inject_count += 1

    def record_feedback(self, outcome_score: float) -> None:
        """Record feedback from an agent session where this skill was injected.

        Args:
            outcome_score: Task outcome score (0-1). > 0.5 counts as positive.
        """
        if outcome_score > 0.5:
            self.positive_count += 1
        elif outcome_score < 0.2:
            self.negative_count += 1
        if self.inject_count > 0:
            self.effectiveness = self.positive_count / self.inject_count

    def record_quality_event(
        self,
        outcome_score: float,
        *,
        round_id: str = "",
        source: str = "",
        task_id: str = "",
        alpha: float = 0.35,
        metadata: dict | None = None,
    ) -> None:
        """Update this skill's runtime quality from one observed task outcome.

        ``score`` remains the LLM/generalizability score on a 0-10 scale.
        ``quality_score`` is an empirical 0-1 moving estimate driven by repeated
        downstream use, so future retrieval/evolution can prefer skills that
        are actually helping over time.
        """
        if not math.isfinite(outcome_score) or not math.isfinite(alpha):
            raise ValueError("Outcome and EMA alpha must be finite.")
        if self.reward_ema is None:
            self.restore_reward_ema()
        if self.reward_ema is None:
            raise ValueError(
                "Legacy runtime quality cannot be migrated exactly; provide complete history "
                "and explicit legacy_alpha, or start a new quality run."
            )
        observed = _clamp01(outcome_score)
        previous_quality = self.quality_score
        previous_ema = self.reward_ema

        self.record_injection()
        self.record_feedback(observed)

        alpha = _clamp01(alpha)
        self.reward_ema = (1.0 - alpha) * previous_ema + alpha * observed
        self.quality_score = round(0.7 * self.reward_ema + 0.3 * self.effectiveness, 4)
        self.quality_state_status = "observed"
        self.last_quality_update = _utc_now_iso()
        self.quality_history.append(
            {
                "timestamp": self.last_quality_update,
                "round_id": round_id,
                "source": source,
                "task_id": task_id,
                "outcome_score": observed,
                "previous_quality": round(previous_quality, 4),
                "previous_reward_ema": previous_ema,
                "reward_ema": self.reward_ema,
                "alpha": alpha,
                "quality_score": self.quality_score,
                "effectiveness": round(self.effectiveness, 4),
                "inject_count": self.inject_count,
                "positive_count": self.positive_count,
                "negative_count": self.negative_count,
                "metadata": metadata or {},
            }
        )

    def restore_reward_ema(self, *, legacy_alpha: float | None = None) -> bool:
        """Replay complete reward history. Unknown historical alpha is not guessed."""
        if self.reward_ema is not None:
            return True
        initial = _clamp01(self.score / 10.0)
        if self.inject_count == 0:
            self.reward_ema = initial
            self.quality_score = initial
            self.quality_state_status = "initialized"
            return True
        events = [e for e in self.quality_history if "outcome_score" in e]
        if len(events) != self.inject_count:
            self.quality_state_status = "legacy-incomplete"
            return False
        ema = initial
        for index, event in enumerate(events, 1):
            alpha = event.get("alpha", legacy_alpha)
            outcome = event.get("outcome_score")
            if (
                event.get("inject_count") != index
                or not isinstance(alpha, (int, float)) or isinstance(alpha, bool)
                or not math.isfinite(alpha) or not 0 <= alpha <= 1
                or not isinstance(outcome, (int, float)) or isinstance(outcome, bool)
                or not math.isfinite(outcome) or not 0 <= outcome <= 1
            ):
                self.quality_state_status = "legacy-incomplete"
                return False
            ema = (1 - alpha) * ema + alpha * outcome
        self.reward_ema = ema
        self.quality_score = round(0.7 * ema + 0.3 * self.effectiveness, 4)
        self.quality_state_status = "history-replayed"
        return True

    def retire(self, reason: str) -> None:
        """Mark this skill as inactive for future retrieval/injection."""
        self.retired = True
        self.retirement_reason = reason
        self.last_quality_update = _utc_now_iso()

    def bump_version(self, reason: str = "") -> None:
        """Record that the skill text was revised."""
        self.version += 1
        self.last_quality_update = _utc_now_iso()
        self.quality_history.append(
            {
                "timestamp": self.last_quality_update,
                "event": "version_bump",
                "version": self.version,
                "reason": reason,
            }
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2, include_runtime: bool = False) -> str:
        data = self.to_dict()
        if not include_runtime:
            data.pop("quality_history", None)
        return json.dumps(data, indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, text: str) -> "Skill":
        return cls.from_dict(json.loads(text))

    def summary(self) -> str:
        """One-line summary for display purposes."""
        tag = f"L{self.level}"
        src = f" [{self.source_task}]" if self.source_task else ""
        return f"[{tag}]{src} {self.name}: {self.description[:80]}"
