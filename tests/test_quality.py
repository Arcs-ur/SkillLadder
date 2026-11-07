import json

import pytest

from fedskill.models.skill import Skill
from fedskill.server.quality import (
    AmbiguousFeedbackError, SkillFeedbackEvent, apply_feedback_events,
    feedback_events_from_tau2_dynamic_log, feedback_events_from_tau2_static_selection,
    select_static_skills_for_task,
)
from fedskill.utils.run_manifest import atomic_write_json


def skill(**kwargs):
    return Skill("verify", "description", "condition", "procedure", **kwargs)


def test_reward_ema_is_independent_from_mixed_quality():
    s = skill(score=5)
    s.record_quality_event(0)
    s.record_quality_event(1)
    assert s.reward_ema == pytest.approx(0.56125)
    assert s.quality_score == 0.5429
    assert s.effectiveness == 0.5
    assert s.score == 5
    assert Skill.from_dict(s.to_dict()).reward_ema == s.reward_ema


def test_judge_scale_is_always_zero_to_ten_and_zero_ema_is_preserved():
    s = skill(score=1)
    s.record_quality_event(0, alpha=0)
    assert s.reward_ema == 0.1
    s.record_quality_event(0, alpha=1)
    s.record_quality_event(1, alpha=0)
    assert s.reward_ema == 0
    assert s.quality_score == 0.1


def test_complete_history_replays_but_unknown_alpha_does_not():
    s = skill(score=5)
    s.record_quality_event(0)
    s.record_quality_event(1)
    data = s.to_dict()
    del data["reward_ema"]
    restored = Skill.from_dict(data)
    assert restored.restore_reward_ema()
    assert restored.reward_ema == s.reward_ema
    for event in data["quality_history"]:
        event.pop("alpha")
    legacy = Skill.from_dict(data)
    assert not legacy.restore_reward_ema()
    assert legacy.quality_state_status == "legacy-incomplete"
    assert legacy.restore_reward_ema(legacy_alpha=0.35)
    assert legacy.quality_score == 0.5429


def test_incomplete_history_blocks_updates_without_mutating_counters():
    s = skill(score=5, inject_count=3, quality_score=0.4)
    with pytest.raises(ValueError, match="cannot be migrated"):
        s.record_quality_event(1)
    assert s.inject_count == 3
    assert s.reward_ema is None
    summary = apply_feedback_events([s], [SkillFeedbackEvent([s.id], 1)])
    assert summary["unmigratable_skill_ids"] == [s.id]
    assert summary["applied_skill_events"] == 0


def test_feedback_aliases_do_not_double_count():
    s = skill(id="a", score=5)
    result = apply_feedback_events([s], [SkillFeedbackEvent(["a", "verify"], 1)])
    assert result["applied_skill_events"] == s.inject_count == 1


def test_short_static_lists_filter_retired_and_duplicate_skills():
    active = skill(id="a")
    retired = skill(id="b", retired=True)
    assert select_static_skills_for_task([retired], {}, max_skills=10) == []
    assert select_static_skills_for_task([active, active, retired], {}, max_skills=10) == [active]


def test_multi_trial_usage_joins_actual_simulation_and_rejects_ambiguous_logs(tmp_path):
    results = tmp_path / "results.json"
    log = tmp_path / "usage.jsonl"
    atomic_write_json(results, {"simulations": [
        dict(id="a", task_id="t", trial=0, reward_info={"reward": 0}),
        dict(id="b", task_id="t", trial=1, reward_info={"reward": 1}),
    ]})
    log.write_text("\n".join(json.dumps(dict(task_id="t", simulation_id=identity,
                                           selected_skills=[{"id": "s"}])) for identity in ("a", "b")))
    events = feedback_events_from_tau2_dynamic_log(results, log)
    assert [e.score for e in events] == [0, 1]
    assert [e.metadata["trial"] for e in events] == [0, 1]
    log.write_text(json.dumps(dict(task_id="t", selected_skills=[{"id": "s"}])))
    with pytest.raises(AmbiguousFeedbackError, match="multiple trials"):
        feedback_events_from_tau2_dynamic_log(results, log)


def test_static_feedback_uses_real_logs(tmp_path):
    results = tmp_path / "results.json"
    log = tmp_path / "usage.jsonl"
    atomic_write_json(results, {"simulations": [
        dict(id="a", task_id="t", trial=0, reward_info={"reward": 0}),
    ]})
    log.write_text(json.dumps(dict(task_id="t", selected_skills=[{"id": "selected"}])))
    events = feedback_events_from_tau2_static_selection(
        results, [skill(id="other")], max_skills=10, selection_log_path=log,
    )
    assert events[0].skill_refs == ["selected"]
    assert events[0].score == 0
