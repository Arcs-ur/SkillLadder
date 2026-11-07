import json

import pytest

from fedskill.models.skill import Skill
from fedskill.server import evolver


def _parents():
    return [
        Skill(name="check", description="Validate", when_to_use="Before writes",
              procedure="Read the record before updating it.", id="parent-a"),
        Skill(name="confirm", description="Confirm", when_to_use="Before writes",
              procedure="Check the policy before updating it.", id="parent-b"),
    ]


def _candidate():
    return json.dumps({
        "name": "validate-before-update",
        "description": "Validate state and policy before writing.",
        "when_to_use": "Before updating a record.",
        "procedure": "Read the record, check the policy, then update.",
        "examples": [],
        "scripts": [],
    })


@pytest.mark.parametrize("reply", ["not json", "{}", '{"consistent":"false"}',
                                  '{"consistent":false}', "[]"])
def test_evolve_skill_cluster_rejects_invalid_or_false_judgment(monkeypatch, reply):
    """[defect-probing] 异常裁判不能使候选通过。"""
    calls = []

    def llm_call(prompt, system_prompt="", **kwargs):
        calls.append(system_prompt)
        return _candidate() if system_prompt == evolver.EVOLVE_SYSTEM else reply

    monkeypatch.setattr(evolver, "llm_call", llm_call)
    result = evolver.evolve_skill_cluster(_parents(), disable_critical_step=True)

    assert result is None
    assert calls.count(evolver.EVOLVE_SYSTEM) == 3


@pytest.mark.parametrize("reply", ["not json", "{}", '{"preserved":"false"}',
                                  '{"preserved":false}', "[]"])
def test_evolve_skill_cluster_rejects_invalid_step_judgment(monkeypatch, reply):
    """[defect-probing] 关键步骤检查也必须得到合法通过结果。"""
    def llm_call(prompt, system_prompt="", **kwargs):
        return _candidate() if system_prompt == evolver.EVOLVE_SYSTEM else reply

    monkeypatch.setattr(evolver, "llm_call", llm_call)
    assert evolver.evolve_skill_cluster(_parents(), disable_consistency=True) is None


def test_evolve_skill_cluster_preserves_valid_candidate(monkeypatch):
    def llm_call(prompt, system_prompt="", **kwargs):
        if system_prompt == evolver.EVOLVE_SYSTEM:
            return _candidate()
        if system_prompt == evolver.CONSISTENCY_SYSTEM:
            return '{"consistent":true}'
        return '{"preserved":true}'

    monkeypatch.setattr(evolver, "llm_call", llm_call)
    parents = _parents()
    result = evolver.evolve_skill_cluster(parents)
    assert result is not None
    assert result.level == 1
    assert set(result.parent_ids) == {"parent-a", "parent-b"}
    assert all(parent.level == 0 for parent in parents)


def test_evaluate_skill_uses_complete_task_rubric(monkeypatch):
    response = {"per_task": [
        {"relevance": 8, "applicability": 3},
        {"relevance": 10, "applicability": 2},
    ]}
    monkeypatch.setattr(evolver, "llm_call", lambda *args, **kwargs: json.dumps(response))
    assert evolver.evaluate_skill(_parents()[0], ["task-a", "task-b"]) == 4.0


@pytest.mark.parametrize("response", [
    {"per_task": [{"relevance": 10, "applicability": 10}]},
    {"per_task": [{"relevance": True, "applicability": 3}] * 2},
    {"per_task": [{"relevance": 11, "applicability": 3}] * 2},
    {"per_task": [{"relevance": float("nan"), "applicability": 3}] * 2},
    {"avg": 10},
])
def test_evaluate_skill_requires_valid_complete_scores(monkeypatch, response):
    monkeypatch.setattr(evolver, "llm_call", lambda *args, **kwargs: json.dumps(response))
    with pytest.raises(evolver.InvalidJudgeResponse):
        evolver.evaluate_skill(_parents()[0], ["task-a", "task-b"])


def test_evolve_skill_cluster_rejects_invalid_synthesis(monkeypatch):
    calls = []

    def llm_call(*args, **kwargs):
        calls.append(kwargs["system_prompt"])
        return "not json"

    monkeypatch.setattr(evolver, "llm_call", llm_call)
    assert evolver.evolve_skill_cluster(_parents()) is None
    assert calls == [evolver.EVOLVE_SYSTEM] * 3
