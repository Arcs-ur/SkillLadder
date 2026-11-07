from fedskill.models.skill import Skill
from fedskill.utils.privacy import prepare_for_sharing, scrub_skill


def _skill(**kwargs) -> Skill:
    defaults = {
        "name": "skill",
        "description": "description",
        "when_to_use": "when needed",
        "procedure": "do the task",
    }
    defaults.update(kwargs)
    return Skill(**defaults)


def test_level_zero_is_not_release_eligible() -> None:
    assert prepare_for_sharing(_skill(name="local", level=0)) is None


def test_release_scrubs_entities_and_lineage() -> None:
    skill = _skill(
        name="call https://private.example/path",
        description="token=abcdefghijklmnop",
        level=2,
        source_task="private-task",
        parent_ids=["parent-1"],
    )

    released = prepare_for_sharing(skill)

    assert released is not None
    assert "private.example" not in released.name
    assert "abcdefghijklmnop" not in released.description
    assert released.source_task is None
    assert released.parent_ids == []
    assert skill.source_task == "private-task"


def test_blocklist_is_case_insensitive() -> None:
    skill = _skill(name="Project Nightfall", description="NIGHTFALL", level=1)
    scrubbed = scrub_skill(skill, {"nightfall"})
    assert scrubbed.name == "Project [REDACTED]"
    assert scrubbed.description == "[REDACTED]"


def test_structured_identifiers_and_named_entities_are_scrubbed() -> None:
    skill = _skill(
        description="customer_id='abc123' for customer Jane Doe",
        procedure="Call 415-555-0199 after verification.",
        level=1,
    )
    scrubbed = scrub_skill(skill)
    assert scrubbed.description == "customer_id='[CUSTOMER_ID]' for customer [PERSON]"
    assert scrubbed.procedure == "Call [PHONE] after verification."


def test_external_entity_detector_is_pluggable() -> None:
    skill = _skill(description="internal codename", level=1)
    scrubbed = scrub_skill(
        skill,
        entity_detector=lambda text: [(9, 17, "PROJECT")] if text == "internal codename" else [],
    )
    assert scrubbed.description == "internal [PROJECT]"


def test_prepare_for_sharing_removes_runtime_provenance() -> None:
    """[defect-probing] 分享快照不能保留来源历史与内部标识。"""
    skill = _skill(
        id="internal-origin",
        level=1,
        source_task="private-task",
        parent_ids=["private-parent"],
        quality_history=[{"task_id": "private-task", "metadata": {"origin": "private-origin"}}],
        last_quality_update="private-timestamp",
        retirement_reason="private-source",
        inject_count=7,
        positive_count=4,
    )
    released = prepare_for_sharing(skill)
    assert released is not None
    payload = released.to_json(include_runtime=True)
    assert "private-" not in payload
    assert "internal-origin" not in payload
    assert released.quality_history == []
    assert released.inject_count == 0
    assert skill.id == "internal-origin"
    assert skill.quality_history[0]["task_id"] == "private-task"


def test_prepare_for_sharing_scrubs_all_public_text_fields() -> None:
    """[defect-probing] 文件名与分类同样属于共享内容。"""
    skill = _skill(
        level=2,
        category="team@private.example",
        scripts=[{
            "name": "person@private.example.py",
            "language": "python",
            "code": "pass",
            "description": "Reusable code",
            "source_backend": "private-backend",
        }],
    )
    released = prepare_for_sharing(skill)
    assert released is not None
    assert "private.example" not in released.category
    assert "private.example" not in released.scripts[0]["name"]
    assert "source_backend" not in released.scripts[0]


def test_middleware_maps_recipient_feedback_to_internal_skill(tmp_path) -> None:
    from unittest.mock import Mock

    from fedskill.backends.base import ExecutionBackend, SessionData
    from fedskill.middleware import FedSkillMiddleware

    middleware = FedSkillMiddleware([], output_dir=str(tmp_path))
    backend = Mock(spec=ExecutionBackend)
    backend.name = "receiver"
    backend.inject_skills.return_value = True
    middleware.backends = [backend]
    skill = _skill(level=1, id="internal-skill", source_task="private-task")

    assert middleware.distribute_skills([skill]) == {"receiver": True}
    payload = backend.inject_skills.call_args.args[0]
    release_id = payload[0]["id"]
    assert release_id != skill.id
    assert payload[0]["source_task"] is None
    assert middleware._release_map_path.exists()

    middleware.update_effectiveness({
        "receiver": [SessionData(task_id="next-task", score=1.0,
                                 skills_used=[release_id, release_id])],
    })
    assert skill.inject_count == 1
    assert skill.positive_count == 1
