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
