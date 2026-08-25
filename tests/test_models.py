from fedskill.models.skill import Skill
from skillladder import SkillLadderServer


def test_skill_round_trip() -> None:
    original = Skill(
        name="verify-before-write",
        description="Validate state before a mutating tool call.",
        when_to_use="Before irreversible actions.",
        procedure="Read, validate, then write.",
        level=2,
        parent_ids=["a", "b"],
    )

    restored = Skill.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()


def test_public_package_surface() -> None:
    server = SkillLadderServer(clustering_strategy="random")
    assert server.min_cluster_size == 2
