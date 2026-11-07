from fedskill.eval.privacy_eval import compute_source_mixing
from fedskill.models.skill import Skill


def _skill(identifier, *, source=None, session=None, parents=(), level=0):
    return Skill(
        name=identifier, description="d", when_to_use="t", procedure="p",
        id=identifier, source_task=source, source_session_id=session,
        parent_ids=list(parents), level=level,
    )


def test_source_mixing_counts_distinct_sources_with_shared_ancestors():
    skills = [
        _skill("a", source="one"),
        _skill("b", source="one"),
        _skill("c", source="two"),
        _skill("ab", level=1, parents=["a", "b"]),
        _skill("ac", level=1, parents=["a", "c"]),
        _skill("top", level=2, parents=["ab", "ac", "ab"]),
    ]
    mixing = compute_source_mixing(skills)
    assert mixing[1]["min"] == 1
    assert mixing[1]["max"] == 2
    assert mixing[2]["mean"] == 2


def test_source_mixing_distinguishes_sessions_of_the_same_task():
    skills = [
        _skill("a", source="task", session="session-a"),
        _skill("b", source="task", session="session-b"),
        _skill("ab", level=1, parents=["a", "b"]),
    ]
    assert compute_source_mixing(skills)[1]["min"] == 2


def test_source_mixing_marks_missing_parents_incomplete():
    mixing = compute_source_mixing([_skill("child", level=1, parents=["missing"])])
    assert mixing[1]["complete"] is False
    assert mixing[1]["mean"] is None
    assert "Missing parent" in mixing[1]["errors"][0]


def test_source_mixing_detects_cycles():
    mixing = compute_source_mixing([
        _skill("a", level=1, parents=["b"]),
        _skill("b", level=2, parents=["a"]),
    ])
    assert all(row["complete"] is False for row in mixing.values())
    assert all("Cyclic lineage" in row["errors"][0] for row in mixing.values())
