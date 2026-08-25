import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "configs" / "task_splits"


def _load(name: str) -> list[str]:
    data = json.loads((SPLITS / name).read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert all(isinstance(item, str) for item in data)
    assert len(data) == len(set(data))
    return data


def test_numeric_task_splits() -> None:
    assert _load("airline_50.json") == [str(i) for i in range(50)]
    assert _load("retail_114.json") == [str(i) for i in range(114)]


def test_banking_task_split() -> None:
    task_ids = _load("banking_knowledge_97.json")
    assert len(task_ids) == 97
    assert task_ids[0] == "task_001"
    assert task_ids[-1] == "task_102"


def test_telecom_task_split() -> None:
    task_ids = _load("telecom_200.json")
    assert len(task_ids) == 200
    assert all("[PERSONA:" in item for item in task_ids)
