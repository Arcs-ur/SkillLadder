import pytest

from run_tau2_comparison import summarize_result
from fedskill.utils.run_manifest import atomic_write_json


def test_summary_preserves_trials_and_missing_reward(tmp_path):
    path = tmp_path / "results.json"
    atomic_write_json(path, {"simulations": [
        dict(id="a", task_id="t", trial=0, seed=42, reward_info={"reward": 0}, termination_reason="stop"),
        dict(id="b", task_id="t", trial=1, seed=43, reward_info={"reward": 1}, termination_reason="stop"),
        dict(id="c", task_id="u", trial=0, seed=44, reward_info=None, termination_reason="error"),
    ]})
    summary = summarize_result(path)
    assert len(summary["simulations"]) == summary["num_simulations"] == 3
    assert summary["avg_reward"] == 0.5
    assert summary["reward_se"] == pytest.approx(0.5)
    assert summary["missing_rewards"] == 1
    assert summary["per_task"]["t"] == {"mean": 0.5, "se": pytest.approx(0.5), "n": 2}
    assert summary["per_task"]["u"]["mean"] is None
    assert summary["simulations"][0]["reward"] == 0
    assert summary["simulations"][2]["reward"] is None


def test_single_trial_compatibility(tmp_path):
    path = tmp_path / "results.json"
    atomic_write_json(path, {"simulations": [
        dict(id="a", task_id="t", trial=0, reward_info={"reward": 1}, duration=2,
             termination_reason="stop", messages=[{"role": "assistant"}]),
    ]})
    summary = summarize_result(path)
    assert summary["rewards_by_task"] == {"t": 1}
    assert summary["terminations_by_task"] == {"t": "stop"}
    assert summary["reward_se"] is None
