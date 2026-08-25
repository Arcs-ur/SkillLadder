"""Run AWM-style baseline on a tau2 domain.

This is a one-condition variant of static reinjection where:
- Only L0 skills extracted from successful baseline traces are loaded
  (mirroring AWM's success-only workflow memory).
- Retrieval scoring is cosine similarity over MiniLM embeddings, with NO
  quality bonus and NO level bonus (since L0 skills have neither).
- Top-k=10 skills are injected into the system prompt for the whole episode.

Reuses the same tau2 harness, the same baseline rewards, and the same
seed/temperature setup as ``run_tau2_train_compare.py`` so the AWM result
is directly comparable to the existing baseline / L0-only-static rows.

Example:
    python run_awm_baseline.py \
        --domain airline \
        --train-dir output/tau2_airline_train50 \
        --task-start 0 --task-count 50 \
        --skills-out output/baselines_awm/airline_l0_success.json \
        --run-name fedskill_tau2_airline_awm_0_49_x1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument(
        "--tau2-root",
        default=os.getenv("TAU2_ROOT"),
        required=not bool(os.getenv("TAU2_ROOT")),
        help="Path to a tau2-bench checkout (or set TAU2_ROOT).",
    )
    parser.add_argument(
        "--train-dir",
        required=True,
        help="Directory with skills_from_train/{sessions.json,evolved_all.json}",
    )
    parser.add_argument("--task-split-name", default="base")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=50)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--task-ids-file", help="JSON file containing exact task IDs.")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--agent-model", default=os.getenv("SKILLLADDER_AGENT_MODEL"), required=not bool(os.getenv("SKILLLADDER_AGENT_MODEL")))
    parser.add_argument("--user-model", default=os.getenv("SKILLLADDER_USER_MODEL"), required=not bool(os.getenv("SKILLLADDER_USER_MODEL")))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--max-skills", type=int, default=10)
    parser.add_argument(
        "--skills-out",
        required=True,
        help="Where to write the success-only L0 skills file (also passed to tau2).",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="tau2 --save-to run name (results.json lands under data/simulations/<name>/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if a complete results.json already exists.",
    )
    parser.add_argument("--log-dir", default=None)
    return parser.parse_args()


def task_ids(args: argparse.Namespace) -> list[str]:
    if args.task_ids:
        return [str(t) for t in args.task_ids]
    if args.task_ids_file:
        data = json.loads(Path(args.task_ids_file).read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("--task-ids-file must contain a non-empty JSON array")
        return [str(item) for item in data]
    return [str(i) for i in range(args.task_start, args.task_start + args.task_count)]


def build_skills(args: argparse.Namespace) -> tuple[Path, int]:
    """Filter L0 skills to those whose source_task succeeded at baseline."""
    train_dir = Path(args.train_dir)
    sessions = json.loads((train_dir / "skills_from_train" / "sessions.json").read_text(encoding="utf-8"))
    success_ids = {str(s["task_id"]) for s in sessions if s.get("outcome") == "success"}

    skills = json.loads((train_dir / "skills_from_train" / "evolved_all.json").read_text(encoding="utf-8"))
    l0_success = [
        s
        for s in skills
        if int(s.get("level", 0)) == 0 and str(s.get("source_task", "")) in success_ids
    ]
    out_path = Path(args.skills_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(l0_success, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[awm] domain={args.domain} success_sessions={len(success_ids)} l0_success_skills={len(l0_success)} -> {out_path}"
    )
    return out_path, len(l0_success)


def tau2_result_path(args: argparse.Namespace, run_name: str) -> Path:
    return Path(args.tau2_root) / "data" / "simulations" / run_name / "results.json"


def is_complete(args: argparse.Namespace, run_name: str, expected: int) -> bool:
    path = tau2_result_path(args, run_name)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data.get("simulations", [])) >= expected
    except Exception:
        return False


def make_env(args: argparse.Namespace, skills_path: Path, log_path: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(args.tau2_root) / "src")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_COLOR"] = "1"
    if args.openai_base_url:
        env["OPENAI_API_BASE"] = args.openai_base_url
        env["OPENAI_BASE_URL"] = args.openai_base_url
    if args.openai_api_key:
        env["OPENAI_API_KEY"] = args.openai_api_key
    env["FEDSKILL_TAU2_SKILLS_PATH"] = str(skills_path)
    env["FEDSKILL_TAU2_MAX_SKILLS"] = str(args.max_skills)
    env["FEDSKILL_TAU2_INJECTION_MODE"] = "static"
    env["FEDSKILL_TAU2_SCORE_MODE"] = "embedding"
    if log_path is not None:
        env["FEDSKILL_TAU2_SKILL_LOG_PATH"] = str(log_path)
    return env


def run_tau2(args: argparse.Namespace, skills_path: Path, ids: list[str], log_path: Path | None) -> None:
    expected = len(ids) * args.num_trials
    if not args.force and is_complete(args, args.run_name, expected):
        print(f"[awm] skip: {args.run_name} already has {expected} simulations")
        return

    cmd = [
        sys.executable,
        "-m",
        "tau2.cli",
        "run",
        "--domain",
        args.domain,
        "--task-split-name",
        args.task_split_name,
        "--task-ids",
        *ids,
        "--num-trials",
        str(args.num_trials),
        "--max-concurrency",
        str(args.max_concurrency),
        "--agent",
        "fedskill_llm_agent",
        "--agent-llm",
        args.agent_model,
        "--user-llm",
        args.user_model,
        "--agent-llm-args",
        json.dumps({"temperature": 0.2}),
        "--user-llm-args",
        json.dumps({"temperature": 0.2}),
        "--max-steps",
        str(args.max_steps),
        "--max-retries",
        "0",
        "--retry-delay",
        "1",
        "--auto-resume",
        "--save-to",
        args.run_name,
        "--log-level",
        "INFO",
    ]
    env = make_env(args, skills_path, log_path)
    print("\n$ " + " ".join(cmd), flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(args.tau2_root)),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    log_handle = open(log_path, "a", encoding="utf-8") if log_path else None
    if log_handle:
        log_handle.write("\n" + "=" * 100 + "\n")
        log_handle.write(f"[started] {datetime.now().isoformat()}\n")
        log_handle.write("$ " + " ".join(cmd) + "\n")
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log_handle:
                log_handle.write(line)
    finally:
        rc = proc.wait()
        if log_handle:
            log_handle.write(f"\n[finished] {datetime.now().isoformat()}\n")
            log_handle.write(f"[exit_code] {rc}\n")
            log_handle.close()
    if rc != 0:
        raise SystemExit(rc)


def summarize(args: argparse.Namespace, run_name: str) -> dict:
    path = tau2_result_path(args, run_name)
    if not path.exists():
        return {"exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    sims = data.get("simulations", [])
    rewards = [float((s.get("reward_info") or {}).get("reward") or 0.0) for s in sims]
    return {
        "exists": True,
        "num_simulations": len(sims),
        "reward_mean": mean(rewards) if rewards else 0.0,
        "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "num_success": sum(1 for r in rewards if r > 0.5),
    }


def main() -> None:
    args = parse_args()
    skills_path, n_skills = build_skills(args)
    if n_skills == 0:
        print("[awm] no success-only L0 skills available; aborting.")
        return
    ids = task_ids(args)
    log_dir = Path(args.log_dir) if args.log_dir else Path("output/baselines_awm/logs") / args.domain
    log_path = log_dir / f"{args.run_name}.log"
    run_tau2(args, skills_path, ids, log_path)
    summary = summarize(args, args.run_name)
    print("\n=== AWM baseline summary ===")
    print(json.dumps({"domain": args.domain, "run": args.run_name, **summary}, indent=2))
    summary_path = log_dir / f"{args.run_name}.summary.json"
    summary_path.write_text(
        json.dumps({"domain": args.domain, "run": args.run_name, **summary}, indent=2),
        encoding="utf-8",
    )
    print(f"[awm] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
