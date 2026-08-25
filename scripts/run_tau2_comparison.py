"""Run tau2 task comparison across baseline/static/dynamic FedSkill.

Stages:
1. Baseline ``llm_agent`` on train tasks.
2. Extract/evolve FedSkill skills from the baseline traces.
3. Static pre-task top-k FedSkill reinjection on the same train tasks.
4. Dynamic hidden-librarian FedSkill reinjection on the same train tasks.

The script is designed to be restartable:
- tau2 runs use ``--auto-resume``.
- completed tau2 result files are skipped by default.
- extraction/evolution is skipped if ``evolved_all.json`` already exists unless
  ``--force-extract`` is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from fedskill.server.quality import (
    apply_feedback_events,
    feedback_events_from_tau2_dynamic_log,
    feedback_events_from_tau2_static_selection,
    load_skills,
    save_quality_summary,
    save_skills,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tau2 train baseline/static/dynamic comparison")
    parser.add_argument(
        "--tau2-root",
        default=os.getenv("TAU2_ROOT"),
        required=not bool(os.getenv("TAU2_ROOT")),
        help="Path to a tau2-bench checkout (or set TAU2_ROOT).",
    )
    parser.add_argument("--output-dir", default="output/tau2_airline_train50")
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--task-set-name", default=None)
    parser.add_argument("--task-split-name", default="base")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=50)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument(
        "--task-ids-file",
        help="JSON file containing the exact task ID list; overrides start/count.",
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--evolve-rounds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--agent-model", default=os.getenv("SKILLLADDER_AGENT_MODEL"), required=not bool(os.getenv("SKILLLADDER_AGENT_MODEL")))
    parser.add_argument("--user-model", default=os.getenv("SKILLLADDER_USER_MODEL"), required=not bool(os.getenv("SKILLLADDER_USER_MODEL")))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--max-static-skills", type=int, default=10)
    parser.add_argument("--dynamic-top-k", type=int, default=3)
    parser.add_argument("--retrieval-candidates", type=int, default=8)
    parser.add_argument("--librarian-max-rounds", type=int, default=2)
    parser.add_argument("--skill-min-overlap", type=int, default=1)
    parser.add_argument("--disable-quality-update", action="store_true")
    parser.add_argument("--quality-alpha", type=float, default=0.35)
    parser.add_argument("--retire-quality-threshold", type=float, default=0.15)
    parser.add_argument("--retire-min-injections", type=int, default=5)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-runs", action="store_true", help="Run tau2 stages even when result files look complete")
    return parser.parse_args()


def task_ids(args: argparse.Namespace) -> list[str]:
    if args.task_ids:
        return [str(item) for item in args.task_ids]
    if args.task_ids_file:
        data = json.loads(Path(args.task_ids_file).read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("--task-ids-file must contain a non-empty JSON array")
        return [str(item) for item in data]
    return [str(i) for i in range(args.task_start, args.task_start + args.task_count)]


def run_prefix(args: argparse.Namespace, ids: list[str]) -> str:
    if ids and all(item.isdigit() for item in ids):
        sorted_ids = sorted(int(item) for item in ids)
        if sorted_ids == list(range(sorted_ids[0], sorted_ids[-1] + 1)):
            return f"fedskill_tau2_{args.domain}_train_{sorted_ids[0]}_{sorted_ids[-1]}_x{args.num_trials}"
    raw = "_".join(ids[:2])
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")[:48] or "ids"
    digest = hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()[:10]
    return f"fedskill_tau2_{args.domain}_train_custom_{slug}_{digest}_n{len(ids)}_x{args.num_trials}"


def make_env(
    args: argparse.Namespace,
    *,
    skills_path: Path | None = None,
    injection_mode: str | None = None,
    skill_log_path: Path | None = None,
) -> dict[str, str]:
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
    if skills_path is not None:
        env["FEDSKILL_TAU2_SKILLS_PATH"] = str(skills_path)
        env["FEDSKILL_TAU2_MAX_SKILLS"] = str(args.max_static_skills)
    if injection_mode is not None:
        env["FEDSKILL_TAU2_INJECTION_MODE"] = injection_mode
    if injection_mode == "dynamic":
        env["FEDSKILL_TAU2_DYNAMIC_TOP_K"] = str(args.dynamic_top_k)
        env["FEDSKILL_TAU2_RETRIEVAL_CANDIDATES"] = str(args.retrieval_candidates)
        env["FEDSKILL_TAU2_LIBRARIAN_MAX_ROUNDS"] = str(args.librarian_max_rounds)
        env["FEDSKILL_TAU2_MIN_SKILL_OVERLAP"] = str(args.skill_min_overlap)
        if skill_log_path is not None:
            env["FEDSKILL_TAU2_SKILL_LOG_PATH"] = str(skill_log_path)
    return env


def run_logged(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(f"[started] {datetime.now().isoformat()}\n")
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
        log.write(f"\n[finished] {datetime.now().isoformat()}\n")
        log.write(f"[exit_code] {code}\n")
    if code != 0:
        raise SystemExit(code)


def tau2_result_path(args: argparse.Namespace, run_name: str) -> Path:
    return Path(args.tau2_root) / "data" / "simulations" / run_name / "results.json"


def is_complete(args: argparse.Namespace, run_name: str, expected: int) -> bool:
    path = tau2_result_path(args, run_name)
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data.get("simulations", [])) >= expected
    except Exception:
        return False


def tau2_run(
    args: argparse.Namespace,
    *,
    run_name: str,
    ids: list[str],
    agent: str,
    log_path: Path,
    skills_path: Path | None = None,
    injection_mode: str | None = None,
    skill_log_path: Path | None = None,
) -> None:
    expected = len(ids) * args.num_trials
    if not args.force_runs and is_complete(args, run_name, expected):
        print(f"[skip] {run_name} already has {expected} simulations")
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
        agent,
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
        run_name,
        "--log-level",
        "INFO",
    ]
    if args.task_set_name:
        cmd[cmd.index("--task-split-name"):cmd.index("--task-split-name")] = [
            "--task-set-name",
            args.task_set_name,
        ]
    run_logged(
        cmd,
        Path(args.tau2_root),
        make_env(
            args,
            skills_path=skills_path,
            injection_mode=injection_mode,
            skill_log_path=skill_log_path,
        ),
        log_path,
    )


def summarize_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sims = data.get("simulations", [])
    rewards = [float((s.get("reward_info") or {}).get("reward") or 0.0) for s in sims]
    durations = [float(s.get("duration") or 0.0) for s in sims]
    messages = [len(s.get("messages") or []) for s in sims]
    tool_calls = [sum(len(m.get("tool_calls") or []) for m in s.get("messages") or []) for s in sims]
    return {
        "exists": True,
        "num_simulations": len(sims),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "avg_duration": sum(durations) / len(durations) if durations else 0.0,
        "avg_messages": sum(messages) / len(messages) if messages else 0.0,
        "avg_tool_calls": sum(tool_calls) / len(tool_calls) if tool_calls else 0.0,
        "total_messages": sum(messages),
        "total_tool_calls": sum(tool_calls),
        "rewards_by_task": {
            str(s.get("task_id")): float((s.get("reward_info") or {}).get("reward") or 0.0)
            for s in sims
        },
        "terminations_by_task": {str(s.get("task_id")): s.get("termination_reason") for s in sims},
        "durations_by_task": {str(s.get("task_id")): float(s.get("duration") or 0.0) for s in sims},
        "messages_by_task": {str(s.get("task_id")): len(s.get("messages") or []) for s in sims},
        "tool_calls_by_task": {
            str(s.get("task_id")): sum(len(m.get("tool_calls") or []) for m in s.get("messages") or [])
            for s in sims
        },
    }


def consultation_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    counts: Counter[str] = Counter()
    need_counts: Counter[str] = Counter()
    skill_counter: Counter[str] = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            task_id = str(item.get("task_id"))
            counts[task_id] += 1
            if item.get("need_skills"):
                need_counts[task_id] += 1
            for skill in item.get("selected_skills") or []:
                name = skill.get("name")
                if name:
                    skill_counter[str(name)] += 1
    return {
        "exists": True,
        "total_consultations": sum(counts.values()),
        "consultations_by_task": dict(sorted(counts.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999999)),
        "need_skills_by_task": dict(sorted(need_counts.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999999)),
        "top_selected_skills": skill_counter.most_common(20),
    }


def write_summary(output_dir: Path, results: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = ["# tau2 Task Comparison Summary", ""]
    for name, result in results.items():
        lines.append(f"## {name}")
        if isinstance(result, dict):
            for key, value in result.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append(str(result))
        lines.append("")
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def update_skill_quality_from_tau2_runs(
    args: argparse.Namespace,
    *,
    skills_path: Path,
    output_dir: Path,
    static_run: str,
    dynamic_run: str,
    dynamic_skill_log: Path,
) -> dict[str, Any]:
    """Update skill quality from static/dynamic tau2 outcomes and save artifacts."""
    if args.disable_quality_update:
        return {"enabled": False, "reason": "disabled by --disable-quality-update"}
    if not skills_path.exists():
        return {"enabled": False, "reason": f"skills file not found: {skills_path}"}

    skills = load_skills(skills_path)
    if not skills:
        return {"enabled": False, "reason": "no skills loaded"}

    static_results = tau2_result_path(args, static_run)
    dynamic_results = tau2_result_path(args, dynamic_run)
    static_events = feedback_events_from_tau2_static_selection(
        static_results,
        skills,
        max_skills=args.max_static_skills,
        source="tau2_static_reinject",
        round_id=static_run,
    )
    dynamic_events = feedback_events_from_tau2_dynamic_log(
        dynamic_results,
        dynamic_skill_log,
        source="tau2_dynamic_librarian",
        round_id=dynamic_run,
    )

    summary = apply_feedback_events(
        skills,
        [*static_events, *dynamic_events],
        alpha=args.quality_alpha,
        retire_quality_threshold=args.retire_quality_threshold,
        retire_min_injections=args.retire_min_injections,
    )
    quality_skills_path = output_dir / "skills_from_train" / "evolved_quality_updated.json"
    quality_summary_path = output_dir / "skill_quality_summary.json"
    save_skills(quality_skills_path, skills)
    summary = {
        "enabled": True,
        "source_skills_path": str(skills_path),
        "quality_skills_path": str(quality_skills_path),
        "quality_summary_path": str(quality_summary_path),
        "static_feedback_events": len(static_events),
        "dynamic_feedback_events": len(dynamic_events),
        **summary,
    }
    save_quality_summary(quality_summary_path, summary)
    return summary


def write_settings(
    output_dir: Path,
    args: argparse.Namespace,
    ids: list[str],
    baseline_run: str,
    static_run: str,
    dynamic_run: str,
    skills_path: Path,
    skill_log_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# tau2 {args.domain} Task Comparison Settings",
        "",
        f"Created: {datetime.now().isoformat()}",
        "",
        "## Natural language",
        "",
        f"Run tau2 {args.domain} tasks {ids[0]}-{ids[-1]} ({len(ids)} tasks) with three methods:",
        "1. baseline `llm_agent` without FedSkill.",
        "2. static `fedskill_llm_agent` with pre-task top-10 skill injection.",
        "3. dynamic `fedskill_llm_agent` with hidden skill librarian and per-turn logging.",
        "",
        "Skills for both injected methods are extracted/evolved from the baseline traces.",
        "",
        "## Key settings",
        "",
        f"- tau2 root: {args.tau2_root}",
        f"- output dir: {output_dir}",
        f"- domain: {args.domain}",
        f"- task set name: {args.task_set_name}",
        f"- task split name: {args.task_split_name}",
        f"- task ids: {' '.join(ids)}",
        f"- num trials: {args.num_trials}",
        f"- max steps: {args.max_steps}",
        f"- max concurrency: {args.max_concurrency}",
        f"- agent model: {args.agent_model}",
        f"- user model: {args.user_model}",
        f"- proxy: {args.openai_base_url}",
        f"- skills path: {skills_path}",
        f"- static max skills: {args.max_static_skills}",
        f"- dynamic top-k: {args.dynamic_top_k}",
        f"- dynamic retrieval candidates: {args.retrieval_candidates}",
        f"- dynamic librarian max rounds: {args.librarian_max_rounds}",
        f"- dynamic min skill overlap: {args.skill_min_overlap}",
        f"- dynamic skill log: {skill_log_path}",
        "",
        "## Run names",
        "",
        f"- baseline: {baseline_run}",
        f"- static: {static_run}",
        f"- dynamic: {dynamic_run}",
    ]
    with open(output_dir / "experiment_settings.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    ids = task_ids(args)
    output_dir = REPO_ROOT / args.output_dir
    logs_dir = output_dir / "logs"
    skills_dir = output_dir / "skills_from_train"
    skills_path = skills_dir / "evolved_all.json"
    expected = len(ids) * args.num_trials

    prefix = run_prefix(args, ids)
    baseline_run = f"{prefix}_baseline"
    static_run = f"{prefix}_static_reinject"
    dynamic_run = f"{prefix}_dynamic_librarian"
    dynamic_skill_log = logs_dir / "03_dynamic_librarian_skill_consultations.jsonl"

    write_settings(output_dir, args, ids, baseline_run, static_run, dynamic_run, skills_path, dynamic_skill_log)

    tau2_run(
        args,
        run_name=baseline_run,
        ids=ids,
        agent="llm_agent",
        log_path=logs_dir / "01_baseline.log",
    )

    if args.force_extract or not skills_path.exists():
        run_logged(
            [
                sys.executable,
                str(SCRIPT_DIR / "run_tau2_phase1.py"),
                "--tau2-root",
                str(Path(args.tau2_root)),
                "--run-name",
                baseline_run,
                "--output-dir",
                str(skills_dir),
                "--limit",
                str(expected),
                "--evolve-rounds",
                str(args.evolve_rounds),
            ],
            REPO_ROOT,
            os.environ.copy(),
            logs_dir / "02_extract_evolve.log",
        )
    else:
        print(f"[skip] skills already exist: {skills_path}")

    tau2_run(
        args,
        run_name=static_run,
        ids=ids,
        agent="fedskill_llm_agent",
        log_path=logs_dir / "03_static_reinject.log",
        skills_path=skills_path,
        injection_mode="static",
    )

    tau2_run(
        args,
        run_name=dynamic_run,
        ids=ids,
        agent="fedskill_llm_agent",
        log_path=logs_dir / "04_dynamic_librarian.log",
        skills_path=skills_path,
        injection_mode="dynamic",
        skill_log_path=dynamic_skill_log,
    )

    results = {
        "settings": {
            "task_ids": ids,
            "expected_simulations_per_run": expected,
            "skills_path": str(skills_path),
        },
        "baseline": summarize_result(tau2_result_path(args, baseline_run)),
        "static_reinject": summarize_result(tau2_result_path(args, static_run)),
        "dynamic_librarian": summarize_result(tau2_result_path(args, dynamic_run)),
        "dynamic_consultations": consultation_summary(dynamic_skill_log),
    }
    results["skill_quality_update"] = update_skill_quality_from_tau2_runs(
        args,
        skills_path=skills_path,
        output_dir=output_dir,
        static_run=static_run,
        dynamic_run=dynamic_run,
        dynamic_skill_log=dynamic_skill_log,
    )
    write_summary(output_dir, results)
    print(f"[done] summary written to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
