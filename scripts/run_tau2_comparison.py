"""Compare tau2 baseline, L0, shared L1+, Full static and Full dynamic.

Generate a source library or reuse fixed source artifacts for transfer.
Manifests bind resumed stages to code, parameters and inputs; force flags
create new attempts. Dry-run validates inputs without writing run artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections import defaultdict
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
from fedskill.utils.run_manifest import (
    PROTOCOL_VERSION, TAU2_REVISION, atomic_write_json, bind_state,
    code_identity, file_digest, fingerprint, prompt_digest, read_json, public_endpoint,
)
from fedskill.utils.results import mean_se, simulation_rows
from fedskill.utils.config import (
    add_evolution_args, parse_config_args, public_parameters, skill_env,
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
    parser.add_argument("--agent-model", default=os.getenv("SKILLLADDER_AGENT_MODEL"))
    parser.add_argument("--user-model", default=os.getenv("SKILLLADDER_USER_MODEL"))
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
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--agent-temperature", type=float, default=0.2)
    parser.add_argument("--user-temperature", type=float, default=0.2)
    parser.add_argument("--conditions", default="baseline,full-static,full-dynamic")
    parser.add_argument("--source-artifacts-dir", help="Reuse an existing source library without collecting consumer traces.")
    parser.add_argument("--skills-path", help="Use an existing evolved library; L0-only also needs source artifacts.")
    parser.add_argument("--source-model", help="Source simulator model ID for transfer (Sonnet in the paper).")
    parser.add_argument("--dry-run", action="store_true")
    add_evolution_args(parser)
    args = parse_config_args(parser)
    for key in ("tau2_root", "agent_model", "user_model"):
        if not getattr(args, key):
            parser.error(f"--{key.replace('_', '-')} or its environment variable is required")
    return args


def task_ids(args: argparse.Namespace) -> list[str]:
    if args.task_ids:
        ids = [str(item) for item in args.task_ids]
    elif args.task_ids_file:
        data = json.loads(Path(args.task_ids_file).read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("--task-ids-file must contain a non-empty JSON array")
        ids = [str(item) for item in data]
    else:
        ids = [str(i) for i in range(args.task_start, args.task_start + args.task_count)]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Task IDs must be nonempty and unique.")
    return ids


def run_prefix(args: argparse.Namespace, ids: list[str]) -> str:
    digest = fingerprint(run_identity(args, ids))[:12]
    if ids and all(item.isdigit() for item in ids):
        sorted_ids = sorted(int(item) for item in ids)
        if sorted_ids == list(range(sorted_ids[0], sorted_ids[-1] + 1)):
            return f"fedskill_tau2_{args.domain}_train_{sorted_ids[0]}_{sorted_ids[-1]}_x{args.num_trials}_{digest}"
    raw = "_".join(ids[:2])
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")[:48] or "ids"
    return f"fedskill_tau2_{args.domain}_train_custom_{slug}_{digest}_n{len(ids)}_x{args.num_trials}"


def run_identity(args, ids, *, agent=None, skills_path=None, injection_mode=None):
    keys = (
        "domain", "task_set_name", "task_split_name", "num_trials", "max_steps",
        "max_concurrency", "agent_model", "user_model", "seed", "max_retries",
        "agent_temperature", "user_temperature", "max_static_skills",
        "dynamic_top_k", "retrieval_candidates", "librarian_max_rounds", "skill_min_overlap", "score_mode",
        "embedding_model", "embedding_revision", "strict_embeddings",
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "tau2_revision": TAU2_REVISION,
        "code": code_identity(REPO_ROOT),
        "prompt_digest": prompt_digest(REPO_ROOT),
        "task_ids": ids,
        "parameters": {key: getattr(args, key, None) for key in keys},
        "provider_base_url": public_endpoint(getattr(args, "openai_base_url", "")),
        "agent": agent,
        "condition": injection_mode,
        "skills_digest": file_digest(skills_path) if skills_path else None,
    }


def make_env(
    args: argparse.Namespace,
    *,
    skills_path: Path | None = None,
    injection_mode: str | None = None,
    skill_log_path: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("FEDSKILL_TAU2_"):
            del env[key]
    env["PYTHONPATH"] = str(Path(args.tau2_root) / "src")
    env["TAU2_DATA_DIR"] = str(Path(args.tau2_root) / "data")
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
    if skills_path is not None:
        env["FEDSKILL_TAU2_SCORE_MODE"] = getattr(args, "score_mode", "token_overlap")
        if getattr(args, "embedding_model", None):
            env["FEDSKILL_TAU2_EMBEDDING_MODEL"] = args.embedding_model
        if getattr(args, "embedding_revision", None):
            env["FEDSKILL_TAU2_EMBEDDING_REVISION"] = args.embedding_revision
        if getattr(args, "strict_embeddings", False):
            env["FEDSKILL_TAU2_STRICT_EMBEDDINGS"] = "1"
    if skill_log_path is not None:
        env["FEDSKILL_TAU2_SKILL_LOG_PATH"] = str(skill_log_path)
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


def is_complete(args: argparse.Namespace, run_name: str, expected: int,
                *, ids: list[str] | None = None, expected_fingerprint: str | None = None) -> bool:
    path = tau2_result_path(args, run_name)
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sims = data.get("simulations", [])
        if len(sims) != expected:
            return False
        if expected_fingerprint:
            manifest_path = path.parent / "skillladder_manifest.json"
            if not manifest_path.exists() or read_json(manifest_path).get("fingerprint") != expected_fingerprint:
                return False
        from fedskill.utils.results import valid_simulation_coverage
        return valid_simulation_coverage(sims, ids or task_ids(args), args.num_trials)
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
) -> str:
    import uuid
    identity = run_identity(args, ids, agent=agent, skills_path=skills_path, injection_mode=injection_mode)
    run_name = f"{run_name}_{fingerprint(identity)[:12]}"
    if args.force_runs:
        run_name += f"_attempt_{uuid.uuid4().hex[:12]}"
    manifest_path = tau2_result_path(args, run_name).parent / "skillladder_manifest.json"
    state = bind_state(manifest_path, identity)
    expected = len(ids) * args.num_trials
    if not args.force_runs and is_complete(args, run_name, expected, ids=ids,
                                          expected_fingerprint=state["fingerprint"]):
        print(f"[skip] {run_name} already has {expected} simulations")
        return run_name

    cmd = tau2_command(args, run_name=run_name, ids=ids, agent=agent)
    run_logged(
        cmd,
        Path(args.tau2_root),
        make_env(args, skills_path=skills_path, injection_mode=injection_mode,
                 skill_log_path=tau2_skill_log_path(args, run_name) if skills_path else None),
        log_path,
    )
    if not is_complete(args, run_name, expected, ids=ids, expected_fingerprint=state["fingerprint"]):
        raise RuntimeError(f"Incomplete or invalid task/trial coverage in {tau2_result_path(args, run_name)}")
    state.update(status="complete", results_digest=file_digest(tau2_result_path(args, run_name)))
    atomic_write_json(manifest_path, state)
    return run_name


def tau2_command(args, *, run_name, ids, agent):
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
        json.dumps({"temperature": getattr(args, "agent_temperature", 0.2)}),
        "--user-llm-args",
        json.dumps({"temperature": getattr(args, "user_temperature", 0.2)}),
        "--max-steps",
        str(args.max_steps),
        "--max-retries",
        str(getattr(args, "max_retries", 0)),
        "--retry-delay",
        "1",
        "--auto-resume",
        "--save-to",
        run_name,
        "--log-level",
        "INFO",
        "--seed",
        str(getattr(args, "seed", 42)),
    ]
    if args.task_set_name:
        cmd[cmd.index("--task-split-name"):cmd.index("--task-split-name")] = [
            "--task-set-name",
            args.task_set_name,
        ]
    return cmd


def tau2_skill_log_path(args, run_name):
    return tau2_result_path(args, run_name).parent / "skill_usage.jsonl"


def summarize_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = simulation_rows(data)
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    rewards = mean_se([row["reward"] for row in rows])
    task_stats = {task: mean_se([r["reward"] for r in group]) for task, group in by_task.items()}
    return {
        "exists": True,
        "num_simulations": len(rows),
        "simulations": rows,
        "missing_rewards": len(rows) - rewards["n"],
        "avg_reward": rewards["mean"],
        "reward_se": rewards["se"],
        "reward_se_unit": "simulation (descriptive; trials of a task may be correlated)",
        "task_mean_reward_stats": mean_se([s["mean"] for s in task_stats.values()]),
        "task_mean_reward_se_unit": "task",
        "per_task": task_stats,
        "avg_duration": mean_se([r["duration"] for r in rows])["mean"],
        "avg_messages": mean_se([r["messages"] for r in rows])["mean"],
        "avg_tool_calls": mean_se([r["tool_calls"] for r in rows])["mean"],
        "total_messages": sum(r["messages"] for r in rows),
        "total_tool_calls": sum(r["tool_calls"] for r in rows),
        "rewards_by_task": {task: stats["mean"] for task, stats in task_stats.items()},
        "terminations_by_task": {task: [r["status"] for r in group] if len(group) > 1 else group[0]["status"]
                                 for task, group in by_task.items()},
        **{f"{'durations' if key == 'duration' else key}_by_task": {task: mean_se([r[key] for r in group])["mean"]
                              for task, group in by_task.items()}
           for key in ("duration", "messages", "tool_calls")},
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
    static_skill_log: Path | None = None,
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
    from fedskill.server.quality import AmbiguousFeedbackError
    paused = {}
    try:
        static_events = feedback_events_from_tau2_static_selection(
        static_results,
        skills,
        max_skills=args.max_static_skills,
        source="tau2_static_reinject",
        round_id=static_run,
        selection_log_path=static_skill_log,
        )
    except AmbiguousFeedbackError as exc:
        static_events = []
        paused["static"] = str(exc)
    try:
        dynamic_events = feedback_events_from_tau2_dynamic_log(
        dynamic_results,
        dynamic_skill_log,
        source="tau2_dynamic_librarian",
        round_id=dynamic_run,
        )
    except AmbiguousFeedbackError as exc:
        dynamic_events = []
        paused["dynamic"] = str(exc)

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
        "paused_feedback": paused,
        **summary,
    }
    save_quality_summary(quality_summary_path, summary)
    return summary


CONDITIONS = {"baseline", "l0-static", "l1plus-static", "full-static", "full-dynamic"}


def selected_conditions(args):
    conditions = [name.strip() for name in args.conditions.split(",") if name.strip()]
    if not conditions or len(conditions) != len(set(conditions)) or set(conditions) - CONDITIONS:
        raise ValueError(f"--conditions must be unique members of {sorted(CONDITIONS)}")
    return conditions


def source_files(args):
    source = Path(args.source_artifacts_dir).resolve() if args.source_artifacts_dir else None
    if source:
        if not source.exists():
            raise ValueError(f"Missing source artifact directory {source}")
        if not (source / "evolved_all.json").exists() and (source / "skills_from_train").is_dir():
            source = source / "skills_from_train"
    skills = Path(args.skills_path).resolve() if args.skills_path else (source / "evolved_all.json" if source else None)
    if args.skills_path and not skills.is_file():
        raise ValueError(f"Missing skills file {skills}")
    if args.source_model and args.user_model != args.source_model:
        raise ValueError("Fixed-library transfer must preserve --source-model as the user simulator.")
    if source and (source / "phase1_manifest.json").exists():
        models = read_json(source / "phase1_manifest.json").get("identity", {}).get("source_models", {})
        if models.get("user") and args.user_model != models["user"]:
            raise ValueError("The user simulator differs from the saved source-library simulator.")
    return source, skills


def condition_candidates(condition, source, skills):
    if condition == "l0-static":
        if source is None or not (source / "level0_all.json").is_file():
            raise ValueError("l0-static requires original level0_all.json via --source-artifacts-dir.")
        candidates = load_skills(source / "level0_all.json")
        candidates = [s for s in candidates if s.level == 0]
    else:
        if skills is None:
            raise ValueError(f"{condition} requires an evolved skills library.")
        candidates = load_skills(skills)
        if condition == "l1plus-static":
            candidates = [s for s in candidates if s.level >= 1]
    unique = {}
    for skill in candidates:
        unique.setdefault(skill.id, skill)
    if not unique:
        raise ValueError(f"No eligible skills for {condition}")
    return list(unique.values())


def prepare_condition_library(condition, source, skills, output_dir):
    candidates = condition_candidates(condition, source, skills)
    path = output_dir / "condition_skills" / f"{condition}.json"
    manifest = path.with_suffix(".manifest.json")
    state = bind_state(manifest, {"condition": condition, "source": [s.to_dict() for s in candidates]})
    if state.get("status") == "complete":
        if not path.exists() or file_digest(path) != state.get("output_digest"):
            raise ValueError(f"Changed condition library {path}")
        return path
    if condition == "l1plus-static":
        from fedskill.utils.privacy import prepare_for_sharing
        released = [prepare_for_sharing(s) for s in candidates]
        atomic_write_json(path.parent / "private/l1plus_release_id_map.json",
                          {public.id: private.id for public, private in zip(released, candidates)})
        candidates = released
    save_skills(path, candidates)
    state.update(status="complete", output_digest=file_digest(path))
    atomic_write_json(manifest, state)
    return path


def phase1_command(args, baseline_run, skills_dir, expected):
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_tau2_phase1.py"),
        "--tau2-root", args.tau2_root, "--run-name", baseline_run,
        "--domain", args.domain, "--output-dir", str(skills_dir), "--limit", str(expected),
    ]
    # Forward the final effective values; child parsing cannot reintroduce YAML defaults.
    for key in (
        "evolve_rounds", "seed", "clustering_strategy", "embedding_model", "distance_threshold",
        "max_cluster_size", "min_cluster_size", "minimum_judge_score", "score_gap_ratio",
        "max_synthesis_retries", "extraction_temperature", "synthesis_temperature", "gate_temperature",
    ):
        cmd.extend([f"--{key.replace('_', '-')}", str(getattr(args, key))])
    if args.embedding_revision:
        cmd.extend(["--embedding-revision", args.embedding_revision])
    if args.strict_embeddings:
        cmd.append("--strict-embeddings")
    for key in ("disable_score_gap", "disable_consistency", "disable_nondegradation", "disable_critical_step"):
        if getattr(args, key):
            cmd.append(f"--{key.replace('_', '-')}")
    return cmd


def validate_tau2_checkout(args, need_agent):
    root = Path(args.tau2_root).resolve()
    if not (root / "src/tau2/cli.py").is_file():
        raise ValueError(f"Missing tau2 CLI in {root}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != TAU2_REVISION:
        raise ValueError(f"Expected pinned tau2 revision {TAU2_REVISION}; got {revision}")
    if need_agent and "def create_fedskill_llm_agent(" not in (root / "src/tau2/agent/llm_agent.py").read_text():
        raise ValueError("Apply integrations/tau2/fedskill_tau2_integration.patch before skill evaluation.")
    args.tau2_root = str(root)


def main() -> None:
    import uuid
    args = parse_args()
    ids = task_ids(args)
    conditions = selected_conditions(args)
    need_skills = any(c != "baseline" for c in conditions)
    validate_tau2_checkout(args, need_skills)
    source, skills = source_files(args)
    external = source is not None or skills is not None
    if external and args.force_extract:
        raise ValueError("--force-extract cannot be combined with a fixed source library.")
    if need_skills and not external and not args.skill_model:
        raise ValueError("--skill-model / SKILLLADDER_MODEL is required to generate skills.")
    if external:
        for condition in conditions:
            if condition != "baseline":
                condition_candidates(condition, source, skills)
    parameters = public_parameters(args)
    identity = {
        **run_identity(args, ids),
        "effective_parameters": {k: v for k, v in parameters.items()
                                 if k not in {"dry_run", "force_runs", "force_extract", "output_dir"}},
        "models": {"consumer": args.agent_model, "simulator": args.user_model,
                   "skill_producer": args.skill_model, "fixed_source": args.source_model},
        "source_skills_digest": file_digest(skills) if skills else None,
        "source_l0_digest": file_digest(source / "level0_all.json") if source and (source / "level0_all.json").exists() else None,
    }
    output_dir = (REPO_ROOT / args.output_dir).resolve() / "runs" / fingerprint(identity)[:16]
    if args.force_runs or args.force_extract:
        output_dir /= f"attempt_{uuid.uuid4().hex[:12]}"
    prefix = run_prefix(args, ids)
    expected = len(ids) * args.num_trials
    skills_dir = output_dir / "skills_from_train"
    if args.dry_run:
        plan = {
            "protocol_version": PROTOCOL_VERSION, "task_ids": ids, "parameters": parameters,
            "output_dir": str(output_dir), "fixed_library": external,
            "generate_skills": need_skills and not external, "commands": [],
        }
        if need_skills and not external:
            plan["commands"].append({"stage": "source-baseline", "command": tau2_command(
                args, run_name=f"{prefix}_baseline", ids=ids, agent="llm_agent")})
            plan["commands"].append({"stage": "extract-evolve", "command": phase1_command(
                args, f"{prefix}_baseline", skills_dir, expected)})
        for condition in conditions:
            plan["commands"].append({
                "condition": condition,
                "command": tau2_command(args, run_name=f"{prefix}_{condition}", ids=ids,
                                       agent="llm_agent" if condition == "baseline" else "fedskill_llm_agent"),
                "source": str(skills) if skills else "generated from source-baseline",
                "eligible_skills": len(condition_candidates(condition, source, skills)) if external and condition != "baseline" else None,
                "injection_mode": "dynamic" if condition == "full-dynamic" else ("static" if condition != "baseline" else None),
            })
        print(json.dumps(plan, indent=2))
        return
    manifest_path = output_dir / "comparison_manifest.json"
    state = bind_state(manifest_path, identity)
    logs_dir = output_dir / "logs"
    source_run = None
    if "baseline" in conditions or (need_skills and not external):
        source_run = tau2_run(args, run_name=f"{prefix}_baseline", ids=ids, agent="llm_agent",
                              log_path=logs_dir / "baseline.log")
    if need_skills and not external:
        run_logged(phase1_command(args, source_run, skills_dir, expected), REPO_ROOT,
                   skill_env(args), logs_dir / "extract_evolve.log")
        source, skills = skills_dir, skills_dir / "evolved_all.json"
    runs = {}
    libraries = {}
    if "baseline" in conditions:
        runs["baseline"] = source_run
    for condition in conditions:
        if condition == "baseline":
            continue
        library = prepare_condition_library(condition, source, skills, output_dir)
        libraries[condition] = library
        runs[condition] = tau2_run(
            args, run_name=f"{prefix}_{condition}", ids=ids, agent="fedskill_llm_agent",
            skills_path=library, injection_mode="dynamic" if condition == "full-dynamic" else "static",
            log_path=logs_dir / f"{condition}.log",
        )
        state["runs"] = runs
        atomic_write_json(manifest_path, state)
    results = {"settings": {**parameters, "task_ids": ids, "run_names": runs,
                            "source_artifacts_dir": str(source), "output_dir": str(output_dir)}}
    results.update({condition: summarize_result(tau2_result_path(args, name)) for condition, name in runs.items()})
    if "full-static" in runs and "full-dynamic" in runs:
        results["skill_quality_update"] = update_skill_quality_from_tau2_runs(
            args, skills_path=libraries["full-static"], output_dir=output_dir,
            static_run=runs["full-static"], dynamic_run=runs["full-dynamic"],
            static_skill_log=tau2_skill_log_path(args, runs["full-static"]),
            dynamic_skill_log=tau2_skill_log_path(args, runs["full-dynamic"]),
        )
    write_summary(output_dir, results)
    state.update(status="complete", runs=runs, summary_digest=file_digest(output_dir / "summary.json"))
    atomic_write_json(manifest_path, state)
    print(f"[done] summary: {output_dir / 'summary.json'}; source artifacts: {source}")


if __name__ == "__main__":
    main()
