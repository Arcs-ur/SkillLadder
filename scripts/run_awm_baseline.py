"""AWM success-only original L0, with the same validated tau2 run identities."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from build_awm_skills import success_only_l0
import run_tau2_comparison as comparison
from fedskill.utils.run_manifest import atomic_write_json, file_digest, bind_state
from fedskill.utils.config import parse_config_args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--tau2-root", default=os.getenv("TAU2_ROOT"))
    parser.add_argument("--train-dir", required=True,
                        help="Directory with skills_from_train/{sessions.json,level0_all.json}")
    parser.add_argument("--skills-out", required=True)
    parser.add_argument("--run-name", required=True, help="Human-readable prefix; fingerprint is appended.")
    parser.add_argument("--task-split-name", default="base")
    parser.add_argument("--task-set-name")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=50)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--task-ids-file")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--config")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--strict-embeddings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-temperature", type=float, default=0.2)
    parser.add_argument("--user-temperature", type=float, default=0.2)
    parser.add_argument("--agent-model", default=os.getenv("SKILLLADDER_AGENT_MODEL"))
    parser.add_argument("--user-model", default=os.getenv("SKILLLADDER_USER_MODEL"))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--max-skills", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parse_config_args(parser)
    for key in ("tau2_root", "agent_model", "user_model"):
        if not getattr(args, key):
            parser.error(f"--{key.replace('_', '-')} or its environment variable is required")
    return args


def task_ids(args):
    return comparison.task_ids(args)


def build_skills(args):
    candidates = success_only_l0(Path(args.train_dir))
    path = Path(args.skills_out)
    if not candidates:
        raise ValueError("No original success-only L0 skills in source artifacts.")
    # Refuse replacing a different baseline pool at the same destination.
    bind_state(path.with_suffix(".manifest.json"), {"candidate_pool": candidates})
    atomic_write_json(path, candidates)
    return path, len(candidates)


def tau2_result_path(args, run_name):
    return comparison.tau2_result_path(args, run_name)


def is_complete(args, run_name, expected):
    return comparison.is_complete(args, run_name, expected)


def make_env(args, skills_path, log_path):
    args.max_static_skills = args.max_skills
    args.score_mode = "embedding"
    return comparison.make_env(args, skills_path=skills_path, injection_mode="static", skill_log_path=log_path)


def run_tau2(args, skills_path, ids, log_path):
    args.max_static_skills = args.max_skills
    args.score_mode = "embedding"
    args.force_runs = args.force
    return comparison.tau2_run(
        args, run_name=args.run_name, ids=ids, agent="fedskill_llm_agent",
        injection_mode="static", skills_path=skills_path,
        log_path=log_path or Path(args.train_dir) / "awm.log",
    )


def summarize(args, run_name):
    result = comparison.summarize_result(tau2_result_path(args, run_name))
    result["reward_mean"] = result.get("avg_reward")
    return result


def main():
    args = parse_args()
    ids = task_ids(args)
    comparison.validate_tau2_checkout(args, need_agent=True)
    if args.dry_run:
        candidates = success_only_l0(Path(args.train_dir))
        if not candidates:
            raise ValueError("No original success-only L0 skills.")
        print({"protocol": "corrected-v1", "condition": "awm-static", "task_ids": ids,
               "success_only_l0": len(candidates), "skills_out": args.skills_out,
               "agent_model": args.agent_model, "user_model": args.user_model,
               "retrieval": "MiniLM cosine", "tau2_root": args.tau2_root})
        return
    skills_path, n_skills = build_skills(args)
    log_dir = Path(args.log_dir) if args.log_dir else Path("output/baselines_awm/logs") / args.domain
    run_name = run_tau2(args, skills_path, ids, log_dir / f"{args.run_name}.log")
    result = {"domain": args.domain, "run": run_name, "num_skills": n_skills,
              "skills_digest": file_digest(skills_path), **summarize(args, run_name)}
    atomic_write_json(log_dir / f"{run_name}.summary.json", result)
    print(result)


if __name__ == "__main__":
    main()
