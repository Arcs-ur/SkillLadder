"""Paper configuration: explicit CLI > YAML > compatible defaults."""

from __future__ import annotations

import argparse
import os
import math
import sys
from pathlib import Path

import yaml
from fedskill.utils.run_manifest import public_endpoint


CONFIG_FIELDS = {
    "benchmark": {
        "num_trials": "num_trials", "max_steps": "max_steps", "max_retries": "max_retries",
        "agent_temperature": "agent_temperature", "user_temperature": "user_temperature", "seed": "seed",
    },
    "skill_evolution": {
        "rounds": "evolve_rounds", "clustering_strategy": "clustering_strategy",
        "embedding_model": "embedding_model", "embedding_revision": "embedding_revision",
        "distance_threshold": "distance_threshold", "max_cluster_size": "max_cluster_size",
        "min_cluster_size": "min_cluster_size", "minimum_judge_score": "minimum_judge_score",
        "score_gap_demotion_ratio": "score_gap_ratio", "max_synthesis_retries": "max_synthesis_retries",
        "extraction_temperature": "extraction_temperature",
        "synthesis_temperature": "synthesis_temperature", "gate_temperature": "gate_temperature",
        "strict_embeddings": "strict_embeddings",
    },
    "retrieval": {
        "static_top_k": "max_static_skills", "dynamic_top_k": "dynamic_top_k",
        "retrieval_candidates": "retrieval_candidates", "librarian_max_rounds": "librarian_max_rounds",
        "minimum_token_overlap": "skill_min_overlap",
    },
    "quality": {
        "alpha": "quality_alpha", "retire_quality_threshold": "retire_quality_threshold",
        "retire_min_injections": "retire_min_injections",
    },
    "models": {"agent": "agent_model", "user": "user_model", "skill": "skill_model"},
}


def add_evolution_args(parser):
    parser.add_argument("--skill-model", default=os.getenv("SKILLLADDER_MODEL") or os.getenv("OPENAI_MODEL"))
    parser.add_argument("--skill-base-url", default=os.getenv("SKILLLADDER_BASE_URL") or os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--skill-api-key", default=os.getenv("SKILLLADDER_API_KEY") or os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--clustering-strategy", default="hybrid",
                        choices=["llm", "embedding", "tag", "hybrid", "reverse_hybrid", "random"])
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--distance-threshold", type=float, default=0.5)
    parser.add_argument("--max-cluster-size", type=int, default=8)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--minimum-judge-score", type=float, default=1.0)
    parser.add_argument("--score-gap-ratio", type=float, default=0.3)
    parser.add_argument("--max-synthesis-retries", type=int, default=2)
    parser.add_argument("--extraction-temperature", type=float, default=0.4)
    parser.add_argument("--synthesis-temperature", type=float, default=0.5)
    parser.add_argument("--gate-temperature", type=float, default=0.2)
    parser.add_argument("--strict-embeddings", action=argparse.BooleanOptionalAction, default=False)
    for gate in ("score-gap", "consistency", "nondegradation", "critical-step"):
        parser.add_argument(f"--disable-{gate}", action="store_true")


def parse_config_args(parser, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    bootstrap.add_argument("--domain", default="airline")
    initial, _ = bootstrap.parse_known_args(argv)
    if initial.config:
        path = Path(initial.config).resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config must be a YAML mapping.")
        destinations = {a.dest for a in parser._actions}
        defaults = {}
        for section, fields in CONFIG_FIELDS.items():
            values = data.get(section, {})
            if not isinstance(values, dict):
                raise ValueError(f"Config section {section} must be a mapping.")
            for name, dest in fields.items():
                if dest == "max_static_skills" and dest not in destinations and "max_skills" in destinations:
                    dest = "max_skills"
                if name in values and dest in destinations:
                    defaults[dest] = values[name]
        split = data.get("task_splits", {}).get(initial.domain)
        task_set = data.get("task_sets", {}).get(initial.domain)
        if task_set and "task_set_name" in destinations:
            defaults["task_set_name"] = task_set
        split_name = data.get("task_split_names", {}).get(initial.domain)
        if split_name and "task_split_name" in destinations:
            defaults["task_split_name"] = split_name
        explicit = {a.split("=", 1)[0] for a in argv if a.startswith("--")}
        if split and "task_ids_file" in destinations and not explicit.intersection(
            {"--task-start", "--task-count", "--task-ids", "--task-ids-file"}
        ):
            root = path.parent.parent if path.parent.name == "configs" else path.parent
            defaults["task_ids_file"] = str((root / split).resolve())
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    validate_parameters(args)
    return args


def validate_parameters(args):
    positive = ("num_trials", "max_steps", "max_concurrency", "task_count", "max_cluster_size",
                "min_cluster_size", "max_synthesis_retries", "retrieval_candidates", "librarian_max_rounds")
    for key in positive:
        if hasattr(args, key) and (type(getattr(args, key)) is not int or getattr(args, key) < 1):
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("evolve_rounds", "max_static_skills", "max_skills", "dynamic_top_k", "max_retries",
                "skill_min_overlap", "retire_min_injections"):
        if hasattr(args, key) and (type(getattr(args, key)) is not int or getattr(args, key) < 0):
            raise ValueError(f"{key} must be a nonnegative integer.")
    for key in ("quality_alpha", "retire_quality_threshold", "score_gap_ratio"):
        if hasattr(args, key) and not 0 <= getattr(args, key) <= 1:
            raise ValueError(f"{key} must be in [0, 1].")
    if hasattr(args, "min_cluster_size") and args.min_cluster_size > args.max_cluster_size:
        raise ValueError("min_cluster_size cannot exceed max_cluster_size.")
    if hasattr(args, "strict_embeddings") and type(args.strict_embeddings) is not bool:
        raise ValueError("strict_embeddings must be boolean.")
    for key in ("agent_temperature", "user_temperature", "extraction_temperature",
                "synthesis_temperature", "gate_temperature", "distance_threshold", "minimum_judge_score"):
        if hasattr(args, key):
            value = getattr(args, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number.")


def evolution_options(args):
    return {
        "min_score": args.minimum_judge_score, "max_retries": args.max_synthesis_retries,
        "score_gap_ratio": args.score_gap_ratio, "synthesis_temperature": args.synthesis_temperature,
        "gate_temperature": args.gate_temperature,
    }


def clustering_options(args):
    if args.clustering_strategy in {"embedding", "hybrid", "reverse_hybrid"}:
        return dict(distance_threshold=args.distance_threshold, max_cluster_size=args.max_cluster_size,
                    embedding_model=args.embedding_model, embedding_revision=args.embedding_revision,
                    strict_embeddings=args.strict_embeddings, seed=args.seed,
                    min_cluster_size=args.min_cluster_size)
    if args.clustering_strategy == "random":
        return dict(min_size=args.min_cluster_size, max_size=args.max_cluster_size, seed=args.seed,
                    min_cluster_size=args.min_cluster_size)
    if args.clustering_strategy == "tag":
        return dict(max_cluster_size=args.max_cluster_size, min_cluster_size=args.min_cluster_size)
    return dict(min_cluster_size=args.min_cluster_size)


def skill_env(args):
    env = os.environ.copy()
    if args.skill_model:
        env["SKILLLADDER_MODEL"] = args.skill_model
    if args.skill_base_url:
        env["SKILLLADDER_BASE_URL"] = args.skill_base_url
    if args.skill_api_key:
        env["SKILLLADDER_API_KEY"] = args.skill_api_key
    return env


def public_parameters(args):
    values = {k: v for k, v in vars(args).items() if not k.endswith("api_key") and not k.startswith("_")}
    for key in ("openai_base_url", "skill_base_url"):
        if key in values:
            values[key] = public_endpoint(values[key])
    return values
