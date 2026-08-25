"""Measure provider-reported token usage for full-gate skill evolution.

The audit samples saved clusters, reruns synthesis and all enabled quality
gates, and records usage separately for synthesis, consistency, scoring, and
critical-step preservation. Outputs are isolated from the gate-ablation runs.

Example:
  DEEPSEEK_API_KEY=... .venv/bin/python tools/evolution_token_audit.py \
    --provider deepseek --rounds 2,3 --samples-per-round 5 --tasks 10
"""
from __future__ import annotations

import argparse
import contextvars
import csv
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(TOOLS_DIR))

# Importing this module installs the non-MSAL LLM stub before loading evolver.
from non_claude_gate_ablation import (  # noqa: E402
    create_client,
    evolver,
    load_round_clusters,
    load_task_descriptions,
    resolve_provider,
    response_content,
    sanitize_tag,
)

DEFAULT_ARTIFACT_DIR = REPO_ROOT / "output" / "tau2_airline_train50" / "skills_from_train"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "evolution_token_audit"

STAGES = {
    evolver.EVOLVE_SYSTEM: "synthesis",
    evolver.CONSISTENCY_SYSTEM: "consistency_gate",
    evolver.SCORE_SYSTEM: "score_gate",
    evolver.STEP_PRESERVATION_SYSTEM: "critical_step_gate",
}

CURRENT_RECORDS: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("evolution_token_records", default=None)
)


def usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(name, 0)
    else:
        value = getattr(usage, name, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 2),
        "median": percentile(values, 0.5),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def sample_clusters(
    clusters: list[dict[str, Any]],
    rounds: list[int],
    samples_per_round: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for round_id in rounds:
        candidates = [item for item in clusters if int(item["round"]) == round_id]
        rng = random.Random(seed + round_id)
        if len(candidates) > samples_per_round:
            candidates = rng.sample(candidates, samples_per_round)
        selected.extend(candidates)
    return selected


def make_recording_llm_call(
    client: Any,
    backend: str,
    model: str,
    *,
    timeout_s: float,
    default_temperature: float,
    max_tokens: int,
    thinking: str,
    reasoning_effort: str,
    retries: int,
    sleep_s: float,
):
    def call(prompt: str, system_prompt: str = "", temperature: float | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        temp = default_temperature if temperature is None else temperature
        stage = STAGES.get(system_prompt, "other")
        last_error: Exception | None = None

        for api_attempt in range(retries + 1):
            started = time.time()
            try:
                if backend == "zai":
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        thinking={"type": thinking},
                        reasoning_effort=reasoning_effort,
                        max_tokens=max_tokens,
                        temperature=temp,
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_tokens,
                        timeout=timeout_s,
                    )
                usage = getattr(response, "usage", None)
                prompt_tokens = usage_value(usage, "prompt_tokens")
                completion_tokens = usage_value(usage, "completion_tokens")
                total_tokens = usage_value(usage, "total_tokens") or prompt_tokens + completion_tokens
                records = CURRENT_RECORDS.get()
                if records is None:
                    raise RuntimeError("Token recorder is not bound to the current worker")
                records.append(
                    {
                        "stage": stage,
                        "api_attempt": api_attempt + 1,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "wall_s": round(time.time() - started, 3),
                    }
                )
                return response_content(response)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if api_attempt >= retries:
                    break
                time.sleep(sleep_s * (api_attempt + 1))
        raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_error}")

    return call


def aggregate(rows: list[dict[str, Any]], call_rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [int(row["total_tokens"]) for row in rows]
    prompt_totals = [int(row["prompt_tokens"]) for row in rows]
    completion_totals = [int(row["completion_tokens"]) for row in rows]
    call_counts = [int(row["n_calls"]) for row in rows]
    accepted = [row for row in rows if row["accepted"]]

    by_stage: dict[str, dict[str, Any]] = {}
    for stage in sorted({str(row["stage"]) for row in call_rows}):
        subset = [row for row in call_rows if row["stage"] == stage]
        by_stage[stage] = {
            "calls": len(subset),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in subset),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in subset),
            "total_tokens": sum(int(row["total_tokens"]) for row in subset),
            "mean_tokens_per_call": round(
                sum(int(row["total_tokens"]) for row in subset) / len(subset), 2
            ),
        }

    return {
        "n_clusters": len(rows),
        "n_accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / len(rows), 4) if rows else None,
        "provider_reported_usage": {
            "prompt_tokens_per_cluster": stats(prompt_totals),
            "completion_tokens_per_cluster": stats(completion_totals),
            "total_tokens_per_cluster": stats(totals),
            "calls_per_cluster": stats(call_counts),
            "grand_prompt_tokens": sum(prompt_totals),
            "grand_completion_tokens": sum(completion_totals),
            "grand_total_tokens": sum(totals),
        },
        "by_stage": by_stage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["deepseek", "glm", "zai", "openai", "generic"], default="deepseek")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--rounds", default="2,3")
    parser.add_argument("--samples-per-round", type=int, default=5)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--reasoning-effort", default="max")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent clusters; calls within each cluster remain sequential.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    rounds = [int(item) for item in args.rounds.split(",") if item.strip()]
    all_clusters = load_round_clusters(artifact_dir, rounds, None)
    clusters = sample_clusters(all_clusters, rounds, args.samples_per_round, args.seed)
    tasks = load_task_descriptions(artifact_dir, args.tasks)
    print(f"[load] artifact_dir={artifact_dir}")
    print(f"[load] rounds={rounds}, sampled_clusters={len(clusters)}, tasks={len(tasks)}")
    print(f"[sample] {Counter(int(item['round']) for item in clusters)}")

    if args.dry_run:
        for item in clusters:
            print(f"  R{item['round']} cluster={item['cluster_idx']} size={len(item['cluster'])}")
        return

    api_key, base_url, model, backend = resolve_provider(args)
    client = create_client(api_key, base_url, backend, args.timeout)
    run_name = sanitize_tag(
        args.run_name or f"{args.provider}_{model}_airline_R{'-'.join(map(str, rounds))}_n{len(clusters)}"
    )
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    evolver.llm_call = make_recording_llm_call(
        client,
        backend,
        model,
        timeout_s=args.timeout,
        default_temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        retries=args.retries,
        sleep_s=args.sleep,
    )

    def run_one(sample_idx: int, item: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        context_token = CURRENT_RECORDS.set(records)
        started = time.time()
        evolved = None
        error = ""
        try:
            evolved = evolver.evolve_skill_cluster(item["cluster"], task_descriptions=tasks)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        finally:
            CURRENT_RECORDS.reset(context_token)

        prompt_tokens = sum(int(row["prompt_tokens"]) for row in records)
        completion_tokens = sum(int(row["completion_tokens"]) for row in records)
        total_tokens = sum(int(row["total_tokens"]) for row in records)
        stage_counts = Counter(str(row["stage"]) for row in records)
        cluster_row = {
            "sample_idx": sample_idx,
            "round": int(item["round"]),
            "cluster_idx": int(item["cluster_idx"]),
            "cluster_size": len(item["cluster"]),
            "accepted": evolved is not None,
            "score": round(float(evolved.score or 0.0), 4) if evolved is not None else "",
            "best_parent_score": round(float(item["best_parent_score"]), 4),
            "n_calls": len(records),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "stage_calls": dict(stage_counts),
            "wall_s": round(time.time() - started, 2),
            "error": error,
        }
        call_rows: list[dict[str, Any]] = []
        for call_idx, call_row in enumerate(records, start=1):
            call_rows.append(
                {
                    "sample_idx": sample_idx,
                    "round": int(item["round"]),
                    "cluster_idx": int(item["cluster_idx"]),
                    "call_idx": call_idx,
                    **call_row,
                }
            )
        return cluster_row, call_rows

    workers = max(1, int(args.workers))
    print(f"[run] workers={workers}", flush=True)
    cluster_rows: list[dict[str, Any]] = []
    all_call_rows: list[dict[str, Any]] = []
    completed = 0

    def record_result(cluster_row: dict[str, Any], call_rows: list[dict[str, Any]]) -> None:
        nonlocal completed
        cluster_rows.append(cluster_row)
        all_call_rows.extend(call_rows)
        completed += 1
        print(
            f"[{completed}/{len(clusters)}] R{cluster_row['round']} cluster={cluster_row['cluster_idx']} "
            f"accepted={cluster_row['accepted']} calls={cluster_row['n_calls']} "
            f"tokens={cluster_row['total_tokens']} "
            f"wall={cluster_row['wall_s']:.1f}s",
            flush=True,
        )

    if workers == 1:
        for sample_idx, item in enumerate(clusters, start=1):
            record_result(*run_one(sample_idx, item))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_one, sample_idx, item): sample_idx
                for sample_idx, item in enumerate(clusters, start=1)
            }
            for future in as_completed(futures):
                record_result(*future.result())

    cluster_rows.sort(key=lambda row: int(row["sample_idx"]))
    all_call_rows.sort(key=lambda row: (int(row["sample_idx"]), int(row["call_idx"])))

    with (out_dir / "per_cluster.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(cluster_rows[0]) if cluster_rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(cluster_rows)
    with (out_dir / "per_call.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(all_call_rows[0]) if all_call_rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(all_call_rows)

    summary = aggregate(cluster_rows, all_call_rows)
    summary["_meta"] = {
        "provider": args.provider,
        "model": model,
        "artifact_dir": str(artifact_dir),
        "rounds": rounds,
        "samples_per_round": args.samples_per_round,
        "tasks": args.tasks,
        "seed": args.seed,
        "workers": workers,
        "sampling": "deterministic uniform sample within each round",
        "usage_source": "provider-reported response.usage",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
