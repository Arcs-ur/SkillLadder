"""Non-Claude synthesis + quality-gate ablation for saved SkillLadder clusters.

This is stronger than `non_claude_judge_review.py`: it does not only rescore
saved Claude-synthesized children. It monkey-patches the evolver's LLM call so
that synthesis, consistency checking, scoring, and critical-step preservation
are all performed by a non-Claude model.

Default scope is airline R2/R3 because R1 parents are L0 skills with score=0,
so the non-degradation gate is not very informative there.

Examples:
  DEEPSEEK_API_KEY=... .venv/bin/python tools/non_claude_gate_ablation.py \
    --provider deepseek --rounds 2,3 --tasks 10 --run-name deepseek_R2-3

  ZAI_API_KEY=... .venv/bin/python tools/non_claude_gate_ablation.py \
    --provider zai --model glm-5.2 --rounds 2,3 --tasks 10 --run-name glm52_R2-3

Smoke test without API calls:
  .venv/bin/python tools/non_claude_gate_ablation.py --dry-run --limit 2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(TOOLS_DIR))

# `fedskill.server.evolver` imports `fedskill.utils.llm.llm_call`, whose real
# module eagerly initializes MSAL. For non-Claude replications we never want
# MSAL. Install a tiny stub before importing the evolver, then monkey-patch the
# evolver's module-level `llm_call` after the non-Claude client is created.
llm_stub = types.ModuleType("fedskill.utils.llm")


def _unpatched_llm_call(*_: Any, **__: Any) -> str:
    raise RuntimeError("non-Claude llm_call has not been installed yet")


llm_stub.llm_call = _unpatched_llm_call
sys.modules.setdefault("fedskill.utils.llm", llm_stub)

from fedskill.models.skill import Skill  # noqa: E402
import fedskill.server.evolver as evolver  # noqa: E402
from non_claude_judge_review import (  # noqa: E402
    create_client,
    resolve_provider,
    response_content,
)

DEFAULT_ARTIFACT_DIR = REPO_ROOT / "output" / "tau2_airline_train50" / "skills_from_train"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "non_claude_gate_ablation"

CONDITIONS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_score_gap": {"disable_score_gap": True},
    "no_consistency": {"disable_consistency": True},
    "no_min_score": {"min_score": 0.0},
    "no_nondegrad": {"disable_nondegradation": True},
    "no_critical": {"disable_critical_step": True},
}


def _skill_from_dict(data: dict[str, Any]) -> Skill:
    return Skill.from_dict(data)


def load_round_clusters(artifact_dir: Path, rounds: list[int], limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_id in rounds:
        path = artifact_dir / f"round{round_id}_clusters.json"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            raw_clusters = json.load(f)
        for cluster_idx, raw_cluster in enumerate(raw_clusters):
            if len(raw_cluster) < 2:
                continue
            cluster = [_skill_from_dict(item) for item in raw_cluster]
            best_parent = max(cluster, key=lambda s: float(s.score or 0.0))
            rows.append(
                {
                    "round": round_id,
                    "cluster_idx": cluster_idx,
                    "cluster": cluster,
                    "best_parent_score": float(best_parent.score or 0.0),
                    "best_parent_id": best_parent.id,
                    "parent_levels": sorted({int(s.level or 0) for s in cluster}),
                }
            )
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_task_descriptions(artifact_dir: Path, limit: int) -> list[str]:
    path = artifact_dir / "sessions.json"
    with path.open("r", encoding="utf-8") as f:
        sessions = json.load(f)
    tasks = [item.get("task_description", "") for item in sessions if item.get("task_description")]
    return tasks[:limit]


def sanitize_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def make_llm_call(
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
) -> Callable[[str, str, float | None], str]:
    def call(prompt: str, system_prompt: str = "", temperature: float | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        temp = default_temperature if temperature is None else temperature
        last_error: Exception | None = None
        for attempt in range(retries + 1):
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
                return response_content(response)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(sleep_s * (attempt + 1))
        raise RuntimeError(f"non-Claude LLM call failed after {retries + 1} attempts: {last_error}")

    return call


def load_done(csv_path: Path) -> set[tuple[int, int, str]]:
    done: set[tuple[int, int, str]] = set()
    if not csv_path.exists():
        return done
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add((int(row["round"]), int(row["cluster_idx"]), row["condition"]))
            except (KeyError, ValueError):
                continue
    return done


def append_csv(csv_path: Path, row: dict[str, Any]) -> None:
    fields = [
        "round",
        "cluster_idx",
        "cluster_size",
        "parent_levels",
        "condition",
        "accepted",
        "score",
        "best_parent_score",
        "margin",
        "wall_s",
        "evolved_id",
        "evolved_name",
        "error",
    ]
    new_file = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = sorted({(int(r["round"]), str(r["condition"])) for r in rows})
    for round_id, condition in keys:
        subset = [r for r in rows if int(r["round"]) == round_id and r["condition"] == condition]
        out[f"R{round_id}:{condition}"] = summarize_subset(subset)
    for condition in sorted({str(r["condition"]) for r in rows}):
        subset = [r for r in rows if r["condition"] == condition]
        out[f"all:{condition}"] = summarize_subset(subset)
    return out


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [r for r in rows if str(r.get("accepted")) == "True"]
    scores = [float(r["score"]) for r in accepted if r.get("score") not in ("", None)]
    margins = [float(r["margin"]) for r in accepted if r.get("margin") not in ("", None)]
    return {
        "n_clusters": len(rows),
        "n_accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / len(rows), 3) if rows else None,
        "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
        "mean_margin": round(sum(margins) / len(margins), 3) if margins else None,
        "median_margin": round(statistics.median(margins), 3) if margins else None,
        "frac_margin_ge_0": round(sum(1 for m in margins if m >= 0) / len(margins), 3) if margins else None,
        "frac_margin_ge_05": round(sum(1 for m in margins if m >= 0.5) / len(margins), 3) if margins else None,
        "n_errors": sum(1 for r in rows if r.get("error")),
    }


def read_existing_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
    parser.add_argument("--rounds", default="2,3", help="Comma-separated evolution rounds to rerun.")
    parser.add_argument("--conditions", default=",".join(CONDITIONS), help="Comma-separated conditions.")
    parser.add_argument("--limit", type=int, default=None, help="Limit total clusters across rounds for smoke tests.")
    parser.add_argument("--tasks", type=int, default=10, help="Number of task descriptions used by the score gate.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--reasoning-effort", default="max")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1, help="Parallel cluster-condition workers. Start with 2.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    rounds = [int(item) for item in args.rounds.split(",") if item.strip()]
    selected_conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = [item for item in selected_conditions if item not in CONDITIONS]
    if unknown:
        raise SystemExit(f"Unknown conditions: {unknown}. Valid: {sorted(CONDITIONS)}")

    clusters = load_round_clusters(artifact_dir, rounds, args.limit)
    tasks = load_task_descriptions(artifact_dir, args.tasks)
    print(f"[load] artifact_dir={artifact_dir}")
    print(f"[load] rounds={rounds}, clusters={len(clusters)}, tasks={len(tasks)}")
    for round_id in rounds:
        round_clusters = [item for item in clusters if item["round"] == round_id]
        nonzero = sum(1 for item in round_clusters if item["best_parent_score"] > 0)
        print(f"  R{round_id}: {len(round_clusters)} multi-clusters, best_parent_score>0 for {nonzero}")
    print(f"[conditions] {selected_conditions}")

    if args.dry_run:
        print("[dry-run] no model calls made")
        return

    api_key, base_url, model, backend = resolve_provider(args)
    client = create_client(api_key, base_url, backend, args.timeout)
    model_tag = sanitize_tag(model)
    run_name = sanitize_tag(args.run_name or f"{args.provider}_{model_tag}_R{'-'.join(map(str, rounds))}")
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_cluster.csv"
    jsonl_path = out_dir / "evolved_rows.jsonl"
    summary_path = out_dir / "summary.json"

    evolver.llm_call = make_llm_call(
        client,
        backend,
        model,
        timeout_s=args.timeout,
        default_temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        retries=args.retries,
        sleep_s=max(args.sleep, 1.0),
    )

    done = load_done(csv_path)
    total = len(clusters) * len(selected_conditions)
    print(f"[resume] {len(done)}/{total} existing rows")

    work_items: list[tuple[dict[str, Any], str]] = []
    for item in clusters:
        round_id = int(item["round"])
        cluster_idx = int(item["cluster_idx"])
        for condition in selected_conditions:
            key = (round_id, cluster_idx, condition)
            if key not in done:
                work_items.append((item, condition))

    def run_one(item: dict[str, Any], condition: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        round_id = int(item["round"])
        cluster_idx = int(item["cluster_idx"])
        cluster: list[Skill] = item["cluster"]
        best_parent_score = float(item["best_parent_score"])
        flags = dict(CONDITIONS[condition])
        row: dict[str, Any] = {
            "round": round_id,
            "cluster_idx": cluster_idx,
            "cluster_size": len(cluster),
            "parent_levels": "/".join(map(str, item["parent_levels"])),
            "condition": condition,
            "best_parent_score": round(best_parent_score, 4),
        }
        t1 = time.time()
        evolved_skill: Skill | None = None
        try:
            evolved_skill = evolver.evolve_skill_cluster(
                cluster,
                task_descriptions=tasks,
                **flags,
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        wall_s = time.time() - t1
        accepted = evolved_skill is not None
        row["accepted"] = accepted
        row["wall_s"] = round(wall_s, 2)
        evolved_payload: dict[str, Any] | None = None
        if accepted and evolved_skill is not None:
            score = float(evolved_skill.score or 0.0)
            row["score"] = round(score, 4)
            row["margin"] = round(score - best_parent_score, 4)
            row["evolved_id"] = evolved_skill.id
            row["evolved_name"] = evolved_skill.name
            evolved_payload = {**row, "evolved": evolved_skill.to_dict()}
        else:
            row.setdefault("score", "")
            row.setdefault("margin", "")
            row.setdefault("evolved_id", "")
            row.setdefault("evolved_name", "")
        return row, evolved_payload

    t0 = time.time()
    completed = len(done)

    def record_result(row: dict[str, Any], evolved_payload: dict[str, Any] | None) -> None:
        nonlocal completed
        if evolved_payload is not None:
            with jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(evolved_payload, ensure_ascii=False) + "\n")
        append_csv(csv_path, row)
        completed += 1
        elapsed = max(time.time() - t0, 1e-6)
        rate = completed / elapsed
        eta = (total - completed) / rate if rate > 0 else math.nan
        print(
            f"[{completed}/{total}] R{row['round']} cluster={row['cluster_idx']} cond={row['condition']} "
            f"accepted={row['accepted']} score={row.get('score') or '--'} "
            f"margin={row.get('margin') or '--'} wall={row.get('wall_s', 0):.1f}s ETA={eta:.0f}s",
            flush=True,
        )

    workers = max(1, int(args.workers or 1))
    print(f"[run] pending rows={len(work_items)}, workers={workers}")
    if workers == 1:
        for item, condition in work_items:
            row, evolved_payload = run_one(item, condition)
            record_result(row, evolved_payload)
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_key = {
                executor.submit(run_one, item, condition): (int(item["round"]), int(item["cluster_idx"]), condition)
                for item, condition in work_items
            }
            for future in as_completed(future_to_key):
                try:
                    row, evolved_payload = future.result()
                except Exception as exc:  # noqa: BLE001
                    round_id, cluster_idx, condition = future_to_key[future]
                    row = {
                        "round": round_id,
                        "cluster_idx": cluster_idx,
                        "cluster_size": "",
                        "parent_levels": "",
                        "condition": condition,
                        "accepted": False,
                        "score": "",
                        "best_parent_score": "",
                        "margin": "",
                        "wall_s": "",
                        "evolved_id": "",
                        "evolved_name": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    evolved_payload = None
                record_result(row, evolved_payload)

    rows = read_existing_rows(csv_path)
    summary = summarize(rows)
    summary["_meta"] = {
        "provider": args.provider,
        "model": model,
        "backend": backend,
        "artifact_dir": str(artifact_dir),
        "rounds": rounds,
        "conditions": selected_conditions,
        "tasks": args.tasks,
        "limit": args.limit,
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
