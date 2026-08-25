"""Non-Claude judge review for saved SkillLadder artifacts.

This script does not re-run synthesis or agent evaluation. It reloads accepted
L1/L2/L3 skills from saved artifacts, pairs each child with its strongest
Claude-scored parent, and asks a non-Claude OpenAI-compatible judge to score
both skills on the same task-description subset.

Example:
  DEEPSEEK_API_KEY=... .venv/bin/python tools/non_claude_judge_review.py \
    --provider deepseek --limit 40 --levels 2,3

  ZAI_API_KEY=... .venv/bin/python tools/non_claude_judge_review.py \
    --provider zai --model glm-5.2 --limit 40 --levels 2,3

  OPENAI_API_KEY=... .venv/bin/python tools/non_claude_judge_review.py \
    --provider openai --model gpt-4.1-mini --limit 40 --levels 2,3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from openai import OpenAI  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_ARTIFACT_DIRS = [
    OUTPUT_ROOT / "tau2_airline_train50" / "skills_from_train",
    OUTPUT_ROOT / "tau2_banking_knowledge_all97_max50" / "skills_from_train",
    OUTPUT_ROOT / "tau2_retail_all114" / "skills_from_train",
    OUTPUT_ROOT / "tau2_telecom_remaining200_max50" / "skills_from_train",
]

PROVIDERS = {
    "deepseek": {
        "backend": "openai",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
    },
    "glm": {
        "backend": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPUAI_API_KEY",
        "model": "glm-4-plus",
    },
    "zai": {
        "backend": "zai",
        "base_url": "",
        "api_key_envs": ["ZAI_API_KEY", "ZHIPUAI_API_KEY"],
        "model": "glm-5.2",
    },
    "openai": {
        "backend": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
    },
    "generic": {
        "backend": "openai",
        "base_url": "",
        "api_key_env": "JUDGE_API_KEY",
        "model": "",
    },
}

SYSTEM_PROMPT = """You are an independent skill-quality evaluator.
You compare two procedural agent skills, A and B, against the same set of task
descriptions. Score each skill independently for each task:

- relevance (0-10): how related the skill's strategy is to the task.
- applicability (0-10): whether an agent would actually use this skill on the task.

Return JSON only:
{
  "skill_a": [{"relevance": int, "applicability": int}, ...],
  "skill_b": [{"relevance": int, "applicability": int}, ...],
  "preference": "A" or "B" or "tie",
  "reason": "one concise sentence"
}

Use one entry per task and preserve task order. Do not favor a skill because it
is longer; prefer concrete, broadly reusable procedures that preserve useful
operational steps."""

USER_PROMPT = """### Skill A: child evolved skill
{child_json}

### Skill B: strongest parent skill
{parent_json}

### Tasks
{tasks_text}
"""


@dataclass
class Pair:
    domain: str
    level: int
    round_id: int
    child: dict[str, Any]
    parent: dict[str, Any]
    claude_child_score: float
    claude_parent_score: float


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "skills" in data:
        data = data["skills"]
    return [x for x in data if isinstance(x, dict)]


def compact_skill(skill: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": skill.get("name", ""),
        "description": skill.get("description", ""),
        "when_to_use": skill.get("when_to_use", ""),
        "procedure": skill.get("procedure", ""),
        "examples": (skill.get("examples") or [])[:2],
        "level": skill.get("level", 0),
        "score": skill.get("score", 0),
    }


def score_from_per_task(rows: list[dict[str, Any]]) -> float:
    applicable = []
    for row in rows:
        try:
            app = float(row.get("applicability", 0))
            rel = float(row.get("relevance", 0))
        except (TypeError, ValueError):
            continue
        if app >= 3:
            applicable.append(rel)
    if not rows or not applicable:
        return 0.0
    return round((len(applicable) / len(rows)) * (sum(applicable) / len(applicable)), 3)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def load_tasks(artifact_dir: Path, tasks_per_domain: int, seed: int) -> list[str]:
    sessions = load_json(artifact_dir / "sessions.json")
    tasks = [s.get("task_description", "") for s in sessions if s.get("task_description")]
    rng = random.Random(seed)
    if len(tasks) > tasks_per_domain:
        tasks = rng.sample(tasks, tasks_per_domain)
    return tasks


def build_pairs(artifact_dirs: list[Path], levels: set[int], seed: int) -> list[Pair]:
    pairs: list[Pair] = []
    for artifact_dir in artifact_dirs:
        if not artifact_dir.exists():
            continue
        domain = artifact_dir.parts[-2]
        id_to_skill: dict[str, dict[str, Any]] = {}
        for name in ("level0_all.json", "round1_skills.json", "round2_skills.json", "round3_skills.json"):
            for skill in load_json(artifact_dir / name):
                sid = skill.get("id")
                if sid:
                    id_to_skill[sid] = skill

        for round_id in (1, 2, 3):
            for child in load_json(artifact_dir / f"round{round_id}_skills.json"):
                level = int(child.get("level") or 0)
                if level != round_id or level not in levels:
                    continue
                parent_ids = child.get("parent_ids") or []
                parents = [id_to_skill[pid] for pid in parent_ids if pid in id_to_skill]
                if not parents:
                    continue
                parent = max(parents, key=lambda s: float(s.get("score") or 0.0))
                pairs.append(
                    Pair(
                        domain=domain,
                        level=level,
                        round_id=round_id,
                        child=child,
                        parent=parent,
                        claude_child_score=float(child.get("score") or 0.0),
                        claude_parent_score=float(parent.get("score") or 0.0),
                    )
                )
    random.Random(seed).shuffle(pairs)
    return pairs


def resolve_provider(args: argparse.Namespace) -> tuple[str, str, str, str]:
    spec = PROVIDERS[args.provider]
    backend = spec["backend"]
    if args.api_key_env:
        api_key_envs = [args.api_key_env]
    else:
        api_key_envs = spec.get("api_key_envs") or [spec["api_key_env"]]
    api_key = args.api_key
    for api_key_env in api_key_envs:
        if api_key:
            break
        api_key = os.environ.get(api_key_env, "")
    base_url = args.base_url or spec["base_url"]
    model = args.model or spec["model"]
    if not api_key:
        raise SystemExit(f"Missing API key. Set one of {api_key_envs} or pass --api-key.")
    if backend == "openai" and not base_url:
        raise SystemExit("Missing base URL. Pass --base-url for provider=generic.")
    if not model:
        raise SystemExit("Missing model. Pass --model for provider=generic.")
    return api_key, base_url, model, backend


def create_client(api_key: str, base_url: str, backend: str, timeout_s: float) -> Any:
    if backend == "zai":
        from zai import ZhipuAiClient
        return ZhipuAiClient(api_key=api_key, base_url=base_url or None, timeout=timeout_s)
    return OpenAI(api_key=api_key, base_url=base_url)


def response_content(response: Any) -> str:
    message = response.choices[0].message
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def judge_pair(
    client: Any,
    backend: str,
    model: str,
    pair: Pair,
    tasks: list[str],
    timeout_s: float,
    temperature: float,
    max_tokens: int,
    thinking: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    tasks_text = "\n".join(f"{i + 1}. {task}" for i, task in enumerate(tasks))
    prompt = USER_PROMPT.format(
        child_json=json.dumps(compact_skill(pair.child), ensure_ascii=False, indent=2),
        parent_json=json.dumps(compact_skill(pair.parent), ensure_ascii=False, indent=2),
        tasks_text=tasks_text,
    )
    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
    ]
    if backend == "zai":
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            thinking={"type": thinking},
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout_s,
        )
    text = response_content(response)
    parsed = extract_json(text)
    child_rows = parsed.get("skill_a") or []
    parent_rows = parsed.get("skill_b") or []
    child_score = score_from_per_task(child_rows)
    parent_score = score_from_per_task(parent_rows)
    return {
        "child_score": child_score,
        "parent_score": parent_score,
        "margin": round(child_score - parent_score, 3),
        "preference": parsed.get("preference", ""),
        "reason": parsed.get("reason", ""),
        "raw": parsed,
    }


def _summary_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(r["judge_margin"]) for r in rows]
    if not rows:
        return {"n": 0}
    wins = sum(1 for m in margins if m >= 0)
    strict = sum(1 for m in margins if m > 0)
    return {
        "n": len(rows),
        "mean_margin": round(sum(margins) / len(margins), 3),
        "median_margin": round(statistics.median(margins), 3),
        "frac_child_ge_parent": round(wins / len(rows), 3),
        "frac_child_gt_parent": round(strict / len(rows), 3),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summary_stats(rows)
    if not rows:
        return summary
    summary.update({
        "by_level": {
            str(level): _summary_stats([r for r in rows if int(r["level"]) == level])
            for level in sorted({int(r["level"]) for r in rows})
        },
        "by_domain": {
            domain: _summary_stats([r for r in rows if r["domain"] == domain])
            for domain in sorted({r["domain"] for r in rows})
        },
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="deepseek")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--levels", default="2,3", help="Comma-separated skill levels to review.")
    parser.add_argument("--run-name", default="", help="Optional filename tag; defaults to a level tag such as L2-3.")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--tasks-per-domain", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--reasoning-effort", default="max")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "non_claude_judge_review"))
    args = parser.parse_args()

    levels = {int(x) for x in args.levels.split(",") if x.strip()}
    pairs = build_pairs(DEFAULT_ARTIFACT_DIRS, levels=levels, seed=args.seed)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[load] sampled {len(pairs)} child-parent pairs for levels={sorted(levels)}")
    for level in sorted(levels):
        print(f"  level {level}: {sum(1 for p in pairs if p.level == level)}")
    for domain in sorted({p.domain for p in pairs}):
        print(f"  {domain}: {sum(1 for p in pairs if p.domain == domain)}")

    tasks_by_domain = {
        artifact_dir.parts[-2]: load_tasks(artifact_dir, args.tasks_per_domain, args.seed)
        for artifact_dir in DEFAULT_ARTIFACT_DIRS
        if artifact_dir.exists()
    }
    if args.dry_run:
        print("[dry-run] no model calls made")
        return

    api_key, base_url, model, backend = resolve_provider(args)
    client = create_client(api_key, base_url, backend, args.timeout)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = model.replace("/", "_").replace(":", "_")
    level_tag = args.run_name or ("L" + "-".join(str(level) for level in sorted(levels)))
    level_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", level_tag)
    output_prefix = f"{args.provider}_{model_tag}_{level_tag}"
    jsonl_path = out_dir / f"{output_prefix}_rows.jsonl"
    csv_path = out_dir / f"{output_prefix}_rows.csv"
    summary_path = out_dir / f"{output_prefix}_summary.json"

    done: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add(row["child_id"])

    csv_exists = csv_path.exists()
    rows: list[dict[str, Any]] = []
    if csv_exists:
        with csv_path.open("r", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))

    with jsonl_path.open("a", encoding="utf-8") as jf, csv_path.open("a", encoding="utf-8", newline="") as cf:
        fieldnames = [
            "domain", "level", "child_id", "parent_id",
            "claude_child_score", "claude_parent_score",
            "judge_child_score", "judge_parent_score", "judge_margin",
            "preference", "reason",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        for idx, pair in enumerate(pairs, start=1):
            child_id = pair.child.get("id", "")
            if child_id in done:
                continue
            tasks = tasks_by_domain.get(pair.domain, [])
            try:
                result = judge_pair(
                    client,
                    backend,
                    model,
                    pair,
                    tasks,
                    args.timeout,
                    args.temperature,
                    args.max_tokens,
                    args.thinking,
                    args.reasoning_effort,
                )
                row = {
                    "domain": pair.domain,
                    "level": pair.level,
                    "child_id": child_id,
                    "parent_id": pair.parent.get("id", ""),
                    "claude_child_score": pair.claude_child_score,
                    "claude_parent_score": pair.claude_parent_score,
                    "judge_child_score": result["child_score"],
                    "judge_parent_score": result["parent_score"],
                    "judge_margin": result["margin"],
                    "preference": result["preference"],
                    "reason": result["reason"],
                }
                writer.writerow(row)
                cf.flush()
                jf.write(json.dumps({**row, "raw": result["raw"]}, ensure_ascii=False) + "\n")
                jf.flush()
                rows.append(row)
                print(
                    f"[{idx}/{len(pairs)}] {pair.domain} L{pair.level} "
                    f"margin={result['margin']:+.3f} child={result['child_score']:.3f} "
                    f"parent={result['parent_score']:.3f} pref={result['preference']}"
                )
            except Exception as exc:
                print(f"[error] {pair.domain} child={child_id}: {exc}")
            if args.sleep:
                time.sleep(args.sleep)

    summary = summarize(rows)
    summary.update({
        "provider": args.provider,
        "model": model,
        "backend": backend,
        "levels": sorted(levels),
        "tasks_per_domain": args.tasks_per_domain,
        "seed": args.seed,
        "rows_csv": str(csv_path),
        "rows_jsonl": str(jsonl_path),
    })
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
