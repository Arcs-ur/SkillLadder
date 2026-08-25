"""Hard-negative source-attribution sensitivity audit.

This script extends the original five-way SA audit with:
  * paired 5-way and 10-way lineups;
  * same-domain, shared-tool, embedding-nearest hard negatives;
  * repeated lineups per skill;
  * skill-level bootstrap confidence intervals; and
  * normalized advantage (SA - 1/m) / (1 - 1/m).

The 5-way lineup is nested inside the corresponding 10-way lineup for each
(skill, repeat), so changing m does not also change the first four negatives.
Repeated lineups are treated as repeated measurements: confidence intervals
bootstrap skills, not individual model calls.

Examples:
  ZAI_API_KEY=... .venv/bin/python tools/source_attribution_sensitivity.py \
    --provider zai --model glm-5.2 --levels 3 --repeats 5

  DEEPSEEK_API_KEY=... .venv/bin/python tools/source_attribution_sensitivity.py \
    --provider deepseek --levels 3 --repeats 5

Use the lexical backend only for local plumbing checks:
  .venv/bin/python tools/source_attribution_sensitivity.py \
    --embedding-backend lexical --levels 3 --limit 2 --repeats 1 --dry-run
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
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
        "api_key_envs": ["DEEPSEEK_API_KEY"],
        "model": "deepseek-chat",
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
        "api_key_envs": ["OPENAI_API_KEY"],
        "model": "gpt-4.1-mini",
    },
    "generic": {
        "backend": "openai",
        "base_url": "",
        "api_key_envs": ["JUDGE_API_KEY"],
        "model": "",
    },
}

SYSTEM_PROMPT = """You are auditing source linkability of a sanitized agent skill.
You will receive one skill and a numbered lineup of candidate raw execution
traces. Exactly one candidate was used, directly or through its descendants, to
produce the skill. The other candidates are deliberately difficult same-domain,
same-tool alternatives.

Identify the most likely source. Return JSON only:
{"choice": <1-indexed integer>, "confidence": <number from 0 to 1>,
 "reason": "one concise sentence"}"""

USER_PROMPT = """### Sanitized skill (level {level})
{skill_text}

### Candidate raw traces
{traces_text}

Which candidate most likely contributed to the skill?"""


@dataclass
class DomainData:
    name: str
    artifact_dir: Path
    skills: dict[str, dict[str, Any]]
    targets: list[dict[str, Any]]
    sessions: dict[str, dict[str, Any]]
    task_ids: list[str]
    task_texts: list[str]
    embeddings: np.ndarray | None = None


@dataclass
class LineupPlan:
    key: str
    domain: str
    level: int
    skill_id: str
    repeat: int
    m: int
    true_task_id: str
    lineup: list[str]
    true_position: int
    negative_similarities: list[float]
    negative_tool_jaccards: list[float]


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "skills" in data:
        data = data["skills"]
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def task_type_text(session: dict[str, Any]) -> str:
    """Extract the user request without the verbose evaluation criteria."""
    raw = str(session.get("task_description") or "")
    marker = "user_scenario:"
    end_marker = "\nevaluation_criteria:"
    if marker in raw:
        fragment = raw.split(marker, 1)[1]
        if end_marker in fragment:
            fragment = fragment.split(end_marker, 1)[0]
        try:
            scenario = ast.literal_eval(fragment.strip())
            instructions = scenario.get("instructions") or {}
            parts = [
                instructions.get("domain"),
                instructions.get("reason_for_call"),
                instructions.get("task_instructions"),
            ]
            text = " ".join(str(part) for part in parts if part)
            if text:
                return text
        except (SyntaxError, ValueError, AttributeError):
            pass
    return raw[:4000]


def trace_text(session: dict[str, Any], max_chars: int) -> str:
    traces = session.get("traces") or []
    if isinstance(traces, list):
        text = "\n".join(str(item) for item in traces)
    else:
        text = str(traces)
    return text[:max_chars]


def skill_text(skill: dict[str, Any]) -> str:
    parts = [
        f"Name: {skill.get('name', '')}",
        f"Description: {skill.get('description', '')}",
        f"When to use: {skill.get('when_to_use', '')}",
        f"Procedure: {skill.get('procedure', '')}",
    ]
    examples = skill.get("examples") or []
    if examples:
        parts.append("Examples: " + json.dumps(examples[:2], ensure_ascii=False))
    return "\n".join(parts)


def tools(session: dict[str, Any]) -> set[str]:
    return {str(tool) for tool in (session.get("tools_called") or []) if tool}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def lexical_embeddings(texts: list[str]) -> np.ndarray:
    """Small dependency-free TF-IDF backend for dry runs, not final results."""
    tokenized = [
        re.findall(r"[a-z0-9_]+", text.lower())
        for text in texts
    ]
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    vocab = {
        token: idx
        for idx, token in enumerate(sorted(document_frequency))
    }
    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    n_docs = max(len(texts), 1)
    for row_idx, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        for token, count in counts.items():
            idx = vocab[token]
            idf = math.log((1 + n_docs) / (1 + document_frequency[token])) + 1
            matrix[row_idx, idx] = (1 + math.log(count)) * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def embed_texts(
    texts: list[str],
    backend: str,
    model_name: str,
) -> np.ndarray:
    if backend == "lexical":
        return lexical_embeddings(texts)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required for the formal hard-negative "
            "audit. Install it or use --embedding-backend lexical only for "
            "dry-run plumbing checks."
        ) from exc
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_domain(artifact_dir: Path, levels: set[int]) -> DomainData:
    all_skills: dict[str, dict[str, Any]] = {}
    for filename in (
        "level0_all.json",
        "round1_skills.json",
        "round2_skills.json",
        "round3_skills.json",
    ):
        for skill in load_json(artifact_dir / filename):
            skill_id = str(skill.get("id") or "")
            if skill_id:
                all_skills[skill_id] = skill

    targets: list[dict[str, Any]] = []
    if 0 in levels:
        targets.extend(
            skill for skill in load_json(artifact_dir / "level0_all.json")
            if int(skill.get("level") or 0) == 0
        )
    for level in sorted(levels - {0}):
        targets.extend(
            skill for skill in load_json(artifact_dir / f"round{level}_skills.json")
            if int(skill.get("level") or 0) == level
        )

    sessions = {
        str(session.get("task_id")): session
        for session in load_json(artifact_dir / "sessions.json")
        if session.get("task_id") is not None
    }
    task_ids = sorted(sessions)
    return DomainData(
        name=artifact_dir.parts[-2],
        artifact_dir=artifact_dir,
        skills=all_skills,
        targets=targets,
        sessions=sessions,
        task_ids=task_ids,
        task_texts=[task_type_text(sessions[task_id]) for task_id in task_ids],
    )


def source_tasks(
    skill: dict[str, Any],
    skills: dict[str, dict[str, Any]],
    cache: dict[str, set[str]],
    active: set[str] | None = None,
) -> set[str]:
    skill_id = str(skill.get("id") or "")
    if skill_id in cache:
        return cache[skill_id]
    active = set(active or ())
    if skill_id in active:
        return set()
    active.add(skill_id)

    if int(skill.get("level") or 0) == 0:
        source = skill.get("source_task")
        result = {str(source)} if source is not None else set()
    else:
        result: set[str] = set()
        for parent_id in skill.get("parent_ids") or []:
            parent = skills.get(str(parent_id))
            if parent:
                result.update(source_tasks(parent, skills, cache, active))
    cache[skill_id] = result
    return result


def select_targets(
    domains: list[DomainData],
    levels: set[int],
    skills_per_level: int,
    limit: int,
    seed: int,
) -> list[tuple[DomainData, dict[str, Any], set[str]]]:
    eligible: list[tuple[DomainData, dict[str, Any], set[str]]] = []
    for domain in domains:
        cache: dict[str, set[str]] = {}
        for skill in domain.targets:
            sources = source_tasks(skill, domain.skills, cache)
            sources = {source for source in sources if source in domain.sessions}
            if sources:
                eligible.append((domain, skill, sources))

    selected: list[tuple[DomainData, dict[str, Any], set[str]]] = []
    rng = random.Random(seed)
    for level in sorted(levels):
        rows = [
            row for row in eligible
            if int(row[1].get("level") or 0) == level
        ]
        rng.shuffle(rows)
        if skills_per_level > 0:
            rows = rows[:skills_per_level]
        selected.extend(rows)
    rng.shuffle(selected)
    if limit > 0:
        selected = selected[:limit]
    return selected


def weighted_rank_sample(
    ranked_ids: list[str],
    count: int,
    rng: random.Random,
    rank_power: float,
) -> list[str]:
    """Sample without replacement while strongly favoring nearest candidates."""
    races = []
    for rank, task_id in enumerate(ranked_ids):
        weight = 1.0 / ((rank + 1) ** rank_power)
        race = -math.log(max(rng.random(), 1e-12)) / weight
        races.append((race, rank, task_id))
    chosen = sorted(races)[:count]
    chosen.sort(key=lambda row: row[1])
    return [task_id for _, _, task_id in chosen]


def build_plans(
    selected: list[tuple[DomainData, dict[str, Any], set[str]]],
    m_values: list[int],
    repeats: int,
    hard_pool_size: int,
    rank_power: float,
    seed: int,
    require_shared_tool: bool,
) -> tuple[list[LineupPlan], list[dict[str, Any]]]:
    plans: list[LineupPlan] = []
    skipped: list[dict[str, Any]] = []
    max_negatives = max(m_values) - 1

    for domain, skill, sources in selected:
        task_index = {task_id: idx for idx, task_id in enumerate(domain.task_ids)}
        skill_id = str(skill.get("id") or "")
        for repeat in range(repeats):
            rng = random.Random(stable_seed(seed, domain.name, skill_id, repeat))
            true_task_id = rng.choice(sorted(sources))
            true_session = domain.sessions[true_task_id]
            true_tools = tools(true_session)
            true_idx = task_index[true_task_id]
            similarities = np.dot(domain.embeddings, domain.embeddings[true_idx])

            ranked: list[tuple[float, float, str]] = []
            for candidate_id in domain.task_ids:
                if candidate_id in sources:
                    continue
                candidate_tools = tools(domain.sessions[candidate_id])
                tool_overlap = jaccard(true_tools, candidate_tools)
                if require_shared_tool and not (true_tools & candidate_tools):
                    continue
                similarity = float(similarities[task_index[candidate_id]])
                ranked.append((similarity, tool_overlap, candidate_id))

            ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
            pool = ranked[:max(hard_pool_size, max_negatives)]
            if len(pool) < max_negatives:
                skipped.append({
                    "domain": domain.name,
                    "level": int(skill.get("level") or 0),
                    "skill_id": skill_id,
                    "repeat": repeat,
                    "reason": "insufficient shared-tool non-ancestral candidates",
                    "available": len(pool),
                    "required": max_negatives,
                })
                continue

            ranked_ids = [row[2] for row in pool]
            selected_negatives = weighted_rank_sample(
                ranked_ids,
                max_negatives,
                rng,
                rank_power,
            )
            stats = {
                task_id: (similarity, tool_overlap)
                for similarity, tool_overlap, task_id in pool
            }

            # The hardest sampled negatives enter the smaller lineup first.
            selected_negatives.sort(
                key=lambda task_id: stats[task_id],
                reverse=True,
            )
            for m in m_values:
                negatives = selected_negatives[:m - 1]
                lineup = negatives + [true_task_id]
                order_rng = random.Random(
                    stable_seed(seed, domain.name, skill_id, repeat, m, "order")
                )
                order_rng.shuffle(lineup)
                true_position = lineup.index(true_task_id)
                key = f"{domain.name}|L{skill.get('level', 0)}|{skill_id}|r{repeat}|m{m}"
                plans.append(LineupPlan(
                    key=key,
                    domain=domain.name,
                    level=int(skill.get("level") or 0),
                    skill_id=skill_id,
                    repeat=repeat,
                    m=m,
                    true_task_id=true_task_id,
                    lineup=lineup,
                    true_position=true_position,
                    negative_similarities=[
                        round(stats[task_id][0], 6) for task_id in negatives
                    ],
                    negative_tool_jaccards=[
                        round(stats[task_id][1], 6) for task_id in negatives
                    ],
                ))
    return plans, skipped


def resolve_provider(args: argparse.Namespace) -> tuple[str, str, str, str]:
    spec = PROVIDERS[args.provider]
    api_key = args.api_key
    for env_name in spec["api_key_envs"]:
        if api_key:
            break
        api_key = os.environ.get(env_name, "")
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set one of {spec['api_key_envs']} or pass --api-key."
        )
    base_url = args.base_url or spec["base_url"]
    model = args.model or spec["model"]
    if spec["backend"] == "openai" and not base_url:
        raise SystemExit("Missing --base-url for the generic provider.")
    if not model:
        raise SystemExit("Missing --model.")
    return api_key, base_url, model, spec["backend"]


def create_client(
    api_key: str,
    base_url: str,
    backend: str,
    timeout: float,
) -> Any:
    if backend == "zai":
        from zai import ZhipuAiClient
        return ZhipuAiClient(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def response_content(response: Any) -> str:
    message = response.choices[0].message
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


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


def run_judge(
    client: Any,
    backend: str,
    model: str,
    plan: LineupPlan,
    domain: DomainData,
    skill: dict[str, Any],
    max_trace_chars: int,
    temperature: float,
    max_tokens: int,
    timeout: float,
    thinking: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    traces_text = "\n\n".join(
        f"--- Candidate {idx + 1} ---\n"
        f"{trace_text(domain.sessions[task_id], max_trace_chars)}"
        for idx, task_id in enumerate(plan.lineup)
    )
    prompt = USER_PROMPT.format(
        level=plan.level,
        skill_text=skill_text(skill),
        traces_text=traces_text,
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
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
    parsed = extract_json(response_content(response))
    choice = int(parsed.get("choice", 0)) - 1
    if not 0 <= choice < plan.m:
        raise ValueError(f"Judge choice {choice + 1} is outside 1..{plan.m}")
    confidence = float(parsed.get("confidence", 0))
    return {
        "choice": choice,
        "correct": choice == plan.true_position,
        "confidence": min(max(confidence, 0.0), 1.0),
        "reason": str(parsed.get("reason") or ""),
        "raw": parsed,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def metric_summary(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_skill: dict[str, list[float]] = {}
    for row in rows:
        by_skill.setdefault(str(row["skill_key"]), []).append(
            1.0 if bool(row["correct"]) else 0.0
        )
    skill_means = {
        skill_key: sum(values) / len(values)
        for skill_key, values in by_skill.items()
    }
    if not skill_means:
        return {"n_skills": 0, "n_calls": 0}

    m = int(rows[0]["m"])
    accuracy = statistics.mean(skill_means.values())
    random_baseline = 1 / m
    normalized = (accuracy - random_baseline) / (1 - random_baseline)

    keys = sorted(skill_means)
    rng = random.Random(seed)
    boot_accuracy: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [skill_means[rng.choice(keys)] for _ in keys]
        boot_accuracy.append(statistics.mean(sampled))
    boot_normalized = [
        (value - random_baseline) / (1 - random_baseline)
        for value in boot_accuracy
    ]
    return {
        "n_skills": len(skill_means),
        "n_calls": len(rows),
        "mean_repeats_per_skill": round(len(rows) / len(skill_means), 3),
        "sa": round(accuracy, 4),
        "sa_bootstrap_95ci": [
            round(percentile(boot_accuracy, 0.025), 4),
            round(percentile(boot_accuracy, 0.975), 4),
        ],
        "random_baseline": round(random_baseline, 4),
        "normalized_advantage": round(normalized, 4),
        "normalized_advantage_bootstrap_95ci": [
            round(percentile(boot_normalized, 0.025), 4),
            round(percentile(boot_normalized, 0.975), 4),
        ],
        "mean_confidence": round(
            statistics.mean(float(row["confidence"]) for row in rows),
            4,
        ),
        "mean_negative_embedding_similarity": round(
            statistics.mean(
                similarity
                for row in rows
                for similarity in row["negative_similarities"]
            ),
            4,
        ),
        "mean_negative_tool_jaccard": round(
            statistics.mean(
                overlap
                for row in rows
                for overlap in row["negative_tool_jaccards"]
            ),
            4,
        ),
    }


def paired_delta_summary(
    rows: list[dict[str, Any]],
    m_low: int,
    m_high: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_skill_m: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        by_skill_m.setdefault(
            (str(row["skill_key"]), int(row["m"])),
            [],
        ).append(1.0 if bool(row["correct"]) else 0.0)
    skill_keys = sorted({
        skill_key
        for skill_key, m in by_skill_m
        if (skill_key, m_low) in by_skill_m and (skill_key, m_high) in by_skill_m
    })
    deltas = {
        skill_key: (
            statistics.mean(by_skill_m[(skill_key, m_high)])
            - statistics.mean(by_skill_m[(skill_key, m_low)])
        )
        for skill_key in skill_keys
    }
    if not deltas:
        return {"n_paired_skills": 0}
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [deltas[rng.choice(skill_keys)] for _ in skill_keys]
        boot.append(statistics.mean(sampled))
    return {
        "n_paired_skills": len(deltas),
        f"sa_delta_m{m_high}_minus_m{m_low}": round(
            statistics.mean(deltas.values()),
            4,
        ),
        "skill_bootstrap_95ci": [
            round(percentile(boot, 0.025), 4),
            round(percentile(boot, 0.975), 4),
        ],
    }


def summarize(
    rows: list[dict[str, Any]],
    m_values: list[int],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"by_m": {}}
    for m in m_values:
        m_rows = [row for row in rows if int(row["m"]) == m]
        summary["by_m"][str(m)] = metric_summary(
            m_rows,
            bootstrap_samples,
            stable_seed(seed, "overall", m),
        )
        summary["by_m"][str(m)]["by_level"] = {
            str(level): metric_summary(
                [row for row in m_rows if int(row["level"]) == level],
                bootstrap_samples,
                stable_seed(seed, "level", level, m),
            )
            for level in sorted({int(row["level"]) for row in m_rows})
        }
        summary["by_m"][str(m)]["by_domain"] = {
            domain: metric_summary(
                [row for row in m_rows if row["domain"] == domain],
                bootstrap_samples,
                stable_seed(seed, "domain", domain, m),
            )
            for domain in sorted({str(row["domain"]) for row in m_rows})
        }
    if len(m_values) == 2:
        summary["paired_sensitivity"] = paired_delta_summary(
            rows,
            min(m_values),
            max(m_values),
            bootstrap_samples,
            stable_seed(seed, "paired"),
        )
    return summary


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("key"):
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="zai")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--levels", default="3")
    parser.add_argument("--m-values", default="5,10")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skills-per-level", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-pool-size", type=int, default=30)
    parser.add_argument("--rank-power", type=float, default=0.75)
    parser.add_argument(
        "--allow-no-shared-tool",
        action="store_true",
        help="Allow hard negatives without a shared tool when strict matching is sparse.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformers", "lexical"],
        default="sentence-transformers",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--max-trace-chars", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--reasoning-effort", default="max")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT / "source_attribution_sensitivity"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    levels = {int(value) for value in args.levels.split(",") if value.strip()}
    m_values = sorted({
        int(value) for value in args.m_values.split(",") if value.strip()
    })
    if not levels or not m_values or min(m_values) < 2:
        raise SystemExit("Provide non-empty levels and m-values >= 2.")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1.")

    domains = [
        load_domain(path, levels)
        for path in DEFAULT_ARTIFACT_DIRS
        if path.exists()
    ]
    print(f"[load] {len(domains)} domains")
    for domain in domains:
        print(
            f"  {domain.name}: {len(domain.sessions)} sessions, "
            f"{len(domain.targets)} target skills"
        )

    # Encode each domain separately so nearest neighbors are domain-matched.
    for domain in domains:
        print(f"[embed] {domain.name}: {len(domain.task_texts)} task descriptions")
        domain.embeddings = embed_texts(
            domain.task_texts,
            args.embedding_backend,
            args.embedding_model,
        )

    selected = select_targets(
        domains,
        levels,
        args.skills_per_level,
        args.limit,
        args.seed,
    )
    print(f"[select] {len(selected)} eligible skills")
    for level in sorted(levels):
        print(
            f"  L{level}: "
            f"{sum(1 for _, skill, _ in selected if int(skill.get('level') or 0) == level)}"
        )

    plans, skipped = build_plans(
        selected,
        m_values,
        args.repeats,
        args.hard_pool_size,
        args.rank_power,
        args.seed,
        require_shared_tool=not args.allow_no_shared_tool,
    )
    print(f"[plan] {len(plans)} model calls; {len(skipped)} skipped repeats")
    for m in m_values:
        print(f"  m={m}: {sum(1 for plan in plans if plan.m == m)}")
    if skipped:
        counts = Counter(row["domain"] for row in skipped)
        print(f"[skip] by domain: {dict(counts)}")

    if plans:
        example = plans[0]
        print(
            "[example] "
            f"{example.key} true_pos={example.true_position + 1} "
            f"mean_neg_sim={statistics.mean(example.negative_similarities):.3f} "
            f"mean_tool_jaccard={statistics.mean(example.negative_tool_jaccards):.3f}"
        )
    if args.dry_run:
        print("[dry-run] no judge calls made")
        return

    api_key, base_url, model, backend = resolve_provider(args)
    client = create_client(api_key, base_url, backend, args.timeout)
    domain_by_name = {domain.name: domain for domain in domains}
    skill_by_key = {
        (domain.name, str(skill.get("id") or "")): skill
        for domain, skill, _ in selected
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    level_tag = "L" + "-".join(str(level) for level in sorted(levels))
    m_tag = "m" + "-".join(str(m) for m in m_values)
    run_tag = args.run_name or f"{level_tag}_{m_tag}_r{args.repeats}"
    run_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_tag)
    prefix = f"{args.provider}_{model_tag}_{run_tag}"
    jsonl_path = out_dir / f"{prefix}_rows.jsonl"
    csv_path = out_dir / f"{prefix}_rows.csv"
    summary_path = out_dir / f"{prefix}_summary.json"
    skipped_path = out_dir / f"{prefix}_skipped.json"

    prior_rows = parse_jsonl(jsonl_path)
    done = {str(row["key"]) for row in prior_rows}
    print(f"[resume] {len(done)} completed calls")

    with jsonl_path.open("a", encoding="utf-8") as jf:
        for idx, plan in enumerate(plans, start=1):
            if plan.key in done:
                continue
            domain = domain_by_name[plan.domain]
            skill = skill_by_key[(plan.domain, plan.skill_id)]
            try:
                result = run_judge(
                    client,
                    backend,
                    model,
                    plan,
                    domain,
                    skill,
                    args.max_trace_chars,
                    args.temperature,
                    args.max_tokens,
                    args.timeout,
                    args.thinking,
                    args.reasoning_effort,
                )
                row = {
                    "key": plan.key,
                    "skill_key": f"{plan.domain}|{plan.skill_id}",
                    "domain": plan.domain,
                    "level": plan.level,
                    "skill_id": plan.skill_id,
                    "repeat": plan.repeat,
                    "m": plan.m,
                    "true_task_id": plan.true_task_id,
                    "lineup_task_ids": plan.lineup,
                    "true_position": plan.true_position,
                    "choice": result["choice"],
                    "correct": result["correct"],
                    "confidence": result["confidence"],
                    "reason": result["reason"],
                    "negative_similarities": plan.negative_similarities,
                    "negative_tool_jaccards": plan.negative_tool_jaccards,
                    "raw": result["raw"],
                }
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                jf.flush()
                prior_rows.append(row)
                done.add(plan.key)
                print(
                    f"[{idx}/{len(plans)}] {plan.domain} L{plan.level} "
                    f"skill={plan.skill_id} r={plan.repeat} m={plan.m} "
                    f"correct={result['correct']} conf={result['confidence']:.2f}"
                )
            except Exception as exc:
                print(f"[error] {plan.key}: {exc}")
            if args.sleep:
                time.sleep(args.sleep)

    fieldnames = [
        "domain", "level", "skill_id", "repeat", "m", "correct",
        "confidence", "true_position", "choice", "true_task_id", "reason",
        "mean_negative_similarity", "mean_negative_tool_jaccard",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in prior_rows:
            writer.writerow({
                "domain": row["domain"],
                "level": row["level"],
                "skill_id": row["skill_id"],
                "repeat": row["repeat"],
                "m": row["m"],
                "correct": row["correct"],
                "confidence": row["confidence"],
                "true_position": int(row["true_position"]) + 1,
                "choice": int(row["choice"]) + 1,
                "true_task_id": row["true_task_id"],
                "reason": row["reason"],
                "mean_negative_similarity": round(
                    statistics.mean(row["negative_similarities"]),
                    6,
                ),
                "mean_negative_tool_jaccard": round(
                    statistics.mean(row["negative_tool_jaccards"]),
                    6,
                ),
            })

    summary = summarize(
        prior_rows,
        m_values,
        args.bootstrap_samples,
        args.seed,
    )
    summary.update({
        "provider": args.provider,
        "model": model,
        "backend": backend,
        "levels": sorted(levels),
        "m_values": m_values,
        "repeats": args.repeats,
        "skills_per_level": args.skills_per_level,
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "hard_negative_policy": (
            "same domain; shared tool required; embedding-nearest task type"
            if not args.allow_no_shared_tool
            else "same domain; shared tool preferred; embedding-nearest task type"
        ),
        "hard_pool_size": args.hard_pool_size,
        "rank_power": args.rank_power,
        "max_trace_chars": args.max_trace_chars,
        "bootstrap_unit": "skill",
        "bootstrap_samples": args.bootstrap_samples,
        "planned_calls": len(plans),
        "completed_calls": len(prior_rows),
        "skipped_repeats": len(skipped),
        "rows_jsonl": str(jsonl_path),
        "rows_csv": str(csv_path),
    })
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    skipped_path.write_text(
        json.dumps(skipped, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
