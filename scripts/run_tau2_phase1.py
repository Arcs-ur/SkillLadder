"""Run the tau2-bench extraction and SkillLadder evolution pipeline.

This script starts from an existing tau2-bench baseline run, converts its
simulations into FedSkill sessions, extracts level-0 skills, and optionally runs
one or more evolution rounds.

Outputs are saved incrementally so the script can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fedskill.backends.registry import get_backend
from fedskill.extractor.skill_extractor import extract_skills
from fedskill.models.skill import Skill
from fedskill.server.aggregator import FedSkillServer
from fedskill.utils.privacy import scrub_skill


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> tuple[int, list[Skill]]:
    if not path.exists():
        return 0, []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("next_session", 0)), [Skill.from_dict(d) for d in data.get("skills", [])]


def extract_level0(sessions, output_dir: Path) -> list[Skill]:
    checkpoint = output_dir / "level0_checkpoint.json"
    final_path = output_dir / "level0_all.json"
    start_idx, all_skills = load_checkpoint(checkpoint)

    for idx, session in enumerate(sessions[start_idx:], start=start_idx):
        if not session.traces:
            save_json(checkpoint, {
                "next_session": idx + 1,
                "total_sessions": len(sessions),
                "skills": [s.to_dict() for s in all_skills],
            })
            continue

        outcome = "success" if session.score > 0.5 or session.outcome == "success" else "failure"
        task_name = session.task_id or f"tau2-session-{idx}"
        print(f"[extract] {idx + 1}/{len(sessions)} task={task_name} outcome={outcome}")

        skills = extract_skills(
            task_name=task_name,
            task_description=session.task_description,
            traces=session.traces,
            outcome=outcome,
        )
        for skill in skills:
            skill.source_task = task_name
        all_skills.extend(scrub_skill(skill) for skill in skills)

        save_json(checkpoint, {
            "next_session": idx + 1,
            "total_sessions": len(sessions),
            "skills": [s.to_dict() for s in all_skills],
        })
        print(f"[extract] saved checkpoint with {len(all_skills)} skills")

    save_json(final_path, [s.to_dict() for s in all_skills])
    if checkpoint.exists():
        checkpoint.unlink()
    return all_skills


def evolve_skills(level0_skills: list[Skill], task_descriptions: list[str], output_dir: Path, rounds: int,
                   clustering_strategy: str = "hybrid",
                   disable_score_gap: bool = False,
                   disable_consistency: bool = False,
                   disable_nondegradation: bool = False,
                   disable_critical_step: bool = False) -> list[Skill]:
    current = {"tau2bench": level0_skills}
    all_evolved: list[Skill] = []

    for round_idx in range(rounds):
        if not any(current.values()):
            break

        print(f"[evolve] round {round_idx + 1}/{rounds} (clustering={clustering_strategy})")
        server = FedSkillServer(clustering_strategy=clustering_strategy)
        server.collect_skills(current)
        clusters = server.cluster_skills()
        save_json(
            output_dir / f"round{round_idx + 1}_clusters.json",
            [[skill.to_dict() for skill in cluster] for cluster in clusters],
        )

        checkpoint = output_dir / f"round{round_idx + 1}_evolve_checkpoint.json"
        evolved = server.evolve(
            clusters,
            task_descriptions=task_descriptions,
            checkpoint_path=str(checkpoint),
            disable_score_gap=disable_score_gap,
            disable_consistency=disable_consistency,
            disable_nondegradation=disable_nondegradation,
            disable_critical_step=disable_critical_step,
        )
        save_json(output_dir / f"round{round_idx + 1}_skills.json", [s.to_dict() for s in evolved])
        if checkpoint.exists():
            checkpoint.unlink()

        all_evolved.extend(evolved)
        current = {f"tau2-round-{round_idx + 1}": evolved}
        print(f"[evolve] round {round_idx + 1} produced {len(evolved)} skills")

    save_json(output_dir / "evolved_all.json", [s.to_dict() for s in all_evolved])
    return all_evolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and evolve SkillLadder skills from a tau2 run")
    parser.add_argument(
        "--tau2-root",
        default=os.getenv("TAU2_ROOT"),
        required=not bool(os.getenv("TAU2_ROOT")),
        help="Path to a tau2-bench checkout (or set TAU2_ROOT).",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="output/tau2_phase1")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--evolve-rounds", type=int, default=1)
    parser.add_argument("--clustering-strategy", default="hybrid",
                        choices=["llm", "embedding", "tag", "hybrid", "reverse_hybrid", "random"])
    parser.add_argument("--disable-score-gap", action="store_true")
    parser.add_argument("--disable-consistency", action="store_true")
    parser.add_argument("--disable-nondegradation", action="store_true")
    parser.add_argument("--disable-critical-step", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    backend = get_backend("tau2bench", config={
        "tau2_root": args.tau2_root,
        "run_name": args.run_name,
    })

    sessions = backend.collect_sessions(limit=args.limit)
    print(f"[tau2] collected {len(sessions)} sessions from run={args.run_name}")
    save_json(output_dir / "sessions.json", [s.to_dict() for s in sessions])

    level0 = extract_level0(sessions, output_dir)
    print(f"[extract] total level-0 skills: {len(level0)}")

    if args.evolve_rounds > 0:
        task_descriptions = [s.task_description for s in sessions if s.task_description]
        evolved = evolve_skills(
            level0, task_descriptions, output_dir, args.evolve_rounds,
            clustering_strategy=args.clustering_strategy,
            disable_score_gap=args.disable_score_gap,
            disable_consistency=args.disable_consistency,
            disable_nondegradation=args.disable_nondegradation,
            disable_critical_step=args.disable_critical_step,
        )
        print(f"[evolve] total evolved/retained skills: {len(evolved)}")


if __name__ == "__main__":
    main()
