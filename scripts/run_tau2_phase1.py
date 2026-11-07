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
from fedskill.utils.run_manifest import (
    atomic_write_json, bind_state, file_digest, read_json,
    code_identity, prompt_digest, TAU2_REVISION,
)
from fedskill.utils.config import (
    add_evolution_args, parse_config_args, public_parameters, skill_env,
    clustering_options, evolution_options,
)


def save_json(path: Path, data: object) -> None:
    atomic_write_json(path, data)


def load_checkpoint(path: Path, expected_fingerprint: str, total_sessions: int) -> tuple[int, list[Skill]]:
    if not path.exists():
        return 0, []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    index = data.get("next_session")
    if (data.get("fingerprint") != expected_fingerprint
            or type(index) is not int or not 0 <= index <= total_sessions
            or data.get("total_sessions") != total_sessions):
        raise ValueError(f"Invalid or mismatched extraction checkpoint {path}; use a new output directory.")
    return index, [Skill.from_dict(d) for d in data["skills"]]


def extract_level0(sessions, output_dir: Path, *, extraction_temperature: float = 0.4) -> list[Skill]:
    checkpoint = output_dir / "level0_checkpoint.json"
    final_path = output_dir / "level0_all.json"
    identity = {
        "sessions": [s.to_dict() for s in sessions], "temperature": extraction_temperature,
        "model": os.getenv("SKILLLADDER_MODEL") or os.getenv("OPENAI_MODEL"),
    }
    manifest_path = output_dir / "level0_manifest.json"
    if final_path.exists() and not manifest_path.exists():
        raise ValueError("Unbound legacy L0 artifacts; use a new output directory for corrected reruns.")
    state = bind_state(manifest_path, identity)
    if state.get("status") == "complete":
        if not final_path.exists() or file_digest(final_path) != state.get("output_digest"):
            raise ValueError(f"Changed or missing completed artifact {final_path}")
        return [Skill.from_dict(d) for d in read_json(final_path)]
    start_idx, all_skills = load_checkpoint(checkpoint, state["fingerprint"], len(sessions))

    for idx, session in enumerate(sessions[start_idx:], start=start_idx):
        if not session.traces:
            save_json(checkpoint, {
                "fingerprint": state["fingerprint"],
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
            temperature=extraction_temperature,
        )
        for skill in skills:
            skill.source_task = task_name
            skill.source_session_id = session.session_id
        all_skills.extend(scrub_skill(skill) for skill in skills)

        save_json(checkpoint, {
            "fingerprint": state["fingerprint"],
            "next_session": idx + 1,
            "total_sessions": len(sessions),
            "skills": [s.to_dict() for s in all_skills],
        })
        print(f"[extract] saved checkpoint with {len(all_skills)} skills")

    save_json(final_path, [s.to_dict() for s in all_skills])
    state.update(status="complete", output_digest=file_digest(final_path))
    save_json(manifest_path, state)
    if checkpoint.exists():
        checkpoint.unlink()
    return all_skills


def evolve_skills(level0_skills: list[Skill], task_descriptions: list[str], output_dir: Path, rounds: int,
                   clustering_strategy: str = "hybrid",
                   disable_score_gap: bool = False,
                   disable_consistency: bool = False,
                   disable_nondegradation: bool = False,
                   disable_critical_step: bool = False,
                   seed: int = 42, clustering_kwargs: dict | None = None,
                   evolution_kwargs: dict | None = None) -> list[Skill]:
    current = {"tau2bench": level0_skills}
    all_evolved: list[Skill] = []

    for round_idx in range(rounds):
        if not any(current.values()):
            break

        print(f"[evolve] round {round_idx + 1}/{rounds} (clustering={clustering_strategy})")
        round_path = output_dir / f"round{round_idx + 1}_skills.json"
        cluster_path = output_dir / f"round{round_idx + 1}_clusters.json"
        manifest_path = output_dir / f"round{round_idx + 1}_manifest.json"
        if (round_path.exists() or cluster_path.exists()) and not manifest_path.exists():
            raise ValueError(f"Unbound legacy round artifacts in {output_dir}; use a new output directory.")
        state = bind_state(manifest_path, {
            "input": [[s.to_dict() for s in group] for group in current.values()],
            "tasks": task_descriptions, "strategy": clustering_strategy,
            "clustering": clustering_kwargs or {}, "evolution": evolution_kwargs or {},
            "seed": seed, "gates": [disable_score_gap, disable_consistency,
                                   disable_nondegradation, disable_critical_step],
        })
        if state.get("status") == "complete":
            if not round_path.exists() or file_digest(round_path) != state.get("output_digest"):
                raise ValueError(f"Changed or missing completed artifact {round_path}")
            evolved = [Skill.from_dict(d) for d in read_json(round_path)]
            all_evolved.extend(evolved)
            current = {f"tau2-round-{round_idx + 1}": evolved}
            continue
        server = FedSkillServer(clustering_strategy=clustering_strategy, **(clustering_kwargs or {}))
        if cluster_path.exists():
            if file_digest(cluster_path) != state.get("clusters_digest"):
                raise ValueError(f"Changed cluster snapshot {cluster_path}")
            clusters = [[Skill.from_dict(d) for d in group] for group in read_json(cluster_path)]
        else:
            server.collect_skills(current)
            clusters = server.cluster_skills()
            save_json(cluster_path, [[skill.to_dict() for skill in cluster] for cluster in clusters])
            state["clusters_digest"] = file_digest(cluster_path)
            save_json(manifest_path, state)

        checkpoint = output_dir / f"round{round_idx + 1}_evolve_checkpoint.json"
        evolved = server.evolve(
            clusters,
            task_descriptions=task_descriptions,
            checkpoint_path=str(checkpoint),
            disable_score_gap=disable_score_gap,
            disable_consistency=disable_consistency,
            disable_nondegradation=disable_nondegradation,
            disable_critical_step=disable_critical_step,
            seed=seed,
            **(evolution_kwargs or {}),
        )
        save_json(round_path, [s.to_dict() for s in evolved])
        state.update(status="complete", output_digest=file_digest(round_path))
        save_json(manifest_path, state)
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
    parser.add_argument("--config")
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--output-dir", default="output/tau2_phase1")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--evolve-rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    add_evolution_args(parser)
    args = parse_config_args(parser)

    output_dir = Path(args.output_dir)
    results_path = Path(args.tau2_root) / "data/simulations" / args.run_name / "results.json"
    if not results_path.exists():
        raise ValueError(f"Missing baseline results {results_path}")
    if args.dry_run:
        if not read_json(results_path).get("simulations"):
            raise ValueError(f"Empty baseline simulations in {results_path}")
        print({"input": str(results_path), "parameters": public_parameters(args),
               "clustering": clustering_options(args), "evolution": evolution_options(args)})
        return
    if not args.skill_model:
        raise ValueError("--skill-model / SKILLLADDER_MODEL is required for extraction and evolution.")
    for key, value in skill_env(args).items():
        if key.startswith("SKILLLADDER_"):
            os.environ[key] = value
    backend = get_backend("tau2bench", config={
        "tau2_root": args.tau2_root,
        "run_name": args.run_name,
        "domain": args.domain,
    })

    session_path = output_dir / "sessions.json"
    manifest_path = output_dir / "phase1_manifest.json"
    if session_path.exists() and not manifest_path.exists():
        raise ValueError("Unbound legacy sessions; use a new output directory for corrected reruns.")
    state = bind_state(manifest_path, {
        "code": code_identity(Path(__file__).resolve().parents[1]),
        "prompt_digest": prompt_digest(Path(__file__).resolve().parents[1]),
        "tau2_revision": TAU2_REVISION,
        "results_digest": file_digest(results_path),
        "parameters": public_parameters(args),
        "source_models": {role: (read_json(results_path).get("info", {}).get(f"{role}_info") or {}).get("llm")
                          for role in ("agent", "user")},
    })
    if session_path.exists():
        if file_digest(session_path) != state.get("sessions_digest"):
            raise ValueError(f"Changed session snapshot {session_path}")
        from fedskill.backends.base import SessionData
        sessions = [SessionData.from_dict(d) for d in read_json(session_path)]
    else:
        sessions = backend.collect_sessions(limit=args.limit)
        if not sessions:
            raise ValueError(f"No sessions in {results_path}")
        save_json(session_path, [s.to_dict() for s in sessions])
        state["sessions_digest"] = file_digest(session_path)
        save_json(manifest_path, state)
    print(f"[tau2] fixed snapshot: {len(sessions)} sessions from run={args.run_name}")

    level0 = extract_level0(sessions, output_dir, extraction_temperature=args.extraction_temperature)
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
            seed=args.seed,
            clustering_kwargs=clustering_options(args), evolution_kwargs=evolution_options(args),
        )
        print(f"[evolve] total evolved/retained skills: {len(evolved)}")
    else:
        save_json(output_dir / "evolved_all.json", [s.to_dict() for s in level0])


if __name__ == "__main__":
    main()
