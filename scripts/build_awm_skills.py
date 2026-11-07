"""Build success-only L0 skills file for the AWM baseline.

For each domain, filter `level0_all.json` to L0 skills extracted from
sessions whose baseline outcome was "success". This mirrors the original
Agent Workflow Memory (Wang et al., 2025) protocol of building memory only
from successful trajectories.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from fedskill.utils.run_manifest import atomic_write_json


def success_only_l0(domain_dir: Path) -> list[dict]:
    artifacts = domain_dir / "skills_from_train"
    sessions = json.loads((artifacts / "sessions.json").read_text(encoding="utf-8"))
    skills = json.loads((artifacts / "level0_all.json").read_text(encoding="utf-8"))
    if not isinstance(sessions, list) or not isinstance(skills, list):
        raise ValueError("sessions.json and level0_all.json must contain arrays")
    success_ids = {str(s["task_id"]) for s in sessions if s.get("outcome") == "success"}
    success_sessions = {
        str(s["session_id"]) for s in sessions
        if s.get("outcome") == "success" and s.get("session_id")
    }
    counts = Counter(str(s["task_id"]) for s in sessions)
    selected = []
    seen = set()
    for skill in skills:
        if int(skill.get("level", 0)) != 0:
            continue
        task_id = str(skill.get("source_task", ""))
        session_id = skill.get("source_session_id")
        if not session_id and counts[task_id] > 1:
            raise ValueError("Multi-session AWM input requires source_session_id on L0 skills")
        successful = (
            str(session_id) in success_sessions if session_id else task_id in success_ids
        )
        skill_id = skill.get("id")
        if successful and skill_id not in seen:
            if not skill_id:
                raise ValueError("AWM L0 skills must have stable IDs")
            selected.append(skill)
            seen.add(skill_id)
    return selected


def build(domain_dir: Path, out_path: Path) -> None:
    l0_success = success_only_l0(domain_dir)
    atomic_write_json(out_path, l0_success)
    print(
        f"[{domain_dir.name}] original success-only L0 skills={len(l0_success)} -> {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domain_dir", help="e.g. output/tau2_airline_train50")
    parser.add_argument("out_path", help="destination json")
    args = parser.parse_args()
    build(Path(args.domain_dir), Path(args.out_path))


if __name__ == "__main__":
    main()
