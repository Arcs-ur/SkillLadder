"""Build success-only L0 skills file for the AWM baseline.

For each domain, filter `evolved_all.json` to L0 skills extracted from
sessions whose baseline outcome was "success". This mirrors the original
Agent Workflow Memory (Wang et al., 2024) protocol of building memory only
from successful trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(domain_dir: Path, out_path: Path) -> None:
    sessions_path = domain_dir / "skills_from_train" / "sessions.json"
    skills_path = domain_dir / "skills_from_train" / "evolved_all.json"

    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    success_ids = {str(s["task_id"]) for s in sessions if s.get("outcome") == "success"}

    skills = json.loads(skills_path.read_text(encoding="utf-8"))
    l0_success = [
        s
        for s in skills
        if int(s.get("level", 0)) == 0 and str(s.get("source_task", "")) in success_ids
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(l0_success, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[{domain_dir.name}] success sessions={len(success_ids)} | L0 success skills={len(l0_success)} -> {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domain_dir", help="e.g. output/tau2_airline_train50")
    parser.add_argument("out_path", help="destination json")
    args = parser.parse_args()
    build(Path(args.domain_dir), Path(args.out_path))


if __name__ == "__main__":
    main()
