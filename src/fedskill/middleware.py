"""FedSkill Middleware — orchestrates the full federated skill evolution loop
across one or more pluggable execution backends.

Lifecycle:
  1. Collect session data from all backends
  2. Extract local skills from traces
  3. Cluster & evolve via FedSkillServer
  4. (Optional) Evaluate evolved skills by running tasks on backends
  5. Distribute evolved skills back to all backends
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from fedskill.backends.base import ExecutionBackend, SessionData, TaskSpec
from fedskill.backends.registry import get_backend
from fedskill.client.task_client import TaskClient
from fedskill.models.skill import Skill
from fedskill.server.aggregator import FedSkillServer


class FedSkillMiddleware:
    """Central middleware that connects backends to the evolution engine.

    Args:
        backend_configs: List of backend configurations, each a dict with:
            - type: str  — backend type name ("local_sim", "docker_bench", "http")
            - name: str  — display name (optional, defaults to type)
            - ... backend-specific config keys
        output_dir: Directory to save results.
    """

    def __init__(
        self,
        backend_configs: list[dict[str, Any]],
        output_dir: str = "output",
    ):
        self.backends: list[ExecutionBackend] = []
        for cfg in backend_configs:
            backend_type = cfg.pop("type")
            name = cfg.pop("name", backend_type)
            backend = get_backend(backend_type, config=cfg)
            backend.name = name
            self.backends.append(backend)

        self.server = FedSkillServer()
        self.output_dir = output_dir
        self.evolved_skills: list[Skill] = []
        self._all_sessions: dict[str, list[SessionData]] = {}
        self._skill_library: dict[str, Skill] = {}  # id -> Skill, for tracking effectiveness

    # ------------------------------------------------------------------
    # Step 1: Collect sessions from backends
    # ------------------------------------------------------------------

    def collect_sessions(
        self,
        limit_per_backend: int = 100,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, list[SessionData]]:
        """Collect recent session data from all backends.

        Returns:
            Mapping of backend_name -> list of SessionData.
        """
        self._all_sessions.clear()
        for backend in self.backends:
            if progress:
                progress(f"Collecting sessions from \\[{backend.name}] ...")
            sessions = backend.collect_sessions(limit=limit_per_backend)
            self._all_sessions[backend.name] = sessions
            if progress:
                progress(f"  → {len(sessions)} sessions from \\[{backend.name}]")
        return self._all_sessions

    # ------------------------------------------------------------------
    # Step 2: Run tasks on backends (active mode)
    # ------------------------------------------------------------------

    def run_tasks(
        self,
        tasks: list[TaskSpec],
        skills: list[dict] | None = None,
        backend_name: str | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> list[SessionData]:
        """Run tasks on a specific backend (or the first one) and collect sessions.

        Args:
            tasks: List of task specifications.
            skills: Optional evolved skills to inject.
            backend_name: Which backend to use (None = first).
            progress: Progress callback.

        Returns:
            List of SessionData from all task runs.
        """
        backend = self._get_backend(backend_name)
        sessions = []
        for i, task in enumerate(tasks):
            if progress:
                progress(
                    f"  \\[{backend.name}] Running task {i + 1}/{len(tasks)}: "
                    f"{task.description[:60]}..."
                )
            session = backend.run_task(task, skills=skills)
            sessions.append(session)
        return sessions

    # ------------------------------------------------------------------
    # Step 3: Extract skills from session traces
    # ------------------------------------------------------------------

    def extract_skills_from_sessions(
        self,
        sessions: dict[str, list[SessionData]] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, list[Skill]]:
        """Extract local skills from collected session traces.

        Uses dual-mode extraction (inspired by Trace2Skill):
        - Success analyst for sessions with score > 0.5 (effective strategies)
        - Error analyst for sessions with score <= 0.5 (defensive patterns)

        Returns:
            Mapping of backend_name -> list of extracted Skills.
        """
        from fedskill.extractor.skill_extractor import extract_skills
        from fedskill.utils.privacy import scrub_skill

        sessions = sessions or self._all_sessions
        client_skills: dict[str, list[Skill]] = {}

        for backend_name, session_list in sessions.items():
            if not session_list:
                continue
            if progress:
                progress(f"Extracting skills from \\[{backend_name}] ({len(session_list)} sessions) ...")

            all_skills: list[Skill] = []
            for s in session_list:
                if not s.traces:
                    continue
                # Determine analyst mode based on outcome
                outcome = "unknown"
                if s.score > 0.5 or s.outcome == "success":
                    outcome = "success"
                elif s.outcome == "failure" or s.score <= 0.5:
                    outcome = "failure"

                task_desc = s.task_description or f"Tasks from {backend_name}"
                task_name = s.task_id or backend_name

                skills = extract_skills(
                    task_name=task_name,
                    task_description=task_desc,
                    traces=s.traces,
                    outcome=outcome,
                )
                for sk in skills:
                    sk.source_task = task_name
                all_skills.extend(
                    scrub_skill(
                        skill,
                        blocklist=getattr(self, "_privacy_blocklist", None),
                    )
                    for skill in skills
                )

            client_skills[backend_name] = all_skills

            if progress:
                progress(f"  → Extracted {len(all_skills)} skills from \\[{backend_name}]")

        return client_skills

    # ------------------------------------------------------------------
    # Step 4: Evolve (federated aggregation)
    # ------------------------------------------------------------------

    def evolve(
        self,
        client_skills: dict[str, list[Skill]],
        task_descriptions: list[str] | None = None,
        num_rounds: int = 1,
        progress: Callable[[str], None] | None = None,
    ) -> list[Skill]:
        """Run multi-round federated skill evolution.

        Round 1: cluster + evolve local skills → level-1 skills
        Round 2: cluster + evolve level-1 skills → level-2 skills
        ...and so on.

        Each round takes the output of the previous round as input,
        producing progressively higher-level generalizations.

        Args:
            client_skills: backend_name -> local skills (level-0).
            task_descriptions: For evaluation scoring.
            num_rounds: Number of evolution rounds (default: 1).
            progress: Progress callback.

        Returns:
            All skills across all levels (level-0 + evolved).
        """
        if progress:
            total = sum(len(v) for v in client_skills.values())
            progress(
                f"Starting federated evolution with {total} skills "
                f"from {len(client_skills)} backends, {num_rounds} round(s) ..."
            )

        all_evolved: list[Skill] = []

        # Round 1: evolve from local skills
        current_input = client_skills
        for round_idx in range(num_rounds):
            if progress:
                progress(f"\n  --- Evolution Round {round_idx + 1}/{num_rounds} ---")

            round_evolved = self.server.run_round(
                client_skills=current_input,
                task_descriptions=task_descriptions,
            )

            # Filter: only keep skills that were actually evolved (level > input)
            newly_evolved = [s for s in round_evolved if s.level > round_idx]
            singletons = [s for s in round_evolved if s.level <= round_idx]

            if progress:
                progress(
                    f"  Round {round_idx + 1}: {len(newly_evolved)} evolved "
                    f"(level {round_idx + 1}), {len(singletons)} unchanged"
                )

            all_evolved.extend(newly_evolved)

            # Stop if nothing was evolved this round
            if not newly_evolved:
                if progress:
                    progress(f"  No new evolutions in round {round_idx + 1}, stopping.")
                break

            # Prepare input for next round: evolved skills become the new input
            if round_idx + 1 < num_rounds:
                current_input = {f"evolved-round-{round_idx + 1}": newly_evolved}

        self.evolved_skills = all_evolved

        # Register evolved skills in the library for effectiveness tracking
        for skill in all_evolved:
            self._skill_library[skill.id] = skill

        if progress:
            progress(f"  → Total evolved skills: {len(all_evolved)}")

        return all_evolved

    # ------------------------------------------------------------------
    # Step 5: Update effectiveness from session feedback
    # ------------------------------------------------------------------

    def update_effectiveness(
        self,
        sessions: dict[str, list[SessionData]] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Update skill effectiveness scores based on session outcomes.

        For each session that used skills, record whether the outcome was
        positive or negative, and update the skill's effectiveness metric.
        """
        sessions = sessions or self._all_sessions
        updated = 0

        for backend_name, session_list in sessions.items():
            for session in session_list:
                used_skills = session.skills_used or []
                if not used_skills:
                    continue

                for skill_name in used_skills:
                    # Find the skill in our library
                    matching = [
                        s for s in self._skill_library.values()
                        if s.name == skill_name or s.id == skill_name
                    ]
                    for skill in matching:
                        skill.record_quality_event(
                            session.score,
                            round_id="middleware_feedback",
                            source=backend_name,
                            task_id=session.task_id,
                        )
                        updated += 1

        if progress and updated > 0:
            progress(f"  Updated effectiveness for {updated} skill-session pairs")

    # ------------------------------------------------------------------
    # Step 5: Distribute evolved skills to backends
    # ------------------------------------------------------------------

    def distribute_skills(
        self,
        skills: list[Skill] | None = None,
        backend_name: str | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, bool]:
        """Push evolved skills to backends after privacy sanitization.

        Privacy pipeline (§3.7):
          1. Filter: only level >= 1 skills cross the federation boundary.
          2. Scrub: deterministic entity scrubbing (regex/NER + blocklist).
          3. Sanitize: strip lineage metadata (parent_ids, source_task).

        Args:
            skills: Skills to distribute (default: self.evolved_skills).
            backend_name: Specific backend (None = all backends).
            progress: Progress callback.

        Returns:
            Mapping of backend_name -> success boolean.
        """
        from fedskill.utils.privacy import prepare_for_sharing

        skills = skills or self.evolved_skills

        # Apply privacy pipeline: level gate + scrubbing + lineage sanitization
        shared_skills: list[Skill] = []
        for s in skills:
            sanitized = prepare_for_sharing(
                s,
                blocklist=getattr(self, "_privacy_blocklist", None),
                min_level=1,
            )
            if sanitized is not None:
                shared_skills.append(sanitized)

        if progress:
            progress(
                f"Privacy: {len(skills)} skills → {len(shared_skills)} shared "
                f"(filtered {len(skills) - len(shared_skills)} level-0 skills)"
            )

        skill_dicts = [s.to_dict() for s in shared_skills]

        targets = [self._get_backend(backend_name)] if backend_name else self.backends
        results = {}
        for backend in targets:
            if progress:
                progress(f"Distributing {len(skills)} skills to \\[{backend.name}] ...")
            try:
                ok = backend.inject_skills(skill_dicts)
                results[backend.name] = ok
                if progress:
                    status = "OK" if ok else "FAILED"
                    progress(f"  → \\[{backend.name}] {status}")
            except Exception as e:
                results[backend.name] = False
                if progress:
                    progress(f"  → \\[{backend.name}] ERROR: {e}")

        return results

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_loop(
        self,
        tasks: list[TaskSpec] | None = None,
        task_descriptions: list[str] | None = None,
        num_rounds: int = 1,
        backend_name: str | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> list[Skill]:
        """Run the complete middleware loop:
        collect → extract → evolve → distribute.

        Args:
            tasks: Optional tasks to actively run on backends first.
            task_descriptions: For evolution scoring.
            num_rounds: Number of evolution rounds.
            backend_name: Only use this backend (None = all backends).
            progress: Progress callback.

        Returns:
            Evolved skills.
        """
        run_backends = (
            [self._get_backend(backend_name)] if backend_name else self.backends
        )

        # If tasks provided, run them first
        if tasks:
            all_sessions: dict[str, list[SessionData]] = {}
            for backend in run_backends:
                if progress:
                    progress(f"\n=== Running {len(tasks)} tasks on \\[{backend.name}] ===")
                sessions = []
                for i, task in enumerate(tasks):
                    if progress:
                        progress(f"  Task {i + 1}/{len(tasks)}: {task.description[:60]}...")
                    try:
                        s = backend.run_task(task)
                        sessions.append(s)
                    except Exception as e:
                        if progress:
                            progress(f"  [!] Task {i + 1} failed: {e}")
                        sessions.append(SessionData(
                            task_id=task.task_id,
                            task_description=task.description,
                            outcome="failure",
                            metadata={"error": str(e)},
                        ))
                all_sessions[backend.name] = sessions
            self._all_sessions = all_sessions
        else:
            # Passive mode: collect existing sessions
            self.collect_sessions(progress=progress)

        # Extract
        if progress:
            progress("\n=== Extracting skills from session traces ===")
        client_skills = self.extract_skills_from_sessions(progress=progress)

        if not any(client_skills.values()):
            if progress:
                progress("No skills extracted — nothing to evolve.")
            return []

        # Evolve (multi-round)
        if progress:
            progress("\n=== Federated Skill Evolution ===")
        evolved = self.evolve(
            client_skills,
            task_descriptions=task_descriptions,
            num_rounds=num_rounds,
            progress=progress,
        )

        # Distribute
        if progress:
            progress("\n=== Distributing evolved skills ===")
        self.distribute_skills(progress=progress)

        # Update effectiveness from session feedback
        self.update_effectiveness(progress=progress)

        # Save
        self._save_results(client_skills, evolved, self._all_sessions)

        return evolved

    # ------------------------------------------------------------------
    # Daemon mode: auto-trigger evolution
    # ------------------------------------------------------------------

    def run_daemon(
        self,
        interval_seconds: int = 300,
        min_new_sessions: int = 3,
        num_rounds: int = 1,
        task_descriptions: list[str] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Run as a daemon that periodically collects sessions and evolves skills.

        The daemon loop:
        1. Collect new sessions from all backends
        2. If enough new sessions accumulated, trigger evolution
        3. Distribute evolved skills back to backends
        4. Wait and repeat

        Args:
            interval_seconds: Seconds between collection cycles (default: 300).
            min_new_sessions: Minimum new sessions before triggering evolution (default: 3).
            num_rounds: Evolution rounds per trigger (default: 1).
            task_descriptions: For evolution scoring.
            progress: Progress callback.
        """
        import time

        if progress:
            progress(
                f"[daemon] Starting — check every {interval_seconds}s, "
                f"evolve after {min_new_sessions} new sessions, "
                f"{num_rounds} round(s) per cycle"
            )

        seen_session_ids: set[str] = set()
        cycle = 0

        try:
            while True:
                cycle += 1
                if progress:
                    progress(f"\n[daemon] Cycle {cycle} — collecting sessions ...")

                # Collect
                all_sessions = self.collect_sessions(progress=progress)

                # Find new sessions
                new_sessions: dict[str, list[SessionData]] = {}
                new_count = 0
                for backend_name, session_list in all_sessions.items():
                    new = [s for s in session_list if s.session_id not in seen_session_ids]
                    if new:
                        new_sessions[backend_name] = new
                        new_count += len(new)
                        for s in new:
                            seen_session_ids.add(s.session_id)

                if progress:
                    progress(f"[daemon] Found {new_count} new session(s)")

                if new_count < min_new_sessions:
                    if progress:
                        progress(
                            f"[daemon] Not enough new sessions ({new_count} < {min_new_sessions}), "
                            f"waiting {interval_seconds}s ..."
                        )
                    time.sleep(interval_seconds)
                    continue

                # Extract from new sessions only
                if progress:
                    progress("[daemon] Extracting skills from new sessions ...")
                self._all_sessions = new_sessions
                client_skills = self.extract_skills_from_sessions(
                    sessions=new_sessions, progress=progress
                )

                if not any(client_skills.values()):
                    if progress:
                        progress("[daemon] No skills extracted, waiting ...")
                    time.sleep(interval_seconds)
                    continue

                # Evolve
                if progress:
                    progress("[daemon] Triggering evolution ...")
                evolved = self.evolve(
                    client_skills,
                    task_descriptions=task_descriptions,
                    num_rounds=num_rounds,
                    progress=progress,
                )

                # Distribute
                if progress:
                    progress("[daemon] Distributing evolved skills ...")
                self.distribute_skills(progress=progress)

                # Save
                self._save_results(client_skills, evolved, new_sessions)

                if progress:
                    progress(
                        f"[daemon] Cycle {cycle} complete — "
                        f"{len(evolved)} evolved skills distributed. "
                        f"Waiting {interval_seconds}s ..."
                    )
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            if progress:
                progress("\n[daemon] Stopped by user.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def health_check_all(self) -> dict[str, bool]:
        return {b.name: b.health_check() for b in self.backends}

    def _get_backend(self, name: str | None) -> ExecutionBackend:
        if name is None:
            return self.backends[0]
        for b in self.backends:
            if b.name == name:
                return b
        available = [b.name for b in self.backends]
        raise ValueError(f"Backend {name!r} not found. Available: {available}")

    def _save_results(
        self,
        client_skills: dict[str, list[Skill]],
        evolved: list[Skill],
        all_sessions: dict[str, list[SessionData]] | None = None,
    ) -> None:
        """Save sessions, traces, and skills in a structured directory layout.

        output/
        ├── sessions/{task_id}/
        │   ├── session.json
        │   └── trace.txt
        ├── level-0-skills/{task_id}/{skill_name}.json
        └── level-{N}-skills/{skill_name}.json
        """
        # 1. Save sessions and traces
        all_sessions = all_sessions or self._all_sessions
        for backend_name, session_list in all_sessions.items():
            for session in session_list:
                task_id = session.task_id or "unknown"
                safe_task_id = _safe_filename(task_id)
                session_dir = os.path.join(self.output_dir, "sessions", safe_task_id)
                os.makedirs(session_dir, exist_ok=True)

                # session.json — structured metadata
                session_data = session.to_dict()
                # Keep error info but remove bulky metadata
                error_info = (session.metadata or {}).get("error")
                session_data.pop("metadata", None)
                if error_info:
                    session_data["error"] = error_info
                session_path = os.path.join(session_dir, "session.json")
                with open(session_path, "w", encoding="utf-8") as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)

                # trace.txt — human-readable full trace
                traces = session.traces or []
                trace_text = f"Task: {session.task_description}\n"
                trace_text += f"Backend: {backend_name}\n"
                trace_text += f"Outcome: {session.outcome} (score={session.score})\n"
                trace_text += f"Tools called: {', '.join(session.tools_called)}\n"
                trace_text += "=" * 60 + "\n\n"
                trace_text += "\n\n---\n\n".join(traces)

                # Append turn-level detail if available
                turns = (session.metadata or {}).get("turns", [])
                if turns:
                    trace_text += "\n\n" + "=" * 60 + "\nTURN DETAILS\n" + "=" * 60 + "\n"
                    for t in turns:
                        trace_text += f"\n--- Turn {t.get('turn', '?')} ---\n"
                        trace_text += f"Action: {t.get('action', '?')}\n"
                        args = t.get("args", {})
                        if args:
                            trace_text += f"Args: {json.dumps(args, ensure_ascii=False)[:500]}\n"
                        trace_text += f"Observation: {t.get('observation', '')[:2000]}\n"

                trace_path = os.path.join(session_dir, "trace.txt")
                with open(trace_path, "w", encoding="utf-8") as f:
                    f.write(trace_text)

        # 2. Save local skills: output/level-0-skills/{task_id}/{skill_name}.json
        # Map backend_name → task_ids from sessions
        backend_task_ids: dict[str, list[str]] = {}
        for backend_name, session_list in all_sessions.items():
            backend_task_ids[backend_name] = [
                _safe_filename(s.task_id or "unknown") for s in session_list
            ]

        for client_name, skills in client_skills.items():
            # Use first task_id from this backend as the grouping dir
            task_ids = backend_task_ids.get(client_name, [])
            group_dir = task_ids[0] if len(task_ids) == 1 else _safe_filename(client_name)
            for skill in skills:
                safe_name = _safe_filename(skill.name)
                skill_dir = os.path.join(self.output_dir, "level-0-skills", group_dir)
                os.makedirs(skill_dir, exist_ok=True)
                path = os.path.join(skill_dir, f"{safe_name}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(skill.to_dict(), f, indent=2, ensure_ascii=False)

        # 3. Save evolved skills: output/level-{N}-skills/{skill_name}.json
        for skill in evolved:
            safe_name = _safe_filename(skill.name)
            skill_dir = os.path.join(self.output_dir, f"level-{skill.level}-skills")
            os.makedirs(skill_dir, exist_ok=True)
            path = os.path.join(skill_dir, f"{safe_name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(skill.to_dict(), f, indent=2, ensure_ascii=False)


def _safe_filename(name: str) -> str:
    """Sanitize a string for use as a file/directory name."""
    import re
    safe = re.sub(r'[<>:"/\\|?*]', '-', name.strip())
    safe = re.sub(r'\s+', '_', safe)
    safe = re.sub(r'-+', '-', safe).strip('-_')
    return safe[:100] or "unnamed"
