"""Federated aggregator — groups similar skills and triggers evolution."""

from __future__ import annotations

import json
from fedskill.models.skill import Skill
from fedskill.server.evolver import evolve_skill_cluster, evaluate_skill
from fedskill.clustering import get_clustering_strategy, ClusteringStrategy


class FedSkillServer:
    """Central server that aggregates skills from clients and evolves them.

    Workflow per round:
    1. Collect local skills from all clients.
    2. Cluster similar skills across clients.
    3. Evolve each cluster into a higher-level generalised skill.
    4. (Optional) Evaluate evolved skills for generalisability.
    5. Distribute evolved skills back (or store them).

    Args:
        clustering_strategy: Name of clustering strategy ("llm", "embedding", "tag", "hybrid").
            Default: "hybrid".
        clustering_kwargs: Extra parameters for the clustering strategy.
    """

    def __init__(
        self,
        clustering_strategy: str = "hybrid",
        min_cluster_size: int = 2,
        **clustering_kwargs,
    ) -> None:
        self.all_local_skills: list[Skill] = []
        self.evolved_skills: list[Skill] = []
        self.round: int = 0
        self.min_cluster_size = min_cluster_size
        self._clusterer: ClusteringStrategy = get_clustering_strategy(
            clustering_strategy, **clustering_kwargs
        )

    # ------------------------------------------------------------------
    # Step 1: Collect
    # ------------------------------------------------------------------

    def collect_skills(self, client_skills: dict[str, list[Skill]]) -> None:
        """Collect local skills from all clients."""
        self.all_local_skills.clear()
        for skills in client_skills.values():
            self.all_local_skills.extend(skills)

    # ------------------------------------------------------------------
    # Step 2: Cluster
    # ------------------------------------------------------------------

    def cluster_skills(self) -> list[list[Skill]]:
        """Group similar skills across clients using the configured clustering strategy."""
        if not self.all_local_skills:
            return []
        return self._clusterer.cluster(self.all_local_skills)

    # ------------------------------------------------------------------
    # Step 3: Evolve
    # ------------------------------------------------------------------

    def evolve(self, clusters: list[list[Skill]], task_descriptions: list[str] | None = None,
               checkpoint_path: str | None = None,
               disable_score_gap: bool = False,
               disable_consistency: bool = False,
               disable_nondegradation: bool = False,
               disable_critical_step: bool = False,
               ) -> list[Skill]:
        """Evolve each multi-skill cluster into a generalised skill.

        Includes quality gates: consistency check, score gate, non-degradation.
        If evolution fails all quality gates for a cluster, the original skills
        are preserved unchanged (not dropped, not force-evolved).

        Args:
            checkpoint_path: If provided, incrementally saves results after each
                cluster. On restart, loads existing results and skips completed clusters.
            disable_score_gap: Skip gate 0.
            disable_consistency: Skip gate 1.
            disable_nondegradation: Skip gate 3.
            disable_critical_step: Skip gate 4.
        """
        import json as _json
        from pathlib import Path as _Path

        evolved: list[Skill] = []
        start_idx = 0

        # Resume from checkpoint if exists
        if checkpoint_path and _Path(checkpoint_path).exists():
            with open(checkpoint_path, encoding="utf-8") as f:
                saved = _json.load(f)
            evolved = [Skill.from_dict(d) for d in saved.get("skills", [])]
            start_idx = saved.get("next_cluster", 0)
            print(f"    [evolve] Resuming from cluster {start_idx}/{len(clusters)} "
                  f"({len(evolved)} skills so far)")

        for i, cluster in enumerate(clusters):
            if i < start_idx:
                continue

            if len(cluster) >= self.min_cluster_size:
                skill = evolve_skill_cluster(
                    cluster,
                    task_descriptions=task_descriptions,
                    disable_score_gap=disable_score_gap,
                    disable_consistency=disable_consistency,
                    disable_nondegradation=disable_nondegradation,
                    disable_critical_step=disable_critical_step,
                )
                if skill is not None:
                    evolved.append(skill)
                else:
                    # Evolution failed quality gates — preserve original skills
                    evolved.extend(cluster)
            else:
                # Singleton — keep as-is
                evolved.extend(cluster)

            # Save checkpoint after each cluster
            if checkpoint_path:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    _json.dump({
                        "next_cluster": i + 1,
                        "total_clusters": len(clusters),
                        "skills": [sk.to_dict() for sk in evolved],
                    }, f, ensure_ascii=False)

        self.evolved_skills = evolved
        return evolved

    # ------------------------------------------------------------------
    # Step 4: Evaluate (optional, now integrated into evolve)
    # ------------------------------------------------------------------

    def evaluate(self, task_descriptions: list[str]) -> list[Skill]:
        """Re-score evolved skills (if not already scored during evolution)."""
        for skill in self.evolved_skills:
            if skill.score == 0.0:  # Only score if not already scored
                skill.score = evaluate_skill(skill, task_descriptions)
        self.evolved_skills.sort(key=lambda s: s.score, reverse=True)
        return self.evolved_skills

    # ------------------------------------------------------------------
    # Full round
    # ------------------------------------------------------------------

    def run_round(
        self,
        client_skills: dict[str, list[Skill]],
        task_descriptions: list[str] | None = None,
    ) -> list[Skill]:
        """Execute one full federated evolution round.

        Args:
            client_skills: client_name -> local skills.
            task_descriptions: (optional) task descriptions for evaluation.

        Returns:
            Evolved skills produced this round.
        """
        self.round += 1
        self.collect_skills(client_skills)
        clusters = self.cluster_skills()
        evolved = self.evolve(clusters, task_descriptions=task_descriptions)
        if task_descriptions:
            self.evaluate(task_descriptions)
        return evolved
