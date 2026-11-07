"""Federated aggregator — groups similar skills and triggers evolution."""

from __future__ import annotations

import random
from pathlib import Path
from fedskill.models.skill import Skill
from fedskill.server.evolver import evolve_skill_cluster, evaluate_skill
from fedskill.clustering import get_clustering_strategy, ClusteringStrategy
from fedskill.utils.run_manifest import (
    atomic_write_json, cluster_identity, fingerprint, read_json,
)


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
               seed: int = 42,
               **evolution_kwargs,
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
        evolved: list[Skill] = []
        start_idx = 0
        cluster_ids = [cluster_identity(cluster) for cluster in clusters]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("Duplicate cluster identity; checkpoint cannot identify work uniquely.")
        identity = fingerprint({
            "clusters": [[s.to_dict() for s in c] for c in clusters],
            "tasks": task_descriptions, "seed": seed, "min_cluster_size": self.min_cluster_size,
            "gates": [disable_score_gap, disable_consistency, disable_nondegradation, disable_critical_step],
            "evolution": evolution_kwargs,
        })

        # Resume from checkpoint if exists
        if checkpoint_path and Path(checkpoint_path).exists():
            saved = read_json(checkpoint_path)
            start_idx = saved.get("next_cluster")
            if (
                saved.get("fingerprint") != identity
                or type(start_idx) is not int or not 0 <= start_idx <= len(clusters)
                or saved.get("completed_cluster_ids") != cluster_ids[:start_idx]
                or not isinstance(saved.get("skills"), list)
            ):
                raise ValueError(f"Invalid or mismatched checkpoint {checkpoint_path}; use a new output directory.")
            evolved = [Skill.from_dict(d) for d in saved["skills"]]
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
                    rng=random.Random(f"{seed}:{cluster_ids[i]}"),
                    **evolution_kwargs,
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
                atomic_write_json(checkpoint_path, {
                        "fingerprint": identity,
                        "completed_cluster_ids": cluster_ids[:i + 1],
                        "next_cluster": i + 1,
                        "total_clusters": len(clusters),
                        "skills": [sk.to_dict() for sk in evolved],
                    })

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
