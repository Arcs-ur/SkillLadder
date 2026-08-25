"""Method E: Reverse Hybrid — Embedding first, then LLM refinement.

Pipeline:
  1. Embedding-based agglomerative clustering (reproducible initial grouping)
  2. LLM reviews each cluster and splits/merges as needed (semantic refinement)
  3. Max cluster size enforced

This is the reverse of the default Hybrid (LLM tag first, embedding second).
Used in ablation to compare ordering effects.
"""

from __future__ import annotations

import json
from fedskill.clustering.base import ClusteringStrategy
from fedskill.clustering.embedding_clustering import EmbeddingClustering, _skill_text
from fedskill.models.skill import Skill
from fedskill.utils.llm import llm_call


REFINE_SYSTEM = """You are a skill cluster reviewer.
You receive a cluster of skills that were grouped by embedding similarity.
Your job is to check if they truly belong together semantically.

For each cluster, decide ONE action:
- "keep": All skills share a common core strategy. Keep as-is.
- "split": Skills address fundamentally different strategies despite surface similarity.
  Return sub-groups.
- "rename": Skills belong together but the grouping needs a category label.

Return a JSON object:
{
  "action": "keep" | "split",
  "category": "descriptive-category-name",
  "sub_groups": [[0,1,2], [3,4]]  // only if action="split", indices into the input list
}
Return ONLY the JSON object."""


class ReverseHybridClustering(ClusteringStrategy):
    """Reverse Hybrid: Embedding clustering first, then LLM refinement.

    Stage 1 (Coarse): Embedding similarity + agglomerative clustering
                      produces reproducible initial clusters.
    Stage 2 (Refine): LLM reviews each cluster for semantic coherence,
                      splitting clusters that group different strategies.

    Args:
        distance_threshold: Embedding distance threshold (default: 0.5).
        max_cluster_size: Maximum skills per cluster (default: 8).
    """

    def __init__(self, distance_threshold: float = 0.5, max_cluster_size: int = 8):
        self.distance_threshold = distance_threshold
        self.max_cluster_size = max_cluster_size
        self._embedding_clusterer = EmbeddingClustering(
            distance_threshold=distance_threshold,
            max_cluster_size=max_cluster_size,
        )

    def cluster(self, skills: list[Skill]) -> list[list[Skill]]:
        if len(skills) <= 2:
            return [skills] if skills else []

        # Stage 1: Embedding-based clustering (reproducible)
        initial_clusters = self._embedding_clusterer.cluster(skills)

        # Stage 2: LLM refinement on each cluster
        refined_clusters = []
        for cluster in initial_clusters:
            if len(cluster) <= 2:
                # Too small to refine, assign category
                category = self._assign_category(cluster)
                for s in cluster:
                    s.category = category
                refined_clusters.append(cluster)
            else:
                sub_clusters = self._refine_cluster(cluster)
                refined_clusters.extend(sub_clusters)

        return refined_clusters

    def _refine_cluster(self, cluster: list[Skill]) -> list[list[Skill]]:
        """LLM reviews a cluster and optionally splits it."""
        skills_text = "\n".join(
            f"{i}. [{s.source_task or '?'}] {s.name}: {s.description[:120]}"
            for i, s in enumerate(cluster)
        )

        prompt = f"Review this cluster of {len(cluster)} skills:\n\n{skills_text}"
        raw = llm_call(prompt, system_prompt=REFINE_SYSTEM, temperature=0.2)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            result = json.loads(raw)
            action = result.get("action", "keep")
            category = result.get("category", "uncategorized")

            if action == "split" and "sub_groups" in result:
                sub_groups = result["sub_groups"]
                sub_clusters = []
                for indices in sub_groups:
                    sub = [cluster[i] for i in indices if i < len(cluster)]
                    if sub:
                        for s in sub:
                            s.category = category
                        sub_clusters.append(sub)
                return sub_clusters if sub_clusters else [cluster]
            else:
                for s in cluster:
                    s.category = category
                return [cluster]

        except (json.JSONDecodeError, KeyError, IndexError):
            return [cluster]

    def _assign_category(self, cluster: list[Skill]) -> str:
        """Quick category assignment for small clusters."""
        names = ", ".join(s.name for s in cluster)
        raw = llm_call(
            f"What category best describes these skills: {names}? "
            f"Return ONE short hyphenated category name only.",
            temperature=0.2,
        )
        return raw.strip().lower().replace(" ", "-")[:50] or "uncategorized"

    def name(self) -> str:
        return f"ReverseHybrid(thresh={self.distance_threshold},max={self.max_cluster_size})"
