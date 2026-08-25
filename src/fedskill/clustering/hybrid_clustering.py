"""Method D (recommended): Hybrid clustering with open categories.

Pipeline:
  1. LLM discovers natural semantic categories from a representative subsample
  2. LLM assigns skills to discovered categories in batches
  3. Within each category, embedding-based agglomerative clustering for fine-grained grouping
  4. Max cluster size enforced

Advantages over fixed-tag approaches:
  - Categories adapt to the actual skill distribution (no "general" garbage bin)
  - Embedding sub-clustering is reproducible and handles within-category diversity
  - Max cluster size prevents quality degradation during evolution
  - Batched LLM calls scale to thousands of skills
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from fedskill.clustering.base import ClusteringStrategy
from fedskill.clustering.embedding_clustering import EmbeddingClustering
from fedskill.models.skill import Skill
from fedskill.utils.llm import llm_call


DISCOVER_SYSTEM = """You are a skill organizer.

Given a representative sample of agent skills, discover what semantic categories naturally emerge.
Categories should capture the CORE STRATEGY type (not the task domain).

Good category names describe a reusable pattern:
  "error-recovery-and-fallback", "pre-action-validation", 
  "iterative-search-refinement", "safe-file-output", "multi-step-verification"

Bad category names are too vague or domain-specific:
  "general", "misc", "coding", "search" 

Rules:
- Discover 3-15 categories (proportional to skill diversity)
- Each category should have at least 2 skills (merge tiny categories)
- Return ONLY the JSON array of category names

Return a JSON object:
{
  "categories": ["cat-name-1", "cat-name-2", ...]
}
Return ONLY the JSON object."""


ASSIGN_SYSTEM = """You are a skill classifier.

Given a skill name and description, classify it into exactly ONE of the predefined categories.
Pick the category that best captures the skill's CORE STRATEGY.

Return a JSON object:
{
  "category": "cat-name"
}
Return ONLY the JSON object."""

# Maximum skills per LLM call for assignment
ASSIGN_BATCH_SIZE = 20


class HybridClustering(ClusteringStrategy):
    """Hybrid: LLM-discovered open categories + Embedding-based fine clustering.

    Two-stage approach:
    Stage 1 (Coarse): LLM discovers natural categories from the actual skill set
                      and assigns each skill. No predefined taxonomy.
    Stage 2 (Fine):   Within each category, embedding similarity + agglomerative
                      clustering produces tight, reproducible sub-clusters.

    Args:
        distance_threshold: Embedding distance threshold for sub-clustering (default: 0.5).
        max_cluster_size: Maximum skills per cluster (default: 8).
    """

    def __init__(self, distance_threshold: float = 0.5, max_cluster_size: int = 8):
        self.distance_threshold = distance_threshold
        self.max_cluster_size = max_cluster_size
        self._embedding_clusterer = EmbeddingClustering(
            distance_threshold=distance_threshold,
            max_cluster_size=max_cluster_size,
        )
        self._discovered_categories: list[str] = []
        self._category_sizes: dict[str, int] = {}

    def cluster(self, skills: list[Skill]) -> list[list[Skill]]:
        if len(skills) <= 2:
            return [skills] if skills else []

        # Stage 1: LLM discovers categories & assigns
        tags = self._discover_and_assign(skills)

        # Assign discovered category back to each skill
        for skill, tag in zip(skills, tags):
            skill.category = tag

        tag_groups: dict[str, list[Skill]] = defaultdict(list)
        for skill, tag in zip(skills, tags):
            tag_groups[tag].append(skill)

        self._category_sizes = {tag: len(group) for tag, group in tag_groups.items()}

        # Stage 2: Embedding fine-clustering within each category
        all_clusters = []
        for tag, group in tag_groups.items():
            if len(group) <= 2:
                all_clusters.append(group)
            else:
                sub_clusters = self._embedding_clusterer.cluster(group)
                all_clusters.extend(sub_clusters)

        return all_clusters

    def _discover_and_assign(self, skills: list[Skill]) -> list[str]:
        """Two-step: discover categories from all skills, then assign each via embedding similarity."""
        # Step 1: Discover categories using all skill names + short descriptions
        categories = self._discover_categories(skills)
        if not categories:
            return ["uncategorized"] * len(skills)
        self._discovered_categories = categories

        # Step 2: Assign each skill to the nearest category via embedding similarity
        assignments = self._assign_by_embedding(skills, categories)
        return assignments

    def _skill_summary(self, idx: int, skill: Skill) -> str:
        """Short one-line summary for assignment: index, name, brief description."""
        desc = (skill.description or "")[:60].replace("\n", " ")
        return f"{idx}. {skill.name}: {desc}"

    def _discover_categories(self, skills: list[Skill]) -> list[str]:
        """Discover categories from ALL skills using names + short descriptions."""
        skills_text = "\n".join(
            f"{i+1}. {s.name}: {(s.description or '')[:60].replace(chr(10), ' ')}"
            for i, s in enumerate(skills)
        )
        prompt = (
            f"Here are {len(skills)} agent skills.\n"
            f"Discover what semantic categories naturally emerge:\n\n{skills_text}"
        )

        raw = llm_call(prompt, system_prompt=DISCOVER_SYSTEM, temperature=0.3)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            result = json.loads(raw)
            categories = result.get("categories", [])
            if categories:
                return categories
        except (json.JSONDecodeError, KeyError):
            pass
        return []

    def _assign_by_embedding(self, skills: list[Skill], categories: list[str]) -> list[str]:
        """Assign each skill to the most similar category using embedding cosine similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer("all-MiniLM-L6-v2")

            # Encode category names
            cat_embeddings = model.encode(categories, normalize_embeddings=True)

            # Encode skill texts (name + short description)
            skill_texts = [
                f"{s.name}: {(s.description or '')[:100]}" for s in skills
            ]
            skill_embeddings = model.encode(skill_texts, normalize_embeddings=True)

            # Cosine similarity = dot product (normalized)
            similarities = np.dot(skill_embeddings, cat_embeddings.T)  # (N, C)
            best_indices = np.argmax(similarities, axis=1)

            return [categories[idx] for idx in best_indices]

        except ImportError:
            # Fallback: assign all to first category
            return [categories[0]] * len(skills)

    @property
    def discovered_categories(self) -> list[str]:
        """Categories discovered in the last clustering run."""
        return self._discovered_categories

    @property
    def category_distribution(self) -> dict[str, int]:
        """Number of skills per discovered category."""
        return self._category_sizes

    def name(self) -> str:
        return f"Hybrid-Open(thresh={self.distance_threshold},max={self.max_cluster_size})"
