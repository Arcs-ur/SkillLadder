"""Method C: Open tag-based clustering.

LLM discovers natural categories from the skill set (no predefined taxonomy),
then assigns each skill. Fully adaptive to any domain.
"""

from __future__ import annotations

import json
from collections import defaultdict
from fedskill.clustering.base import ClusteringStrategy
from fedskill.models.skill import Skill
from fedskill.utils.llm import llm_call


DISCOVER_AND_ASSIGN_SYSTEM = """You are a skill organizer.

Step 1: Read all the skills below and discover what semantic categories naturally emerge.
        Categories should capture the CORE STRATEGY type, not the domain. 
        Good examples: "error-recovery-and-fallback", "input-validation-before-action", 
                       "iterative-search-refinement", "multi-step-verification"
        Bad examples: "general", "misc", "other" (too vague)
        Aim for 3-10 categories depending on how many distinct strategy types exist.

Step 2: Assign each skill to exactly ONE category.

Return a JSON object:
{
  "categories": ["category-1", "category-2", ...],
  "assignments": ["category-1", "category-2", ...]  // one per skill, in order
}

Return ONLY the JSON object, no extra text."""


class TagClustering(ClusteringStrategy):
    """Open tag-based clustering with LLM-discovered categories.

    Unlike fixed-taxonomy approaches, the LLM discovers what categories
    naturally emerge from the specific skill set, then assigns each skill.

    Args:
        max_cluster_size: Maximum skills per cluster (default: 8).
    """

    def __init__(self, max_cluster_size: int = 8):
        self.max_cluster_size = max_cluster_size
        self._discovered_categories: list[str] = []

    def cluster(self, skills: list[Skill]) -> list[list[Skill]]:
        if not skills:
            return []

        tags = self._discover_and_assign(skills)

        # Assign category back to each skill
        for skill, tag in zip(skills, tags):
            skill.category = tag

        groups: dict[str, list[Skill]] = defaultdict(list)
        for skill, tag in zip(skills, tags):
            groups[tag].append(skill)

        clusters = []
        for group in groups.values():
            if len(group) <= self.max_cluster_size:
                clusters.append(group)
            else:
                for i in range(0, len(group), self.max_cluster_size):
                    clusters.append(group[i:i + self.max_cluster_size])

        return clusters

    def _discover_and_assign(self, skills: list[Skill]) -> list[str]:
        """LLM discovers categories and assigns skills in one call."""
        skills_text = "\n".join(
            f"{i+1}. [{s.source_task or '?'}] {s.name}: {s.description[:120]}"
            + (f" | When: {s.when_to_use[:80]}" if s.when_to_use else "")
            for i, s in enumerate(skills)
        )

        prompt = f"Organize these {len(skills)} skills into natural categories:\n\n{skills_text}"

        raw = llm_call(prompt, system_prompt=DISCOVER_AND_ASSIGN_SYSTEM, temperature=0.3)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            result = json.loads(raw)
            categories = result.get("categories", [])
            assignments = result.get("assignments", [])
            self._discovered_categories = categories

            if isinstance(assignments, list) and len(assignments) == len(skills):
                return [str(a) for a in assignments]
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: all in one group
        return ["uncategorized"] * len(skills)

    @property
    def discovered_categories(self) -> list[str]:
        return self._discovered_categories

    def name(self) -> str:
        return f"Tag-Open(max={self.max_cluster_size})"
