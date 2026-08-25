"""Method A: LLM one-shot clustering (original baseline).

Sends all skill descriptions to an LLM and asks it to group them.
Simple but not scalable or reproducible.
"""

from __future__ import annotations

import json
from fedskill.clustering.base import ClusteringStrategy
from fedskill.models.skill import Skill
from fedskill.utils.llm import llm_call


CLUSTER_SYSTEM = """You are a skill-clustering engine.
Given a list of skills (each with an ID and a short description), group them into clusters
of *semantically similar* skills that address the same underlying strategy or capability.
Each cluster should contain skills that, despite being from different tasks, share a common
core idea and can be meaningfully merged.

Return a JSON array of arrays, where each inner array contains skill IDs belonging to one cluster.
Skills that are unique and don't fit any cluster should appear in a singleton array.
Return ONLY the JSON array, no extra text."""


class LLMClustering(ClusteringStrategy):
    """Baseline: LLM one-shot clustering."""

    def cluster(self, skills: list[Skill]) -> list[list[Skill]]:
        if not skills:
            return []

        id_map = {s.id: s for s in skills}
        skills_text = "\n".join(
            f"- ID={s.id} | Task={s.source_task} | Name={s.name} | Desc={s.description[:120]}"
            for s in skills
        )

        raw = llm_call(
            f"### Skills to cluster\n{skills_text}",
            system_prompt=CLUSTER_SYSTEM,
            temperature=0.3,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            clusters_ids: list[list[str]] = json.loads(raw)
        except json.JSONDecodeError:
            clusters_ids = [[s.id] for s in skills]

        clusters = []
        for id_group in clusters_ids:
            group = [id_map[sid] for sid in id_group if sid in id_map]
            if group:
                clusters.append(group)
        return clusters

    def name(self) -> str:
        return "LLM-OneShot"
