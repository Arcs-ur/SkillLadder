"""Random clustering strategy for ablation experiments."""

from __future__ import annotations

import random
from fedskill.clustering.base import ClusteringStrategy
from fedskill.models.skill import Skill


class RandomClustering(ClusteringStrategy):
    """Randomly assign skills to clusters of a fixed size range."""

    def __init__(self, min_size: int = 2, max_size: int = 8, **kwargs):
        self.min_size = min_size
        self.max_size = max_size

    def cluster(self, skills: list[Skill]) -> list[list[Skill]]:
        skills = list(skills)
        random.shuffle(skills)
        clusters = []
        i = 0
        while i < len(skills):
            size = random.randint(self.min_size, self.max_size)
            chunk = skills[i:i + size]
            clusters.append(chunk)
            i += size
        return clusters
