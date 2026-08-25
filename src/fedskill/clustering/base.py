"""Abstract base class for clustering strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from fedskill.models.skill import Skill


class ClusteringStrategy(ABC):
    """Base class for skill clustering strategies."""

    @abstractmethod
    def cluster(self, skills: list[Skill]) -> list[list[Skill]]:
        """Group skills into clusters of semantically similar skills.

        Args:
            skills: All skills to cluster.

        Returns:
            List of clusters, each cluster is a list of Skills.
        """

    def name(self) -> str:
        return self.__class__.__name__
