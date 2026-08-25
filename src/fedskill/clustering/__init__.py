"""Skill clustering strategies for federated skill evolution.

This module provides multiple clustering methods for grouping semantically
similar skills before evolution. The recommended method (and default) is
embedding-based clustering with LLM refinement.
"""

from fedskill.clustering.base import ClusteringStrategy
from fedskill.clustering.llm_clustering import LLMClustering
from fedskill.clustering.embedding_clustering import EmbeddingClustering
from fedskill.clustering.tag_clustering import TagClustering
from fedskill.clustering.hybrid_clustering import HybridClustering
from fedskill.clustering.reverse_hybrid import ReverseHybridClustering
from fedskill.clustering.random_clustering import RandomClustering

STRATEGIES = {
    "llm": LLMClustering,
    "embedding": EmbeddingClustering,
    "tag": TagClustering,
    "hybrid": HybridClustering,
    "reverse_hybrid": ReverseHybridClustering,
    "random": RandomClustering,
}


def get_clustering_strategy(name: str = "hybrid", **kwargs) -> ClusteringStrategy:
    """Get a clustering strategy by name.

    Args:
        name: One of "llm", "embedding", "tag", "hybrid".
        **kwargs: Strategy-specific parameters.

    Returns:
        A ClusteringStrategy instance.
    """
    cls = STRATEGIES.get(name)
    if cls is None:
        raise ValueError(f"Unknown clustering strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return cls(**kwargs)
