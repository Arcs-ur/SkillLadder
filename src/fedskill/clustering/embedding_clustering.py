"""Method B: Embedding-based clustering.

Uses sentence embeddings + agglomerative clustering for reproducible,
scalable skill grouping. No LLM calls needed for clustering itself.
"""

from __future__ import annotations

import numpy as np
from fedskill.clustering.base import ClusteringStrategy
from fedskill.models.skill import Skill


def _skill_text(s: Skill) -> str:
    """Build a rich text representation of a skill for embedding."""
    parts = [s.name, s.description]
    if s.when_to_use:
        parts.append(s.when_to_use)
    return " | ".join(parts)


def _simple_embeddings(texts: list[str]) -> np.ndarray:
    """Compute embeddings using sentence-transformers (lazy import).

    Falls back to TF-IDF if sentence-transformers is not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    except ImportError:
        # Fallback: TF-IDF vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        # Normalize
        from sklearn.preprocessing import normalize
        return normalize(matrix.toarray(), norm="l2")


class EmbeddingClustering(ClusteringStrategy):
    """Embedding-based agglomerative clustering.

    Args:
        distance_threshold: Maximum distance for merging clusters (default: 0.6).
            Lower = tighter clusters, higher = more merging.
        max_cluster_size: Maximum number of skills per cluster (default: 8).
            Clusters exceeding this are recursively split.
    """

    def __init__(self, distance_threshold: float = 0.6, max_cluster_size: int = 8):
        self.distance_threshold = distance_threshold
        self.max_cluster_size = max_cluster_size

    def cluster(self, skills: list[Skill]) -> list[list[Skill]]:
        if len(skills) <= 1:
            return [skills] if skills else []

        # Compute embeddings
        texts = [_skill_text(s) for s in skills]
        embeddings = _simple_embeddings(texts)

        # Agglomerative clustering
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        # Group by label
        groups: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(idx)

        # Build clusters with max size enforcement
        clusters = []
        for indices in groups.values():
            cluster_skills = [skills[i] for i in indices]
            if len(cluster_skills) <= self.max_cluster_size:
                clusters.append(cluster_skills)
            else:
                # Recursively split oversized clusters
                sub_embeddings = embeddings[indices]
                sub_clusters = self._split(
                    cluster_skills, sub_embeddings, self.max_cluster_size
                )
                clusters.extend(sub_clusters)

        return clusters

    def _split(
        self, skills: list[Skill], embeddings: np.ndarray, max_size: int
    ) -> list[list[Skill]]:
        """Recursively split a cluster that exceeds max_size."""
        if len(skills) <= max_size:
            return [skills]

        from sklearn.cluster import KMeans
        n_clusters = max(2, len(skills) // max_size + 1)
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        labels = km.fit_predict(embeddings)

        groups: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(idx)

        result = []
        for indices in groups.values():
            sub_skills = [skills[i] for i in indices]
            if len(sub_skills) <= max_size:
                result.append(sub_skills)
            else:
                result.extend(
                    self._split(sub_skills, embeddings[indices], max_size)
                )
        return result

    def name(self) -> str:
        return f"Embedding(thresh={self.distance_threshold},max={self.max_cluster_size})"
