"""
Feature extraction for teachability prediction.

Two tiers of features:
- Tier 1: Structural features (fast, interpretable)
- Tier 2: Embedding features (semantic, requires model inference)
"""

from .tier1_structural import (
    StructuralFeatures,
    extract_structural_features,
    extract_batch as extract_structural_batch,
    compute_action_space_entropy,
    estimate_task_complexity,
    count_constraints,
    get_feature_names,
)

from .tier2_embeddings import (
    EmbeddingFeatures,
    extract_embedding_features,
    extract_batch as extract_embedding_batch,
    cosine_similarity,
    SentenceTransformerEmbedder,
    PolicyEmbedder,
    MockEmbedder,
    create_embedder,
)

__all__ = [
    # Tier 1
    "StructuralFeatures",
    "extract_structural_features",
    "extract_structural_batch",
    "compute_action_space_entropy",
    "estimate_task_complexity",
    "count_constraints",
    "get_feature_names",
    # Tier 2
    "EmbeddingFeatures",
    "extract_embedding_features",
    "extract_embedding_batch",
    "cosine_similarity",
    "SentenceTransformerEmbedder",
    "PolicyEmbedder",
    "MockEmbedder",
    "create_embedder",
]
