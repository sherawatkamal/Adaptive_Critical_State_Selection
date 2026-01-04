"""
Tier 2: Embedding features for teachability prediction.

These features require model inference but capture semantic
information not available from structural features.
"""

from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbedderProtocol(Protocol):
    """Protocol for text embedding models."""
    
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class EmbeddingFeatures:
    """Tier 2 embedding features for a snapshot."""
    
    state_embedding: np.ndarray
    task_embedding: np.ndarray
    action_embedding: Optional[np.ndarray] = None
    
    # Derived similarity features
    state_task_similarity: float = 0.0
    state_action_similarity: float = 0.0
    
    # Embedding statistics
    state_norm: float = 0.0
    task_norm: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Concatenate all embeddings into single vector."""
        vectors = [self.state_embedding, self.task_embedding]
        
        if self.action_embedding is not None:
            vectors.append(self.action_embedding)
        
        return np.concatenate(vectors)
    
    def get_similarity_features(self) -> np.ndarray:
        """Get just the similarity-based features."""
        return np.array([
            self.state_task_similarity,
            self.state_action_similarity,
            self.state_norm,
            self.task_norm,
        ])
    
    def to_dict(self) -> dict:
        return {
            "state_embedding": self.state_embedding.tolist(),
            "task_embedding": self.task_embedding.tolist(),
            "action_embedding": self.action_embedding.tolist() if self.action_embedding is not None else None,
            "state_task_similarity": self.state_task_similarity,
            "state_action_similarity": self.state_action_similarity,
            "state_norm": self.state_norm,
            "task_norm": self.task_norm,
        }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


class SentenceTransformerEmbedder:
    """Embedder using sentence-transformers library."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
    
    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers required for embedding features")
        return self._model
    
    def embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        return model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True)


class PolicyEmbedder:
    """Embedder using the policy model's hidden states."""
    
    def __init__(self, model, tokenizer, layer: int = -1):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
    
    def embed(self, text: str) -> np.ndarray:
        import torch
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[self.layer]
            # Mean pooling
            embedding = hidden.mean(dim=1).squeeze().cpu().numpy()
        
        return embedding
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


class MockEmbedder:
    """Mock embedder for testing."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
    
    def embed(self, text: str) -> np.ndarray:
        # Deterministic based on text content
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dim).astype(np.float32)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


class HashingEmbedder:
    """Fast, dependency-free embedder using feature hashing.

    This is intentionally *very* lightweight so that Phase 4 can run in
    environments without sentence-transformers.

    The embedding is a signed bag-of-tokens with deterministic hashing.
    """

    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    def _token_hash(self, token: str) -> int:
        import hashlib

        h = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
        return int(h, 16)

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec

        # Basic whitespace tokenization is good enough for hashing.
        for tok in text.split():
            hv = self._token_hash(tok)
            idx = hv % self.dim
            sign = 1.0 if ((hv // self.dim) % 2) == 0 else -1.0
            vec[idx] += sign

        # L2 normalize for stability.
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts], axis=0)


def extract_embedding_features(
    snapshot: dict,
    embedder: EmbedderProtocol,
) -> EmbeddingFeatures:
    """
    Extract tier 2 embedding features from a snapshot.
    
    Args:
        snapshot: Snapshot dictionary
        embedder: Text embedding model
        
    Returns:
        EmbeddingFeatures instance
    """
    # Extract texts
    state_text = snapshot.get("observation") or snapshot.get("state") or ""
    task_description = (
        snapshot.get("instruction_text") or 
        snapshot.get("task", {}).get("description") or 
        snapshot.get("task_description") or
        snapshot.get("agent_prefix") or  # v8 fallback: mining stores instruction in agent_prefix
        ""
    )
    if not task_description:
        logger.warning(
            "tier2_embeddings: No instruction_text/task_description/agent_prefix found in snapshot. "
            "Task embedding will be empty, which may degrade predictor quality. "
            f"Snapshot keys: {list(snapshot.keys())}"
        )
    policy_action = snapshot.get("policy_action", "")
    
    # Compute embeddings
    state_embedding = embedder.embed(state_text)
    task_embedding = embedder.embed(task_description)
    
    action_embedding = None
    if policy_action:
        action_embedding = embedder.embed(policy_action)
    
    # Compute similarity features
    state_task_similarity = cosine_similarity(state_embedding, task_embedding)
    
    state_action_similarity = 0.0
    if action_embedding is not None:
        state_action_similarity = cosine_similarity(state_embedding, action_embedding)
    
    # Embedding statistics
    state_norm = float(np.linalg.norm(state_embedding))
    task_norm = float(np.linalg.norm(task_embedding))
    
    return EmbeddingFeatures(
        state_embedding=state_embedding,
        task_embedding=task_embedding,
        action_embedding=action_embedding,
        state_task_similarity=state_task_similarity,
        state_action_similarity=state_action_similarity,
        state_norm=state_norm,
        task_norm=task_norm,
    )


def extract_batch(
    snapshots: list[dict],
    embedder: EmbedderProtocol,
    batch_size: int = 32,
) -> list[EmbeddingFeatures]:
    """
    Extract embedding features for a batch of snapshots.
    
    Uses batched embedding for efficiency.
    
    Args:
        snapshots: List of snapshot dicts
        embedder: Text embedding model
        batch_size: Batch size for embedding
        
    Returns:
        List of EmbeddingFeatures
    """
    # Collect all texts
    state_texts = [
        s.get("observation") or s.get("state") or "" 
        for s in snapshots
    ]
    task_texts = []
    for i, s in enumerate(snapshots):
        task_desc = (
            s.get("instruction_text") or 
            s.get("task", {}).get("description") or 
            s.get("task_description") or
            s.get("agent_prefix") or  # v8 fallback
            ""
        )
        if not task_desc:
            logger.warning(
                f"tier2_embeddings.extract_batch: Snapshot {i} has no instruction_text/task_description/agent_prefix. "
                "Task embedding will be empty, which may degrade predictor quality."
            )
        task_texts.append(task_desc)
    action_texts = [s.get("policy_action", "") for s in snapshots]
    
    # Batch embed
    state_embeddings = embedder.embed_batch(state_texts)
    task_embeddings = embedder.embed_batch(task_texts)
    action_embeddings = embedder.embed_batch(action_texts)
    
    # Build features
    results = []
    for i in range(len(snapshots)):
        state_emb = state_embeddings[i]
        task_emb = task_embeddings[i]
        action_emb = action_embeddings[i] if action_texts[i] else None
        
        state_task_sim = cosine_similarity(state_emb, task_emb)
        state_action_sim = cosine_similarity(state_emb, action_emb) if action_emb is not None else 0.0
        
        results.append(EmbeddingFeatures(
            state_embedding=state_emb,
            task_embedding=task_emb,
            action_embedding=action_emb,
            state_task_similarity=state_task_sim,
            state_action_similarity=state_action_sim,
            state_norm=float(np.linalg.norm(state_emb)),
            task_norm=float(np.linalg.norm(task_emb)),
        ))
    
    return results


def create_embedder(
    embedder_type: str = "sentence_transformer",
    model_name: Optional[str] = None,
    dim: int = 256,
    policy_model=None,
    tokenizer=None,
) -> EmbedderProtocol:
    """
    Factory function to create embedder.
    
    Args:
        embedder_type: Type of embedder (sentence_transformer, policy, mock, hashing)
        model_name: Model name for sentence transformer
        dim: Embedding dimension (for hashing and mock embedders)
        policy_model: Policy model for policy embedder
        tokenizer: Tokenizer for policy embedder
        
    Returns:
        Embedder instance
    """
    if embedder_type == "hashing":
        return HashingEmbedder(dim=dim)
    if embedder_type == "sentence_transformer":
        return SentenceTransformerEmbedder(model_name or "all-MiniLM-L6-v2")
    elif embedder_type == "policy":
        if policy_model is None or tokenizer is None:
            raise ValueError("policy_model and tokenizer required for policy embedder")
        return PolicyEmbedder(policy_model, tokenizer)
    elif embedder_type == "mock":
        return MockEmbedder(dim=dim)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")
