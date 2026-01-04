#!/usr/bin/env python3
"""
Batch Feature Extraction for Dataset Generation.

This script extracts all features (Tier 1 structural + Tier 2 embeddings)
for snapshots and generates the labeled_snapshots.parquet file.

Features extracted:
- Tier 1 (Structural): entropy, margin, step_index, action_counts, page_type
- Tier 2 (Embeddings): observation embeddings, task embeddings

This is required for:
1. Teachability predictor training (Phase 4)
2. Figure 1 (Teachability Landscape)
3. Analysis and visualization

Usage:
    python scripts/phase1/extract_features_batch.py \
        --snapshots results/phase1/raw_snapshots.json \
        --output results/phase1/labeled_snapshots.parquet \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.common import setup_logging, set_seed, ProgressTracker
from src.features.tier1_structural import extract_tier1_features, Tier1Features
from src.features.tier2_embeddings import extract_tier2_embeddings, Tier2Features

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractionConfig:
    """Configuration for batch feature extraction."""
    
    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Processing
    batch_size: int = 32
    max_observation_length: int = 2000
    
    # Caching
    cache_embeddings: bool = True
    cache_dir: str = "cache/embeddings"
    
    # Output
    include_raw_embeddings: bool = False  # If True, store full embedding vectors
    compress_embeddings: bool = True  # Use PCA to reduce embedding dimensionality


class EmbeddingCache:
    """Cache for computed embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load existing cache from disk."""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self._cache = {}
    
    def save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(self._cache, f)
        logger.info(f"Saved {len(self._cache)} embeddings to cache")
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, embedding: np.ndarray):
        """Set embedding in cache."""
        self._cache[key] = embedding


class BatchFeatureExtractor:
    """
    Extract features for a batch of snapshots.
    """
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self._embedding_model = None
        self._cache = None
        
        if config.cache_embeddings:
            self._cache = EmbeddingCache(Path(config.cache_dir))
    
    def _load_embedding_model(self):
        """Lazily load the embedding model."""
        if self._embedding_model is not None:
            return
        
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self._embedding_model = SentenceTransformer(self.config.embedding_model)
        logger.info("Embedding model loaded")
    
    def extract_all_features(
        self,
        snapshots: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        Extract all features for a list of snapshots.
        
        Returns DataFrame with all features.
        """
        self._load_embedding_model()
        
        all_features = []
        n_snapshots = len(snapshots)
        
        # Process in batches
        for batch_start in range(0, n_snapshots, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, n_snapshots)
            batch = snapshots[batch_start:batch_end]
            
            # Extract features for batch
            batch_features = self._extract_batch_features(batch)
            all_features.extend(batch_features)
            
            if progress_callback:
                progress_callback(batch_end, n_snapshots)
        
        # Save cache if enabled
        if self._cache:
            self._cache.save_cache()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        return df
    
    def _extract_batch_features(
        self,
        batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract features for a batch of snapshots."""
        
        # Collect texts for batch embedding
        observations = []
        tasks = []
        
        for snapshot in batch:
            obs = snapshot.get("observation", snapshot.get("state", ""))
            obs = obs[:self.config.max_observation_length]
            observations.append(obs)
            
            task = snapshot.get("task_description", "")
            tasks.append(task)
        
        # Compute embeddings (with caching)
        obs_embeddings = self._compute_embeddings_batch(observations, "obs")
        task_embeddings = self._compute_embeddings_batch(tasks, "task")
        
        # Extract features for each snapshot
        batch_features = []
        
        for i, snapshot in enumerate(batch):
            features = self._extract_snapshot_features(
                snapshot,
                obs_embeddings[i],
                task_embeddings[i],
            )
            batch_features.append(features)
        
        return batch_features
    
    def _compute_embeddings_batch(
        self,
        texts: List[str],
        prefix: str,
    ) -> List[np.ndarray]:
        """Compute embeddings for a batch of texts."""
        
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = f"{prefix}_{hash(text)}"
            
            if self._cache:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    embeddings.append(cached)
                    continue
            
            embeddings.append(None)  # Placeholder
            texts_to_compute.append(text)
            indices_to_compute.append(i)
        
        # Compute missing embeddings
        if texts_to_compute:
            computed = self._embedding_model.encode(
                texts_to_compute,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
            
            for j, idx in enumerate(indices_to_compute):
                embedding = computed[j]
                embeddings[idx] = embedding
                
                if self._cache:
                    cache_key = f"{prefix}_{hash(texts[idx])}"
                    self._cache.set(cache_key, embedding)
        
        return embeddings
    
    def _extract_snapshot_features(
        self,
        snapshot: Dict[str, Any],
        obs_embedding: np.ndarray,
        task_embedding: np.ndarray,
    ) -> Dict[str, Any]:
        """Extract all features for a single snapshot."""
        
        features = {}
        
        # Copy basic fields
        for field in ["snapshot_id", "id", "trajectory_id", "task_id", "step_idx"]:
            if field in snapshot:
                features[field] = snapshot[field]
        
        # Tier 1: Structural features
        tier1 = extract_tier1_features(snapshot)
        features.update({
            "entropy": tier1.entropy,
            "margin": tier1.margin,
            "top_k_spread": tier1.top_k_spread,
            "step_index": tier1.step_index,
            "normalized_step": tier1.normalized_step,
            "search_count": tier1.action_counts.get("search", 0),
            "click_count": tier1.action_counts.get("click", 0),
            "page_type": tier1.page_type,
        })
        
        # Add any pre-computed labels
        for label_field in ["uncertainty", "leverage", "quadrant", "ELP_net", "route"]:
            if label_field in snapshot:
                features[label_field] = snapshot[label_field]
        
        # Tier 2: Embedding features
        if self.config.include_raw_embeddings:
            features["obs_embedding"] = obs_embedding.tolist()
            features["task_embedding"] = task_embedding.tolist()
        
        # Compute derived embedding features
        # Cosine similarity between observation and task
        obs_norm = obs_embedding / (np.linalg.norm(obs_embedding) + 1e-8)
        task_norm = task_embedding / (np.linalg.norm(task_embedding) + 1e-8)
        features["obs_task_similarity"] = float(np.dot(obs_norm, task_norm))
        
        # Embedding norms (can indicate information content)
        features["obs_embedding_norm"] = float(np.linalg.norm(obs_embedding))
        features["task_embedding_norm"] = float(np.linalg.norm(task_embedding))
        
        # First few PCA components of embeddings (for visualization)
        if self.config.compress_embeddings:
            # Store first 8 components
            features["obs_emb_pca"] = obs_embedding[:8].tolist()
            features["task_emb_pca"] = task_embedding[:8].tolist()
        
        return features


def load_snapshots(input_path: Path) -> List[Dict[str, Any]]:
    """Load snapshots from file."""
    
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
        return df.to_dict("records")
    elif input_path.suffix == ".json":
        with open(input_path) as f:
            return json.load(f)
    elif input_path.suffix == ".jsonl":
        snapshots = []
        with open(input_path) as f:
            for line in f:
                snapshots.append(json.loads(line))
        return snapshots
    else:
        raise ValueError(f"Unsupported format: {input_path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch feature extraction for snapshots"
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        required=True,
        help="Input snapshots file (json, jsonl, or parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet file",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--include-raw-embeddings",
        action="store_true",
        help="Include full embedding vectors in output",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/embeddings"),
        help="Cache directory",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    # Load snapshots
    logger.info(f"Loading snapshots from {args.snapshots}")
    snapshots = load_snapshots(args.snapshots)
    logger.info(f"Loaded {len(snapshots)} snapshots")
    
    # Configure extraction
    config = FeatureExtractionConfig(
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        cache_embeddings=not args.no_cache,
        cache_dir=str(args.cache_dir),
        include_raw_embeddings=args.include_raw_embeddings,
    )
    
    # Extract features
    extractor = BatchFeatureExtractor(config)
    
    tracker = ProgressTracker(len(snapshots), "Extracting features")
    
    def progress_callback(completed, total):
        tracker.update(completed - tracker.current if hasattr(tracker, 'current') else 1)
        tracker.current = completed
    
    tracker.current = 0
    
    df = extractor.extract_all_features(snapshots, progress_callback)
    
    tracker.finish()
    
    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    
    logger.info(f"Saved {len(df)} snapshots with features to {args.output}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Feature Extraction Summary")
    print("=" * 50)
    print(f"Snapshots processed: {len(df)}")
    print(f"Features extracted: {len(df.columns)}")
    print(f"\nFeature columns:")
    for col in sorted(df.columns):
        dtype = df[col].dtype
        print(f"  {col}: {dtype}")
    
    # Statistics for key features
    print("\nKey feature statistics:")
    for col in ["entropy", "margin", "leverage", "obs_task_similarity"]:
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
    
    if "quadrant" in df.columns:
        print(f"\nQuadrant distribution:")
        for q, count in df["quadrant"].value_counts().items():
            print(f"  {q}: {count} ({count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
