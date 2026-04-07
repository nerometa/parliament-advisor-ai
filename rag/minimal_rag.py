"""Minimal RAG implementation using pre-computed embeddings.

This module provides zero-dependency (except numpy) vector search
for Thai parliamentary documents.

Usage:
    # Load existing index
    rag = ThaiRAG()
    results = rag.search(query_embedding, k=5)
    
The index must be pre-built using index_knowledge.py (requires sentence-transformers).
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ThaiRAG:
    """Minimal Thai RAG with pre-computed embeddings.
    
    Zero ML dependencies at runtime - only numpy required.
    Embeddings must be pre-computed using index_knowledge.py
    """
    
    def __init__(self, store_path: str = "data/vector_store"):
        """Initialize RAG from pre-computed embeddings.
        
        Args:
            store_path: Path to directory containing embeddings.npy and metadata.json
        """
        self.store_path = Path(store_path)
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        self._loaded = False
        
        if self.store_path.exists():
            self._load()
    
    def _load(self) -> bool:
        """Load pre-computed embeddings and metadata."""
        try:
            embeddings_path = self.store_path / "embeddings.npy"
            metadata_path = self.store_path / "metadata.json"
            
            if not embeddings_path.exists() or not metadata_path.exists():
                logger.warning(f"Vector store not found at {self.store_path}")
                return False
            
            logger.info(f"Loading embeddings from {embeddings_path}")
            self.embeddings = np.load(embeddings_path)
            
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self._loaded = True
            logger.info(f"Loaded {len(self.metadata)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if RAG is ready to serve queries."""
        return self._loaded and self.embeddings is not None and len(self.metadata) > 0
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for most similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector (numpy array)
            k: Number of results to return
            
        Returns:
            List of dicts with keys: text, metadata, score
        """
        if not self.is_ready():
            logger.error("RAG not initialized. Run index_knowledge.py first.")
            return []
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Normalize all embeddings
        embeddings_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity (dot product of normalized vectors)
        similarities = embeddings_norm @ query_norm
        
        # Get top-k indices
        k = min(k, len(self.metadata))
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Build results
        results = []
        for idx in top_k_indices:
            results.append({
                "text": self.metadata[idx].get("text", ""),
                "metadata": {
                    "source": self.metadata[idx].get("source", ""),
                    "page": self.metadata[idx].get("page", 0),
                    "article": self.metadata[idx].get("article"),
                    "token_count": self.metadata[idx].get("token_count", 0),
                },
                "score": float(similarities[idx])
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the loaded knowledge base."""
        if not self.is_ready():
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "num_chunks": len(self.metadata),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "store_path": str(self.store_path),
            "sources": list(set(m.get("source", "") for m in self.metadata)),
        }


class EmbedderInterface:
    """Interface for embedding queries at runtime.
    
    Since we use pre-computed embeddings for documents, this only needs
to handle query embedding. Can be:
    - API-based (OpenAI, etc.)
    - Pre-computed query embeddings
    - Simple keyword matching fallback
    """
    
    @staticmethod
    def from_config():
        """Create embedder based on config."""
        try:
            from sentence_transformers import SentenceTransformer
            # If sentence-transformers is available, use it
            return LocalEmbedder()
        except ImportError:
            # Fallback to API or pre-computed
            logger.warning("sentence-transformers not available, using fallback")
            return FallbackEmbedder()


class LocalEmbedder:
    """Local embedding using sentence-transformers."""
    
    def __init__(self, model_name: str = "KoonJamesZ/sentence-transformers-nina-thai-v3"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str) -> np.ndarray:
        """Embed text to vector."""
        return self.model.encode(text)


class FallbackEmbedder:
    """Fallback embedder when sentence-transformers is not available.
    
    Uses simple keyword matching or can be extended to use API.
    """
    
    def embed(self, text: str) -> np.ndarray:
        """Fallback: raises error telling user to install deps or use API."""
        raise RuntimeError(
            "Embedding not available at runtime.\n"
            "Options:\n"
            "1. Install sentence-transformers: pip install sentence-transformers\n"
            "2. Use API-based embeddings (OpenAI, etc.)\n"
            "3. Pre-compute query embeddings offline"
        )


# Backward compatibility
Retriever = ThaiRAG
