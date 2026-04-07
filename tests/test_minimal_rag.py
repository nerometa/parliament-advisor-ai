"""Tests for minimal RAG implementation (zero runtime ML dependencies)."""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path

from rag.minimal_rag import ThaiRAG, EmbedderInterface


class TestThaiRAGInitialization:
    """Tests for ThaiRAG initialization and loading."""

    def test_init_with_defaults(self):
        """Initialize with default path."""
        rag = ThaiRAG()
        assert rag.store_path == Path("data/vector_store")
        assert not rag._loaded
        assert rag.embeddings is None
        assert rag.metadata == []

    def test_init_with_custom_path(self):
        """Initialize with custom path."""
        custom_path = "/tmp/test_store"
        rag = ThaiRAG(store_path=custom_path)
        assert rag.store_path == Path(custom_path)

    def test_is_ready_without_load(self):
        """is_ready returns False when no index loaded."""
        rag = ThaiRAG()
        assert not rag.is_ready()

    def test_is_ready_with_empty_embeddings(self):
        """is_ready returns False when embeddings array empty."""
        rag = ThaiRAG()
        rag._loaded = True
        rag.embeddings = np.array([])
        rag.metadata = []
        assert not rag.is_ready()


class TestThaiRAGWithMockIndex:
    """Tests for ThaiRAG with a mock index."""

    @pytest.fixture
    def mock_index(self, tmp_path):
        """Create a mock vector store."""
        store_path = tmp_path / "vector_store"
        store_path.mkdir()

        # Create embeddings (5 docs, 768 dims)
        embeddings = np.random.randn(5, 768).astype(np.float32)
        np.save(store_path / "embeddings.npy", embeddings)

        # Create metadata
        metadata = [
            {
                "text": "Document 1 content",
                "source": "doc1.pdf",
                "page": 1,
                "article": "101",
                "token_count": 100
            },
            {
                "text": "Document 2 content",
                "source": "doc2.pdf",
                "page": 2,
                "article": None,
                "token_count": 150
            },
            {
                "text": "มาตรา 103 บุคคลมีสิทธิ",
                "source": "constitution.pdf",
                "page": 5,
                "article": "103",
                "token_count": 200
            },
            {
                "text": "Short doc",
                "source": "doc3.pdf",
                "page": 1,
                "token_count": 10
            },
            {
                "text": "Another document",
                "source": "doc4.pdf",
                "page": 3,
                "token_count": 50
            }
        ]

        with open(store_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        return store_path

    def test_loads_index_on_init(self, mock_index):
        """ThaiRAG loads index automatically on init."""
        rag = ThaiRAG(store_path=str(mock_index))
        assert rag._loaded
        assert rag.embeddings is not None
        assert len(rag.metadata) == 5

    def test_is_ready_with_index(self, mock_index):
        """is_ready returns True with valid index."""
        rag = ThaiRAG(store_path=str(mock_index))
        assert rag.is_ready()

    def test_get_stats_with_index(self, mock_index):
        """get_stats returns correct information."""
        rag = ThaiRAG(store_path=str(mock_index))
        stats = rag.get_stats()

        assert stats["status"] == "loaded"
        assert stats["num_chunks"] == 5
        assert stats["embedding_dim"] == 768
        assert "sources" in stats

    def test_search_returns_results(self, mock_index):
        """search returns top-k results."""
        rag = ThaiRAG(store_path=str(mock_index))
        query = np.random.randn(768)
        results = rag.search(query, k=3)

        assert len(results) == 3
        assert all("text" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("score" in r for r in results)

    def test_search_respects_top_k(self, mock_index):
        """search respects k parameter."""
        rag = ThaiRAG(store_path=str(mock_index))
        query = np.random.randn(768)

        for k in [1, 2, 3, 5]:
            results = rag.search(query, k=k)
            assert len(results) == k

    def test_search_returns_sorted_results(self, mock_index):
        """search returns results sorted by score descending."""
        rag = ThaiRAG(store_path=str(mock_index))
        query = np.random.randn(768)
        results = rag.search(query, k=5)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_article_metadata(self, mock_index):
        """search includes article metadata when available."""
        rag = ThaiRAG(store_path=str(mock_index))
        query = np.random.randn(768)
        results = rag.search(query, k=5)

        # At least one result should have article metadata
        articles = [r["metadata"].get("article") for r in results]
        assert any(a is not None for a in articles)

    def test_search_score_range(self, mock_index):
        """search scores are between -1 and 1."""
        rag = ThaiRAG(store_path=str(mock_index))
        query = np.random.randn(768)
        results = rag.search(query, k=5)

        for r in results:
            assert -1 <= r["score"] <= 1


class TestThaiRAGSearchWithoutIndex:
    """Tests for ThaiRAG search when index not available."""

    def test_search_without_index_returns_empty(self):
        """search returns empty list when no index."""
        rag = ThaiRAG(store_path="/nonexistent/path")
        query = np.random.randn(768)
        results = rag.search(query, k=5)
        assert results == []

    def test_get_stats_without_index(self):
        """get_stats shows not_loaded when no index."""
        rag = ThaiRAG(store_path="/nonexistent/path")
        stats = rag.get_stats()
        assert stats["status"] == "not_loaded"


class TestEmbedderInterface:
    """Tests for EmbedderInterface."""

    def test_from_config_returns_local_if_available(self):
        """from_config returns LocalEmbedder if sentence-transformers available."""
        try:
            from sentence_transformers import SentenceTransformer
            embedder = EmbedderInterface.from_config()
            assert embedder is not None
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_from_config_returns_fallback_if_not_available(self):
        """from_config returns FallbackEmbedder if sentence-transformers missing."""
        # This test assumes sentence-transformers might not be available
        embedder = EmbedderInterface.from_config()
        assert embedder is not None

    def test_fallback_embedder_raises_error(self):
        """FallbackEmbedder raises RuntimeError when embed called."""
        from rag.minimal_rag import FallbackEmbedder
        embedder = FallbackEmbedder()
        with pytest.raises(RuntimeError):
            embedder.embed("test text")


class TestThaiRAGEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_with_k_larger_than_docs(self, tmp_path):
        """search handles k > num_docs gracefully."""
        store_path = tmp_path / "vector_store"
        store_path.mkdir()

        # Create small index (2 docs)
        embeddings = np.random.randn(2, 768).astype(np.float32)
        np.save(store_path / "embeddings.npy", embeddings)

        metadata = [
            {"text": "doc1", "source": "test.pdf", "page": 1},
            {"text": "doc2", "source": "test.pdf", "page": 2}
        ]
        with open(store_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        rag = ThaiRAG(store_path=str(store_path))
        query = np.random.randn(768)
        results = rag.search(query, k=10)  # Request more than available

        assert len(results) == 2  # Returns all available

    def test_search_with_very_small_embeddings(self, tmp_path):
        """search handles edge case with single doc."""
        store_path = tmp_path / "vector_store"
        store_path.mkdir()

        embeddings = np.random.randn(1, 768).astype(np.float32)
        np.save(store_path / "embeddings.npy", embeddings)

        metadata = [{"text": "only doc", "source": "test.pdf", "page": 1}]
        with open(store_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        rag = ThaiRAG(store_path=str(store_path))
        query = np.random.randn(768)
        results = rag.search(query, k=5)

        assert len(results) == 1
        assert results[0]["text"] == "only doc"

    def test_search_with_zero_query(self, tmp_path):
        """search handles zero vector query."""
        store_path = tmp_path / "vector_store"
        store_path.mkdir()

        embeddings = np.random.randn(3, 768).astype(np.float32)
        np.save(store_path / "embeddings.npy", embeddings)

        metadata = [
            {"text": "doc1", "source": "test.pdf", "page": 1},
            {"text": "doc2", "source": "test.pdf", "page": 2},
            {"text": "doc3", "source": "test.pdf", "page": 3}
        ]
        with open(store_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        rag = ThaiRAG(store_path=str(store_path))
        query = np.zeros(768)  # Zero vector
        results = rag.search(query, k=3)

        # Should return results without crashing
        assert len(results) == 3

    def test_missing_embeddings_file(self, tmp_path):
        """ThaiRAG handles missing embeddings.npy."""
        store_path = tmp_path / "vector_store"
        store_path.mkdir()

        # Only create metadata, not embeddings
        metadata = [{"text": "doc", "source": "test.pdf", "page": 1}]
        with open(store_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        rag = ThaiRAG(store_path=str(store_path))
        assert not rag.is_ready()

    def test_missing_metadata_file(self, tmp_path):
        """ThaiRAG handles missing metadata.json."""
        store_path = tmp_path / "vector_store"
        store_path.mkdir()

        # Only create embeddings, not metadata
        embeddings = np.random.randn(1, 768).astype(np.float32)
        np.save(store_path / "embeddings.npy", embeddings)

        rag = ThaiRAG(store_path=str(store_path))
        assert not rag.is_ready()

    def test_corrupted_embeddings_file(self, tmp_path):
        """ThaiRAG handles corrupted embeddings.npy."""
        store_path = tmp_path / "vector_store"
        store_path.mkdir()

        # Create corrupted file
        with open(store_path / "embeddings.npy", "w") as f:
            f.write("not a numpy array")

        metadata = [{"text": "doc", "source": "test.pdf", "page": 1}]
        with open(store_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        rag = ThaiRAG(store_path=str(store_path))
        assert not rag.is_ready()


class TestThaiRAGPerformance:
    """Basic performance tests."""

    def test_search_latency_under_threshold(self, tmp_path):
        """search completes in under 100ms for 1000 chunks."""
        import time

        store_path = tmp_path / "vector_store"
        store_path.mkdir()

        # Create 1000 chunks
        embeddings = np.random.randn(1000, 768).astype(np.float32)
        np.save(store_path / "embeddings.npy", embeddings)

        metadata = [
            {"text": f"doc {i}", "source": "test.pdf", "page": i % 10}
            for i in range(1000)
        ]
        with open(store_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        rag = ThaiRAG(store_path=str(store_path))
        query = np.random.randn(768)

        start = time.time()
        results = rag.search(query, k=5)
        elapsed = time.time() - start

        assert elapsed < 0.1  # 100ms threshold
        assert len(results) == 5
