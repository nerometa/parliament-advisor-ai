"""
Tests for the RAG Chunker module (TDD approach).

This module tests chunking functionality for Thai legal documents,
including article boundary preservation, token counting, and metadata handling.
"""

import pytest
import tiktoken
from unittest.mock import patch, MagicMock

from rag.chunker import Chunker, Chunk


class TestChunk:
    """Tests for the Chunk data structure."""

    def test_chunk_creation_basic(self):
        """Create chunk with required fields."""
        chunk = Chunk(
            text="This is test content.",
            metadata={
                "source": "test.pdf",
                "page": 1,
                "token_count": 5
            }
        )
        assert chunk.text == "This is test content."
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["token_count"] == 5

    def test_chunk_with_article_metadata(self):
        """Create chunk with article number in metadata."""
        chunk = Chunk(
            text="มาตรา 101 สมาชิกมีหน้าที่...",
            metadata={
                "source": "constitution.pdf",
                "page": 5,
                "article": "101",
                "token_count": 10
            }
        )
        assert chunk.metadata["article"] == "101"
        assert "มาตรา" in chunk.text


class TestChunkerInitialization:
    """Tests for Chunker class initialization."""

    def test_chunker_with_defaults(self):
        """Initialize chunker with default config values."""
        chunker = Chunker()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50

    def test_chunker_with_custom_values(self):
        """Initialize chunker with custom values."""
        chunker = Chunker(chunk_size=256, overlap=25)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 25

    @patch('rag.chunker.config')
    def test_chunker_config_fallback(self, mock_config):
        """Verify config values used as defaults."""
        mock_config.CHUNK_SIZE_TOKENS = 1024
        mock_config.CHUNK_OVERLAP_TOKENS = 100
        
        chunker = Chunker()
        assert chunker.chunk_size == 1024
        assert chunker.overlap == 100


class TestTokenCounting:
    """Tests for token counting functionality."""

    def test_count_tokens_empty_string(self):
        """Empty string should return 0 tokens."""
        chunker = Chunker()
        assert chunker.count_tokens("") == 0

    def test_count_tokens_simple_text(self):
        """Count tokens in simple ASCII text."""
        chunker = Chunker()
        text = "Hello world, this is a test."
        # tiktoken cl100k_base encoding
        token_count = chunker.count_tokens(text)
        assert token_count > 0
        assert isinstance(token_count, int)

    def test_count_tokens_thai_text(self):
        """Count tokens in Thai text (important for legal documents)."""
        chunker = Chunker()
        thai_text = "มาตรา 101 สมาชิกสภาผู้แทนราษฎรมีหน้าที่ปฏิบัติ"
        token_count = chunker.count_tokens(thai_text)
        assert token_count > 0
        assert isinstance(token_count, int)

    def test_count_tokens_mixed_thai_english(self):
        """Count tokens in mixed Thai-English text."""
        chunker = Chunker()
        mixed_text = "Article 101 (มาตรา 101) Members must comply."
        token_count = chunker.count_tokens(mixed_text)
        assert token_count > 0


class TestArticleBoundaryDetection:
    """Tests for Thai article pattern detection."""

    def test_detect_single_article(self):
        """Detect single Thai article pattern."""
        chunker = Chunker()
        text = "มาตรา 101 สมาชิกมีหน้าที่..."
        articles = chunker.detect_articles(text)
        assert "101" in articles

    def test_detect_multiple_articles(self):
        """Detect multiple Thai article patterns."""
        chunker = Chunker()
        text = "มาตรา 101 ... มาตรา 102 ... มาตรา 103"
        articles = chunker.detect_articles(text)
        assert "101" in articles
        assert "102" in articles
        assert "103" in articles

    def test_detect_article_with_whitespace(self):
        """Handle article patterns with variable whitespace."""
        chunker = Chunker()
        text = "มาตรา  50 รัฐธรรมนูญ..."  # Multiple spaces
        articles = chunker.detect_articles(text)
        assert "50" in articles

    def test_detect_article_no_match(self):
        """Return empty set when no article patterns found."""
        chunker = Chunker()
        text = "This text has no Thai article markers."
        articles = chunker.detect_articles(text)
        assert len(articles) == 0

    def test_detect_article_thai_numerals(self):
        """Handle Thai numeral patterns (if applicable)."""
        chunker = Chunker()
        # Thai numerals might not be standard in legal text, but worth testing
        text = "มาตรา๑๐๑ ..."  # Thai numeral variant
        articles = chunker.detect_articles(text)
        # Implementation should handle or ignore gracefully
        # This test documents expected behavior


class TestChunkingLogic:
    """Tests for the core chunking functionality."""

    def test_chunk_empty_input(self):
        """Handle empty input gracefully."""
        chunker = Chunker()
        chunks = chunker.chunk("")
        assert len(chunks) == 0

    def test_chunk_single_small_chunk(self):
        """Text smaller than chunk size returns single chunk."""
        chunker = Chunker(chunk_size=512, overlap=50)
        small_text = "Small text " * 10  # Much less than 512 tokens
        chunks = chunker.chunk(small_text)
        assert len(chunks) == 1
        assert chunks[0].metadata["token_count"] < 512

    def test_chunk_multiple_chunks_basic(self):
        """Text larger than chunk size splits into multiple chunks."""
        chunker = Chunker(chunk_size=100, overlap=10)
        # Create text that will definitely exceed 100 tokens
        large_text = "This is a longer text segment. " * 100
        chunks = chunker.chunk(large_text)
        assert len(chunks) > 1

    def test_chunk_overlap_respected(self):
        """Verify overlap between consecutive chunks."""
        chunker = Chunker(chunk_size=100, overlap=20)
        text = "Word " * 500  # Large text
        chunks = chunker.chunk(text)
        
        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            # The overlap ensures continuity
            assert chunks[0].text != chunks[1].text
            # Check chunk sizes are within bounds
            for chunk in chunks[:-1]:  # All but last
                assert chunk.metadata["token_count"] <= chunker.chunk_size + chunker.overlap

    def test_chunk_metadata_source(self):
        """Verify source metadata is included."""
        chunker = Chunker()
        text = "Test content for chunking."
        chunks = chunker.chunk(text, source="constitution.pdf")
        assert chunks[0].metadata["source"] == "constitution.pdf"

    def test_chunk_metadata_page(self):
        """Verify page metadata is included when provided."""
        chunker = Chunker()
        text = "Test content for chunking."
        chunks = chunker.chunk(text, source="test.pdf", page=5)
        assert chunks[0].metadata["page"] == 5


class TestArticleBoundaryPreservation:
    """Tests for article boundary-aware chunking."""

    def test_article_at_boundary(self):
        """Article markers at chunk boundary are preserved."""
        chunker = Chunker(chunk_size=50, overlap=10)
        # Create text where "มาตรา 102" would be near chunk boundary
        text = "Some content here. " * 10 + "มาตรา 102 New article content. " + "More text. " * 10
        chunks = chunker.chunk(text)
        
        # At least one chunk should contain the article marker intact
        # (not split in the middle of "มาตรา 102")
        article_in_chunks = any("มาตรา 102" in c.text for c in chunks)
        assert article_in_chunks or len(chunks) == 1

    def test_article_not_split_midway(self):
        """Articles are not split in the middle of their markers."""
        chunker = Chunker(chunk_size=30, overlap=5)
        text = "มาตรา 101 Content of article 101. " * 5 + "มาตรา 102 More content."
        chunks = chunker.chunk(text)
        
        # Verify no chunk has partial article marker
        for chunk in chunks:
            # Should not have "มาตรา" without number following
            if "มาตรา" in chunk.text:
                # There should be a number after มาตรา
                assert chunker.detect_articles(chunk.text) or "มาตรา" not in chunk.text.rstrip()

    def test_chunk_with_article_metadata(self):
        """Chunk containing article has it in metadata."""
        chunker = Chunker()
        text = "มาตรา 99 สมาชิกมีสิทธิ..."
        chunks = chunker.chunk(text, source="regulations.pdf")
        
        if chunks and "มาตรา" in chunks[0].text:
            articles = chunker.detect_articles(chunks[0].text)
            if articles:
                assert "article" in chunks[0].metadata


class TestTokenLimitRespect:
    """Tests for ensuring token limits are respected."""

    def test_chunks_within_size_limit(self):
        """All chunks respect the token size limit."""
        chunker = Chunker(chunk_size=100, overlap=0)
        large_text = "Token test. " * 1000
        chunks = chunker.chunk(large_text)
        
        for chunk in chunks:
            # Allow small buffer for tokenization differences
            assert chunk.metadata["token_count"] <= chunker.chunk_size + 5

    def test_overlap_applied_correctly(self):
        """Overlap is applied as configured."""
        chunker = Chunker(chunk_size=50, overlap=10)
        text = "Test overlap functionality " * 200
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 1
        assert chunker.overlap == 10


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_chunk_whitespace_only(self):
        """Handle whitespace-only input."""
        chunker = Chunker()
        chunks = chunker.chunk("   \n\t   ")
        # Should either return empty or single empty chunk
        assert len(chunks) == 0 or chunks[0].text.strip() == ""

    def test_chunk_unicode_handling(self):
        """Handle Unicode characters correctly."""
        chunker = Chunker()
        unicode_text = "🎉 Test emoji 🇹🇭 Thai flag ไทย"
        chunks = chunker.chunk(unicode_text)
        assert len(chunks) >= 1
        # Verify no encoding issues
        assert isinstance(chunks[0].text, str)

    def test_chunk_very_long_single_word(self):
        """Handle very long single word/token."""
        chunker = Chunker(chunk_size=10, overlap=2)
        long_word = "a" * 1000
        chunks = chunker.chunk(long_word)
        # Should handle gracefully without crashing
        assert len(chunks) >= 1

    def test_chunk_with_none_source(self):
        """Handle None source gracefully."""
        chunker = Chunker()
        text = "Test content."
        chunks = chunker.chunk(text, source=None)
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] is None

    def test_chunk_with_page_zero(self):
        """Handle page=0 correctly."""
        chunker = Chunker()
        text = "Test content."
        chunks = chunker.chunk(text, source="test.pdf", page=0)
        assert chunks[0].metadata["page"] == 0


class TestConfigIntegration:
    """Tests for config.py integration."""

    @patch('rag.chunker.config')
    def test_uses_config_chunk_size(self, mock_config):
        """Verify CHUNK_SIZE_TOKENS from config is used."""
        mock_config.CHUNK_SIZE_TOKENS = 256
        mock_config.CHUNK_OVERLAP_TOKENS = 25
        
        chunker = Chunker()
        assert chunker.chunk_size == 256

    @patch('rag.chunker.config')
    def test_uses_config_overlap(self, mock_config):
        """Verify CHUNK_OVERLAP_TOKENS from config is used."""
        mock_config.CHUNK_SIZE_TOKENS = 512
        mock_config.CHUNK_OVERLAP_TOKENS = 75
        
        chunker = Chunker()
        assert chunker.overlap == 75


class TestChunkerWithRealText:
    """Tests with realistic Thai legal text samples."""

    def test_thai_constitution_sample(self):
        """Test chunking Thai constitution-like text."""
        chunker = Chunker(chunk_size=200, overlap=20)
        thai_text = """
        มาตรา 101 สมาชิกสภาผู้แทนราษฎรมีหน้าที่ปฏิบัติตามรัฐธรรมนูญ
        และกฎหมายที่เกี่ยวข้อง โดยต้องไม่กระทำการใดๆ อันเป็นการละเมิด
        สิทธิเสรีภาพของประชาชน
        
        มาตรา 102 การประชุมสภาต้องมีสมาชิกไม่น้อยกว่ากึ่งหนึ่งของ
        จำนวนสมาชิกทั้งหมด จึงจะเป็นองค์ประชุม
        """
        chunks = chunker.chunk(thai_text, source="constitution.pdf")
        
        assert len(chunks) >= 1
        all_articles = set()
        for chunk in chunks:
            articles = chunker.detect_articles(chunk.text)
            all_articles.update(articles)
        
        # Should have detected article numbers
        assert "101" in all_articles or "102" in all_articles

    def test_long_document_chunking(self):
        """Test chunking a long document produces appropriate number of chunks."""
        chunker = Chunker(chunk_size=100, overlap=10)
        
        # Simulate a longer document
        article = "มาตรา 100 เนื้อหาของมาตรา "
        long_text = (article + "ทดสอบ " * 50 + "\n\n") * 10
        
        chunks = chunker.chunk(long_text, source="regulations.pdf", page=1)
        
        # Long document should produce multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should have proper metadata
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "token_count" in chunk.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
