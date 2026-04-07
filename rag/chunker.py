"""Chunker module for Thai legal document processing.

Splits text into token-aware chunks while preserving article boundaries.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

import config


@dataclass
class Chunk:
    """A text chunk with associated metadata."""

    text: str
    metadata: dict = field(default_factory=dict)


class Chunker:
    """Token-aware text chunker with Thai article boundary preservation.

    Splits text into chunks that respect:
    - Token count limits (using tiktoken)
    - Overlap between consecutive chunks
    - Thai legal article boundaries (มาตรา patterns)
    """

    ARTICLE_PATTERN = re.compile(r'มาตรา\s*(\d+)')

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE_TOKENS
        self.overlap = overlap if overlap is not None else config.CHUNK_OVERLAP_TOKENS
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def detect_articles(self, text: str) -> set:
        matches = self.ARTICLE_PATTERN.findall(text)
        return set(matches)

    def _find_article_boundaries(self, text: str) -> list:
        boundaries = []
        for match in self.ARTICLE_PATTERN.finditer(text):
            boundaries.append(match.start())
        return boundaries

    def _find_nearest_boundary(self, text: str, target_pos: int, boundaries: list) -> int:
        if not boundaries:
            return target_pos

        valid_boundaries = [b for b in boundaries if b < target_pos]
        if valid_boundaries:
            nearest = max(valid_boundaries)
            distance = target_pos - nearest
            if distance < self.chunk_size // 4:
                return nearest

        return target_pos

    def _split_text_by_tokens(self, text: str) -> list:
        if not text.strip():
            return []

        tokens = self._encoder.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        boundaries = self._find_article_boundaries(text)
        text_chars = len(text)

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))

            if end < len(tokens):
                token_slice = tokens[start:end]
                decoded = self._encoder.decode(token_slice)
                char_pos = len(decoded)

                current_char = 0
                decode_pos = 0
                for i, tk in enumerate(tokens):
                    if i >= start and i < end:
                        current_char = decode_pos + len(self._encoder.decode([tk]))
                    decode_pos += len(self._encoder.decode([tk]))

                approx_char_pos = end * 4

                nearest_boundary = self._find_nearest_boundary(text, approx_char_pos, boundaries)

                if nearest_boundary != approx_char_pos and nearest_boundary < text_chars:
                    boundary_text = text[:nearest_boundary]
                    end = start + self.count_tokens(boundary_text) - start

            chunk_tokens = tokens[start:end]
            chunk_text = self._encoder.decode(chunk_tokens)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break

            overlap_tokens = min(self.overlap, self.chunk_size // 2)
            start = end - overlap_tokens

        return chunks

    def chunk(
        self,
        text: str,
        source: Optional[str] = None,
        page: Optional[int] = None
    ) -> list:
        if not text or not text.strip():
            return []

        text_chunks = self._split_text_by_tokens(text)
        result = []

        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue

            articles = self.detect_articles(chunk_text)
            article = list(articles)[0] if articles else None

            metadata = {
                "source": source,
                "page": page,
                "token_count": self.count_tokens(chunk_text),
            }
            if article:
                metadata["article"] = article

            result.append(Chunk(text=chunk_text, metadata=metadata))

        return result