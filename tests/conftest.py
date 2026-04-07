"""conftest.py — Pytest fixtures for Parliament Advisor AI tests."""

import pytest
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def sample_audio_chunk() -> bytes:
    """Return a sample PCM audio chunk (640 bytes of silence)."""
    return b"\x00\x00" * 320  # 640 bytes of 16-bit silence


@pytest.fixture
def sample_chunks() -> list[bytes]:
    """Return a list of sample audio chunks."""
    return [b"\x00\x00" * 320 for _ in range(5)]


@pytest.fixture
def mock_embedder():
    """Create a mock embedder with async embed method."""
    mock = AsyncMock()
    mock.embed = AsyncMock(return_value=[0.1] * 768)
    return mock


@pytest.fixture
def mock_retriever():
    """Create a mock retriever with async retrieve method."""
    mock = AsyncMock()
    mock.retrieve = AsyncMock(
        return_value=[
            {"text": "Sample document 1", "source": "test1.pdf", "score": 0.9},
            {"text": "Sample document 2", "source": "test2.pdf", "score": 0.8},
        ]
    )
    return mock


@pytest.fixture
def mock_gemini_session():
    """Create a mock GeminiSession with connect/send/receive methods."""
    mock = AsyncMock()
    mock.connect = AsyncMock()
    mock.send_audio = AsyncMock()
    mock.receive_text = AsyncMock(
        return_value=AsyncIteratorMock(["Test response text"])
    )
    mock.close = AsyncMock()
    return mock


class AsyncIteratorMock:
    """Mock async iterator for testing async generators."""

    def __init__(self, items: list):
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


@pytest.fixture
def sample_knowledge_chunks() -> list[dict]:
    """Return sample knowledge chunks for testing."""
    return [
        {
            "text": "มาตรา 83 คุณสมบัติของ ส.ส. ต้องมีสัญชาติไทย",
            "source": "constitution.pdf",
            "article": "มาตรา 83",
        },
        {
            "text": "มาตรา 84 ลักษณะต้องห้ามของ ส.ส.",
            "source": "constitution.pdf",
            "article": "มาตรา 84",
        },
    ]


@pytest.fixture
def sample_ground_truth() -> list[dict]:
    """Return sample ground truth queries for testing."""
    return [
        {
            "query": "มาตรา 101 กล่าวถึงเรื่องอะไร",
            "expected_articles": ["มาตรา 101"],
            "expected_keywords": ["องค์ประชุม", "บุคคล", "อายุ"],
        }
    ]
