"""core.py — Gemini Live session management and audio capture."""

import asyncio
import logging
from typing import AsyncGenerator, Optional

from google import genai
from google.genai import types

import config
from rag.minimal_rag import ThaiRAG, EmbedderInterface

logger = logging.getLogger("parliament")


class GeminiSession:
    """Manages a real-time Gemini Live API session for audio-to-text analysis."""

    def __init__(self):
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._session = None
        self._ctx = None
        self._closed = False
        self._rag = ThaiRAG()
        self._embedder = None

    async def _ensure_embedder(self):
        """Lazy-load embedder for RAG queries."""
        if self._embedder is None:
            self._embedder = EmbedderInterface.from_config()
        return self._embedder

    async def retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context from RAG for a query."""
        if not self._rag.is_ready():
            logger.warning("RAG not ready - run index_knowledge.py first")
            return ""

        try:
            embedder = await self._ensure_embedder()
            query_embedding = embedder.embed(query)
            results = self._rag.search(query_embedding, k=k)

            if not results:
                return ""

            context_parts = []
            for r in results:
                source = r["metadata"].get("source", "Unknown")
                article = r["metadata"].get("article")
                text = r["text"]

                if article:
                    context_parts.append(f"[From {source}, Article {article}]: {text}")
                else:
                    context_parts.append(f"[From {source}]: {text}")

            return "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return ""

    async def connect(self):
        """Open a live connection to the Gemini model."""
        logger.info("Connecting to Gemini Live (%s)...", config.GEMINI_MODEL)

        system_text = config.load_system_prompt()
        logger.info("System instruction: %d chars (prompt only, no knowledge)",
                     len(system_text))

        try:
            self._ctx = self._client.aio.live.connect(
                model=config.GEMINI_MODEL,
                config=types.LiveConnectConfig(
                    response_modalities=["TEXT"],
                    system_instruction=system_text,
                ),
            )
            self._session = await self._ctx.__aenter__()
            logger.info("Gemini Live connected.")
        except Exception as exc:
            error_msg = str(exc)
            if "1011" in error_msg or "internal error" in error_msg.lower():
                raise ConnectionError(
                    "Gemini Live API returned internal error (1011). "
                    "This often indicates API quota exhaustion or rate limiting. "
                    "Please check your usage at https://ai.dev/rate-limit "
                    f"Original error: {exc}"
                ) from exc
            raise

    async def send_audio(self, chunk: bytes):
        """Send a PCM audio chunk to the model."""
        if not self._session:
            raise RuntimeError("Not connected — call connect() first.")
        await self._session.send_realtime_input(
            audio=types.Blob(
                data=chunk,
                mime_type=f"audio/pcm;rate={config.AUDIO_SAMPLE_RATE}",
            )
        )

    async def receive_text(self) -> AsyncGenerator[str, None]:
        """Yield text responses from the model until the turn is complete."""
        if not self._session:
            raise RuntimeError("Not connected — call connect() first.")
        collected = []
        async for msg in self._session.receive():
            if msg.server_content and msg.server_content.model_turn:
                for part in msg.server_content.model_turn.parts:
                    if part.text:
                        collected.append(part.text)
            if msg.server_content and msg.server_content.turn_complete:
                break
        full_text = "".join(collected).strip()
        if full_text:
            yield full_text

    async def close(self):
        """Close the Gemini Live session."""
        if self._closed:
            return
        self._closed = True
        if self._ctx:
            logger.info("Closing Gemini Live session.")
            await self._ctx.__aexit__(None, None, None)
            self._session = None
            self._ctx = None


async def capture_mic() -> AsyncGenerator[bytes, None]:
    """Capture audio from the microphone as PCM int16 chunks.

    Uses sounddevice with an asyncio.Queue bridge. Yields raw PCM bytes
    at 16kHz mono 16-bit (matching Gemini Live input requirements).
    """
    import sounddevice as sd  # lazy — requires libportaudio2 system library

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)  # Bounded for backpressure
    stream = None

    def _callback(indata, frames, time_info, status):
        if status:
            logger.warning("Mic callback status: %s", status)
        # Drop oldest chunk if queue full (backpressure)
        try:
            loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))
        except asyncio.QueueFull:
            try:
                loop.call_soon_threadsafe(queue.get_nowait)
                loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))
                logger.debug("Dropped oldest audio chunk due to backlog")
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    try:
        stream = sd.InputStream(
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="int16",
            blocksize=config.AUDIO_BLOCK_SIZE,
            callback=_callback,
        )
        stream.start()
        logger.info("Microphone capture started (16kHz mono int16).")

        while True:
            chunk = await queue.get()
            yield chunk
    except asyncio.CancelledError:
        logger.info("Mic capture cancelled.")
        raise
    finally:
        if stream:
            stream.stop()
            stream.close()
            logger.info("Microphone capture stopped.")


async def capture_livestream(url: str) -> AsyncGenerator[bytes, None]:
    """Capture audio from a livestream URL via ffmpeg.

    Runs ffmpeg as a subprocess, extracting audio as PCM 16kHz mono
    and yielding raw bytes from stdout.
    """
    cmd = [
        "ffmpeg",
        "-re",
        "-i", url,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-ac", str(config.AUDIO_CHANNELS),
        "-f", "s16le",
        "pipe:1",
    ]
    logger.info("Starting ffmpeg for livestream: %s", url)
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Background task to drain stderr and prevent 64KB buffer deadlock
    async def drain_stderr():
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            logger.debug("ffmpeg: %s", line.decode().strip())

    stderr_task = asyncio.create_task(drain_stderr())

    try:
        while True:
            chunk = await process.stdout.read(config.AUDIO_CHUNK_BYTES)
            if not chunk:
                logger.info("Livestream ended (ffmpeg stdout closed).")
                break
            yield chunk
    except asyncio.CancelledError:
        logger.info("Livestream capture cancelled.")
        raise
    finally:
        stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("FFmpeg didn't terminate gracefully, killing")
                process.kill()
                await process.wait()
            # Log any remaining stderr for debugging
            stderr = await process.stderr.read()
            if stderr:
                logger.debug("FFmpeg stderr: %s", stderr.decode()[:500])
        logger.info("ffmpeg process cleaned up.")
