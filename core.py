"""core.py — Gemini Live session management and audio capture."""

import asyncio
import logging
from typing import AsyncGenerator

from google import genai
from google.genai import types

import config

logger = logging.getLogger("parliament")


class GeminiSession:
    """Manages a real-time Gemini Live API session for audio-to-text analysis."""

    def __init__(self):
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._session = None
        self._ctx = None

    async def connect(self):
        """Open a live connection to the Gemini model."""
        logger.info("Connecting to Gemini Live (%s)...", config.GEMINI_MODEL)
        self._ctx = self._client.aio.live.connect(
            model=config.GEMINI_MODEL,
            config=types.LiveConnectConfig(
                response_modalities=["TEXT"],
                system_instruction=types.Content(
                    parts=[types.Part(text=config.load_system_prompt())]
                ),
            ),
        )
        self._session = await self._ctx.__aenter__()
        logger.info("Gemini Live connected.")

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
        if self._ctx:
            logger.info("Closing Gemini Live session.")
            await self._ctx.__aexit__(None, None, None)
            self._session = None
            self._ctx = None
            self._ctx = None


async def capture_mic() -> AsyncGenerator[bytes, None]:
    """Capture audio from the microphone as PCM int16 chunks.

    Uses sounddevice with an asyncio.Queue bridge. Yields raw PCM bytes
    at 16kHz mono 16-bit (matching Gemini Live input requirements).
    """
    import sounddevice as sd  # lazy — requires libportaudio2 system library

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    stream = None

    def _callback(indata, frames, time_info, status):
        if status:
            logger.warning("Mic callback status: %s", status)
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

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

    try:
        while True:
            chunk = await process.stdout.read(config.AUDIO_CHUNK_BYTES)
            if not chunk:
                logger.info("Livestream ended (ffmpeg stdout closed).")
                break
            yield chunk
    except asyncio.CancelledError:
        logger.info("Livestream capture cancelled.")
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
        logger.info("ffmpeg process cleaned up.")
