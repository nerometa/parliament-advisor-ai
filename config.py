# config.py — Centralized settings loaded from .env and system_prompt.txt
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Required ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise SystemExit(
        "ERROR: GEMINI_API_KEY is not set.\n"
        "Copy .env.example to .env and add your key from https://ai.google.dev"
    )

# --- Optional ---
GOOGLE_CHAT_WEBHOOK_URL = os.getenv("GOOGLE_CHAT_WEBHOOK_URL", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")

# --- Derived ---
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BLOCK_SIZE = 1024
AUDIO_CHUNK_BYTES = 2048  # for livestream ffmpeg reads

WEBHOOK_MAX_RETRIES = 3
WEBHOOK_RETRY_DELAY = 1.0  # seconds, doubles on each retry


def load_system_prompt() -> str:
    """Load system prompt from file. Exit with clear error if missing."""
    path = Path(SYSTEM_PROMPT_PATH)
    if not path.exists():
        raise SystemExit(
            f"ERROR: System prompt file not found: {path.resolve()}\n"
            "Create system_prompt.txt with your Smart MP Advisor instructions."
        )
    return path.read_text(encoding="utf-8").strip()


def has_webhook() -> bool:
    """Check if Google Chat webhook is configured."""
    return bool(GOOGLE_CHAT_WEBHOOK_URL and "YOUR_SPACE" not in GOOGLE_CHAT_WEBHOOK_URL)
