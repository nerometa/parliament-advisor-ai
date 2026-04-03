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
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge")

# --- Derived ---
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BLOCK_SIZE = 1024
AUDIO_CHUNK_BYTES = 6400  # ~200ms chunks at 16kHz mono 16-bit

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


def load_knowledge() -> str:
    """Extract text from all PDFs in KNOWLEDGE_DIR. Returns empty string if dir missing."""
    import logging
    from pypdf import PdfReader

    logger = logging.getLogger("parliament")
    knowledge_path = Path(KNOWLEDGE_DIR)
    if not knowledge_path.is_dir():
        logger.warning("Knowledge directory not found: %s", knowledge_path.resolve())
        return ""

    sections = []
    for pdf_file in sorted(knowledge_path.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_file))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                sections.append(f"=== {pdf_file.name} ===\n{text.strip()}")
                logger.info("Loaded knowledge: %s (%d pages)", pdf_file.name, len(reader.pages))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", pdf_file.name, exc)

    if not sections:
        logger.warning("No PDF files found in %s", knowledge_path.resolve())
        return ""

    result = "\n\n".join(sections)
    logger.info("Knowledge base loaded: %d files, %d chars", len(sections), len(result))
    return result
