"""push.py — Send alerts to Google Chat via incoming webhook with retry logic."""

import logging
import time

import requests

import config

logger = logging.getLogger(__name__)

# Track last send time for rate limiting (1 msg/sec)
_last_send_time: float = 0.0


def send_alert(message: str) -> bool:
    """Send an alert message to Google Chat.

    Args:
        message: The text to send.

    Returns:
        True if sent successfully, False otherwise.
    """
    if not config.has_webhook():
        logger.warning("Webhook not configured — skipping alert.")
        return False

    global _last_send_time

    # Rate limit: respect Google Chat's 1 msg/sec
    now = time.monotonic()
    elapsed = now - _last_send_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    delay = config.WEBHOOK_RETRY_DELAY
    for attempt in range(1, config.WEBHOOK_MAX_RETRIES + 1):
        try:
            logger.info("Sending alert (attempt %d/%d)", attempt, config.WEBHOOK_MAX_RETRIES)
            resp = requests.post(
                config.GOOGLE_CHAT_WEBHOOK_URL,
                json={"text": message},
                timeout=10,
            )

            if resp.status_code == 200:
                logger.info("Alert sent successfully.")
                _last_send_time = time.monotonic()
                return True

            if resp.status_code == 429:
                logger.warning("Rate limited (429) — retrying in %.1fs", delay)
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code >= 500:
                logger.warning(
                    "Server error %d — retrying in %.1fs", resp.status_code, delay
                )
                time.sleep(delay)
                delay *= 2
                continue

            # Other 4xx — don't retry
            logger.error(
                "Client error %d — not retrying. Response: %s",
                resp.status_code,
                resp.text[:200],
            )
            return False

        except requests.RequestException as exc:
            logger.warning("Request failed: %s — retrying in %.1fs", exc, delay)
            time.sleep(delay)
            delay *= 2

    logger.error("Failed to send alert after %d attempts.", config.WEBHOOK_MAX_RETRIES)
    return False
