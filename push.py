"""push.py — Send alerts to Google Chat via incoming webhook with retry logic."""

import asyncio
import logging

import aiohttp

import config

logger = logging.getLogger(__name__)

# Track last send time for rate limiting (1 msg/sec)
_last_send_time: float = 0.0
_send_lock = asyncio.Lock()


async def send_alert(message: str) -> bool:
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

    async with _send_lock:
        # Rate limit: respect Google Chat's 1 msg/sec
        now = asyncio.get_running_loop().time()
        elapsed = now - _last_send_time
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)

        delay = config.WEBHOOK_RETRY_DELAY
        async with aiohttp.ClientSession() as session:
            for attempt in range(1, config.WEBHOOK_MAX_RETRIES + 1):
                try:
                    logger.info("Sending alert (attempt %d/%d)", attempt, config.WEBHOOK_MAX_RETRIES)
                    async with session.post(
                        config.GOOGLE_CHAT_WEBHOOK_URL,
                        json={"text": message},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            logger.info("Alert sent successfully.")
                            _last_send_time = asyncio.get_running_loop().time()
                            return True

                        if resp.status == 429:
                            logger.warning("Rate limited (429) — retrying in %.1fs", delay)
                            await asyncio.sleep(delay)
                            delay *= 2
                            continue

                        if resp.status >= 500:
                            logger.warning(
                                "Server error %d — retrying in %.1fs", resp.status, delay
                            )
                            await asyncio.sleep(delay)
                            delay *= 2
                            continue

                        # Other 4xx — don't retry
                        text = await resp.text()
                        logger.error(
                            "Client error %d — not retrying. Response: %s",
                            resp.status,
                            text[:200],
                        )
                        return False

                except aiohttp.ClientError as exc:
                    logger.warning("Request failed: %s — retrying in %.1fs", exc, delay)
                    await asyncio.sleep(delay)
                    delay *= 2

    logger.error("Failed to send alert after %d attempts.", config.WEBHOOK_MAX_RETRIES)
    return False
