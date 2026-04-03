#!/usr/bin/env python3
"""main.py — Parliament Advisor AI: real-time parliamentary audio monitor.

Captures audio (microphone or livestream), streams to Gemini Live for
analysis, and pushes AT_RISK alerts to Google Chat webhook.
"""

import argparse
import asyncio
import logging
import signal
import sys
from contextlib import aclosing

import config
import core
import push

logger = logging.getLogger("parliament")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parliament Advisor AI — real-time parliamentary audio monitor",
    )
    parser.add_argument(
        "--mode",
        choices=["mic", "livestream"],
        required=True,
        help="Audio source: microphone or livestream URL",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Livestream URL (required when --mode livestream)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print alerts to stdout instead of sending webhook",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()
    if args.mode == "livestream" and not args.url:
        parser.error("--url is required when --mode is livestream")
    return args


async def handle_alert(text: str, dry_run: bool):
    text = text.strip()
    if not text:
        return

    is_alert = "AT_RISK" in text.upper()

    if is_alert:
        logger.warning("AT_RISK detected!")
        print(f"\n{'='*60}")
        print(text)
        print(f"{'='*60}\n")

        if dry_run:
            logger.info("[DRY RUN] Alert would be sent to Google Chat.")
        else:
            success = await push.send_alert(text)
            if success:
                logger.info("Alert pushed to Google Chat.")
            else:
                logger.error("Failed to push alert to Google Chat.")
    else:
        logger.debug("Model response (non-alert): %s", text[:200])


async def run_pipeline(mode: str, url: str | None, dry_run: bool, shutdown_event: asyncio.Event):
    """Capture audio → stream to Gemini → handle responses. Reconnects on failure."""
    session = core.GeminiSession()
    reconnect_attempts = 0
    max_reconnects = 5

    try:
        while not shutdown_event.is_set():
            try:
                await session.connect()
                reconnect_attempts = 0
                logger.info("Session established. Streaming audio...")

                if mode == "mic":
                    audio_gen = core.capture_mic()
                else:
                    audio_gen = core.capture_livestream(url)

                # aclosing ensures generator cleanup on cancellation/error
                async with aclosing(audio_gen):
                    send_task = asyncio.create_task(
                        _send_audio_loop(session, audio_gen, shutdown_event),
                        name="send-audio",
                    )
                    recv_task = asyncio.create_task(
                        _receive_text_loop(session, dry_run, shutdown_event),
                        name="receive-text",
                    )

                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task.exception():
                            raise task.exception()

                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

            except asyncio.CancelledError:
                break
            except Exception as exc:
                reconnect_attempts += 1
                if reconnect_attempts >= max_reconnects:
                    logger.error("Max reconnection attempts (%d) reached.", max_reconnects)
                    raise
                wait = min(2 ** reconnect_attempts, 30)
                logger.error(
                    "Session error: %s — reconnecting in %ds (attempt %d/%d)",
                    exc, wait, reconnect_attempts, max_reconnects,
                )
                await asyncio.sleep(wait)
            finally:
                await session.close()
    finally:
        await session.close()
        logger.info("Pipeline shut down.")


async def _send_audio_loop(session, audio_gen, shutdown_event: asyncio.Event):
    async for chunk in audio_gen:
        if shutdown_event.is_set():
            break
        try:
            await session.send_audio(chunk)
        except Exception as exc:
            logger.error("Failed to send audio chunk: %s", exc)
            raise


async def _receive_text_loop(session, dry_run: bool, shutdown_event: asyncio.Event):
    while not shutdown_event.is_set():
        try:
            async for text in session.receive_text():
                if shutdown_event.is_set():
                    break
                await handle_alert(text, dry_run)
        except Exception as exc:
            logger.error("Receive error: %s", exc)
            raise


async def main():
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger.info("Parliament Advisor AI starting — mode=%s", args.mode)
    if args.dry_run:
        logger.info("DRY RUN mode — alerts will print to stdout only.")
    if config.has_webhook():
        logger.info("Google Chat webhook configured.")
    else:
        logger.warning("Google Chat webhook NOT configured — alerts will not be sent.")

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    try:
        await run_pipeline(args.mode, args.url, args.dry_run, shutdown_event)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())
