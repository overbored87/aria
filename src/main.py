"""Aria Bot — Main entry point."""

import signal
import sys

from telegram.ext import Application

from src.config import cfg
from src.utils.logger import log
from src.handlers.telegram_handlers import register_handlers
from src.services.scheduler import init_scheduler


def main() -> None:
    log.info("=" * 50)
    log.info("  Aria Bot starting up...")
    log.info("=" * 50)

    try:
        cfg.validate()
    except RuntimeError as e:
        log.error(f"Config error: {e}")
        sys.exit(1)

    log.info(f"User timezone: {cfg.user_timezone}")
    log.info(f"Allowed user ID: {cfg.allowed_user_id}")
    log.info(f"Claude model: {cfg.claude_model}")

    app = (
        Application.builder()
        .token(cfg.telegram_token)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(30)
        .build()
    )

    register_handlers(app)
    scheduler = init_scheduler(app)

    def shutdown_handler(signum, frame):
        log.info("Shutdown signal received...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    log.info("Bot is live! Polling for messages...")
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=["message"],
    )


if __name__ == "__main__":
    main()
