"""Logging setup — structured, concise, one-stop import."""

import logging
import sys
from src.config import cfg

def get_logger(name: str = "aria") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))
    return logger

log = get_logger()
