"""Centralised logging setup for HERALD."""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "herald"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``herald`` namespace.

    Args:
        name: Optional sub-module suffix (e.g. ``"detector"``).

    Returns:
        A :class:`logging.Logger` whose name is either ``"herald"`` or
        ``"herald.<name>"``.
    """
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root ``herald`` logger.

    Call this once at application startup (typically from :func:`herald.cli.main`).

    Args:
        level: Python logging level string, e.g. ``"DEBUG"`` or ``"WARNING"``.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        # Already configured – update level only.
        logger.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
