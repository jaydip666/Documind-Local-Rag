"""Central logging setup used by every module instead of print()."""

import logging
import sys

from config import SETTINGS

_CONFIGURED = False


def _configure_root() -> None:
    """Configure the root logger once with a consistent format and level.

    Args:
        None.

    Returns:
        None.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, SETTINGS.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Usually the caller's __name__.

    Returns:
        A logging.Logger instance ready to use.
    """
    _configure_root()
    return logging.getLogger(name)
