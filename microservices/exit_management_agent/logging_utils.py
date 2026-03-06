"""logging_utils: configure structured stdout logging for the agent."""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure root logger with a timestamped formatter on stdout.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Logger named "exit_management_agent".
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)
    if not root.handlers:
        root.addHandler(handler)
    else:
        # Replace existing handlers to avoid duplicate output when the
        # module is reloaded in tests.
        root.handlers = [handler]

    return logging.getLogger("exit_management_agent")
