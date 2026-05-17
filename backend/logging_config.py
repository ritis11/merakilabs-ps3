"""structlog configuration. Single entrypoint: configure_logging()."""
import logging
import sys
from typing import IO, Optional

import structlog


def configure_logging(
    stream: Optional[IO[str]] = None,
    json_output: bool = True,
    level: str = "INFO",
) -> None:
    """Configure structlog for the application.

    Args:
        stream: where to write logs (default sys.stdout).
        json_output: emit JSON (True) or pretty console (False).
        level: log level name.
    """
    target = stream if stream is not None else sys.stdout
    handler = logging.StreamHandler(target)
    handler.setLevel(getattr(logging, level.upper()))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper()))

    renderer = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=target),
        cache_logger_on_first_use=False,
    )
