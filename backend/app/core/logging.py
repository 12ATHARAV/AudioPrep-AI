"""
Structured logging with structlog.

Interview Note: Structured logging (JSON) is essential for production because:
1. Machine-parseable (ELK, Datadog, CloudWatch)
2. Searchable fields (job_id, stage, duration_ms)
3. Correlation IDs across async pipeline stages
"""

import structlog
import logging
import sys


def setup_logging(log_level: str = "INFO"):
    """Configure structured logging for the application."""

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            # In production, switch to JSONRenderer
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Get a logger with the given name bound as context."""
    return structlog.get_logger(name)