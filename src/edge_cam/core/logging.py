"""结构化日志（engineering §6 core）：structlog，跨模块统一。"""

from __future__ import annotations

import structlog


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)
