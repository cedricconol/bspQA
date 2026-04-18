"""Structured JSON logging with OpenTelemetry trace correlation."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from opentelemetry import trace


class _OtelTraceContextFilter(logging.Filter):
    """Inject trace_id and span_id from the active OTel span into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach the current OTel trace context to a log record.

        Args:
            record: The log record to annotate.

        Returns:
            Always True — this filter never drops records.
        """
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            record.trace_id = format(ctx.trace_id, "032x")  # type: ignore[attr-defined]
            record.span_id = format(ctx.span_id, "016x")  # type: ignore[attr-defined]
        else:
            record.trace_id = ""  # type: ignore[attr-defined]
            record.span_id = ""  # type: ignore[attr-defined]
        return True


_RESERVED_LOG_ATTRS = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        "trace_id",
        "span_id",
    }
)


class _JsonFormatter(logging.Formatter):
    """Serialize log records as single-line JSON for stdout / Grafana Loki.

    Fields emitted per record: ts, level, logger, trace_id, span_id, msg,
    plus any keys passed via ``extra={}``.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a LogRecord to a JSON line.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string.
        """
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": getattr(record, "trace_id", ""),
            "span_id": getattr(record, "span_id", ""),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured JSON logging on stdout for the entire process.

    Attaches an OTel trace-context filter so every log line includes
    ``trace_id`` and ``span_id`` for correlation in Grafana. Call once at
    application startup, before any logger is used.

    Args:
        level: Root log level (default INFO).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_OtelTraceContextFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    # Silence uvicorn's access log — FastAPIInstrumentor covers request spans
    logging.getLogger("uvicorn.access").propagate = False
