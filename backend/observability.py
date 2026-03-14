"""Logging, metrics, and tracing helpers for agent orchestration."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from hashlib import sha256
from typing import Any, Dict, Optional

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
except Exception:  # pragma: no cover - fallback for minimal environments
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _NoopMetric:
        def labels(self, **kwargs):
            return self

        def inc(self, amount: float = 1.0) -> None:
            return None

        def observe(self, value: float) -> None:
            return None

    def Counter(*args, **kwargs):  # type: ignore[override]
        return _NoopMetric()

    def Histogram(*args, **kwargs):  # type: ignore[override]
        return _NoopMetric()

    def generate_latest() -> bytes:  # type: ignore[override]
        return b""

REQUEST_LATENCY_SECONDS = Histogram(
    "chat_request_latency_seconds",
    "Chat request latency in seconds.",
    labelnames=("route",),
)
TOOL_CALLS_TOTAL = Counter(
    "tool_calls_total",
    "Count of tool calls by tool and status.",
    labelnames=("tool_id", "status"),
)
VALIDATION_FAILURES_TOTAL = Counter(
    "validation_failures_total",
    "Count of response validation failures.",
)
RETRY_EVENTS_TOTAL = Counter(
    "retry_events_total",
    "Count of retry events by operation type.",
    labelnames=("operation",),
)
FALLBACK_EVENTS_TOTAL = Counter(
    "fallback_events_total",
    "Count of fallback invocations.",
)


class JsonFormatter(logging.Formatter):
    """Simple JSON log formatter for Loki-friendly logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=True)


class LokiHttpHandler(logging.Handler):
    """Minimal Loki push handler using HTTP API."""

    def __init__(self, push_url: str, labels: Optional[Dict[str, str]] = None) -> None:
        super().__init__()
        self.push_url = push_url
        self.labels = labels or {"app": "chat-backend"}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            payload = {
                "streams": [
                    {
                        "stream": self.labels,
                        "values": [[str(int(time.time() * 1e9)), line]],
                    }
                ]
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.push_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=1.5)
        except (urllib.error.URLError, TimeoutError, ValueError):
            return


def get_logger(name: str = "chat.orchestrator") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    from config import settings
    loki_push_url = settings.loki_push_url
    if loki_push_url:
        loki_handler = LokiHttpHandler(
            push_url=loki_push_url,
            labels={"app": "chat-backend", "logger": name},
        )
        loki_handler.setFormatter(JsonFormatter())
        logger.addHandler(loki_handler)

    logger.propagate = False
    return logger


def init_tracing() -> None:
    """Initialize OTEL tracing when endpoint is configured."""
    from config import settings

    endpoint = settings.otel_exporter_otlp_endpoint
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        return

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": settings.otel_service_name,
                "service.version": settings.service_version,
            }
        )
    )
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def init_observability() -> None:
    """Bootstrap logging + optional tracing."""
    get_logger()
    init_tracing()


def metrics_payload() -> bytes:
    return generate_latest()


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST


def hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


SENSITIVE_KEYS = {
    "prompt",
    "message",
    "content",
    "query",
    "tool_input",
    "tool_output",
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
}


def _sanitize_value(key: str, value: Any) -> Any:
    lowered = key.lower()
    if lowered in SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {k: _sanitize_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(key, item) for item in value]
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "...(truncated)"
    return value


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    payload = data or {}
    safe_payload = {k: _sanitize_value(k, v) for k, v in payload.items()}
    logger.log(level, message, extra={"extra_data": safe_payload})
