from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from backend.app.api.routes.router import router
from backend.app.config import get_settings
from backend.app.observability.logging import configure_logging
from backend.app.observability.tracing import configure_tracing

_settings = get_settings()

configure_logging()  # must run before any logger fires
configure_tracing(_settings)  # register global TracerProvider

app = FastAPI()
FastAPIInstrumentor().instrument_app(app)  # auto-spans every HTTP request

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.allowed_origins.split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router)

_logger = logging.getLogger(__name__)
_logger.info("app.started", extra={"service": _settings.otel_service_name})
