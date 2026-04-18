"""OpenTelemetry tracer provider setup for bspQA."""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from backend.app.config import Settings


def configure_tracing(settings: Settings) -> None:
    """Configure and register the global OTel TracerProvider.

    Uses OTLPSpanExporter (HTTP) when ``otel_exporter_otlp_endpoint`` is set in
    Settings; otherwise falls back to ConsoleSpanExporter for local development.

    pydantic-settings does not populate os.environ, so we bridge the settings
    values into the environment variables the OTel SDK reads natively. This lets
    the SDK handle URL path construction and header parsing (including URL-decoding)
    using its own tested logic.

    Args:
        settings: Application settings (reads otel_* fields).
    """
    resource = Resource.create(
        {
            SERVICE_NAME: settings.otel_service_name,
            SERVICE_VERSION: "0.1.0",
        }
    )
    provider = TracerProvider(resource=resource)

    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # Bridge settings into os.environ so the SDK reads them the way it's
        # designed to — including appending /v1/traces and URL-decoding headers.
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = settings.otel_exporter_otlp_endpoint
        if settings.otel_exporter_otlp_headers:
            os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = settings.otel_exporter_otlp_headers

        exporter = OTLPSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
