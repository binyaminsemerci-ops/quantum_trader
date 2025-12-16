"""
Distributed Tracing Module
EPIC-OBS-001 - Phase 2

OpenTelemetry-based distributed tracing for microservices.
"""

import logging
from typing import Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning(
        "OpenTelemetry not installed. Tracing disabled. "
        "Install: pip install opentelemetry-api opentelemetry-sdk "
        "opentelemetry-instrumentation-fastapi opentelemetry-exporter-otlp"
    )

from .config import config

# Global tracer instance
if OTEL_AVAILABLE:
    _tracer: Optional[trace.Tracer] = None
    _tracer_provider: Optional[TracerProvider] = None
else:
    _tracer = None
    _tracer_provider = None


def init_tracing(service_name: str) -> None:
    """
    Initialize OpenTelemetry tracing for the service.
    
    Args:
        service_name: Name of the service
    
    Environment Variables:
        OTLP_ENDPOINT: OTLP collector endpoint (e.g., "http://jaeger:4317")
        OTLP_INSECURE: Use insecure connection (default: true)
        TRACE_SAMPLE_RATE: Sampling rate 0.0-1.0 (default: 1.0)
        ENABLE_TRACING: Enable/disable tracing (default: true)
    
    Example:
        # At service startup
        init_tracing(service_name="ai-engine")
        
        # Tracer is now available globally
        tracer = get_tracer()
    """
    global _tracer, _tracer_provider
    
    if not OTEL_AVAILABLE:
        logging.warning(f"OpenTelemetry not available, tracing disabled for {service_name}")
        return
    
    if not config.enable_tracing:
        logging.info(f"Tracing disabled by config for {service_name}")
        return
    
    if _tracer_provider is not None:
        logging.warning(f"Tracing already initialized for {config.service_name}, skipping")
        return
    
    # Update config
    config.service_name = service_name
    
    # Create resource with service identification
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: config.service_version,
        "environment": config.environment,
    })
    
    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)
    
    # Configure exporter if OTLP endpoint is set
    if config.otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(
            endpoint=config.otlp_endpoint,
            insecure=config.otlp_insecure,
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        _tracer_provider.add_span_processor(span_processor)
        
        logging.info(
            f"OTLP tracing configured for {service_name}",
            extra={
                "otlp_endpoint": config.otlp_endpoint,
                "sample_rate": config.trace_sample_rate,
            }
        )
    else:
        logging.warning(
            f"OTLP_ENDPOINT not set, tracing enabled but spans won't be exported. "
            f"Set OTLP_ENDPOINT to send traces to Jaeger/Tempo."
        )
    
    # Set global tracer provider
    trace.set_tracer_provider(_tracer_provider)
    
    # Get tracer instance
    _tracer = trace.get_tracer(__name__)
    
    logging.info(f"OpenTelemetry tracing initialized for {service_name}")


def get_tracer():
    """
    Get the global tracer instance.
    
    Returns:
        OpenTelemetry tracer
    
    Example:
        tracer = get_tracer()
        
        with tracer.start_as_current_span("process_signal") as span:
            span.set_attribute("symbol", "BTCUSDT")
            result = process_signal(data)
            span.set_attribute("result", result)
    """
    global _tracer
    
    if not OTEL_AVAILABLE:
        # Return no-op tracer
        return trace.get_tracer(__name__)
    
    if _tracer is None:
        # Auto-initialize with default service name
        init_tracing(service_name=config.service_name)
    
    return _tracer or trace.get_tracer(__name__)


def instrument_fastapi(app) -> None:
    """
    Instrument a FastAPI application with automatic tracing.
    
    Args:
        app: FastAPI application instance
    
    Example:
        from fastapi import FastAPI
        from backend.infra.observability import init_tracing, instrument_fastapi
        
        app = FastAPI()
        init_tracing(service_name="ai-engine")
        instrument_fastapi(app)
        
        # All HTTP requests are now traced automatically
    """
    if not OTEL_AVAILABLE:
        logging.warning("OpenTelemetry not available, cannot instrument FastAPI")
        return
    
    if not config.enable_tracing:
        return
    
    try:
        FastAPIInstrumentor.instrument_app(app)
        logging.info(f"FastAPI instrumented with OpenTelemetry for {config.service_name}")
    except Exception as e:
        logging.error(f"Failed to instrument FastAPI: {e}")
