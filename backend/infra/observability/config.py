"""
Observability Configuration
EPIC-OBS-001 - Phase 2

Environment-driven configuration for tracing, metrics, and logging.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ObservabilityConfig:
    """
    Configuration for observability components.
    
    All values are read from environment variables with sensible defaults.
    """
    
    # Service identification
    service_name: str = os.getenv("SERVICE_NAME", "unknown-service")
    service_version: str = os.getenv("SERVICE_VERSION", "1.0.0")
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")  # "json" or "text"
    
    # OpenTelemetry Tracing
    otlp_endpoint: Optional[str] = os.getenv("OTLP_ENDPOINT")  # e.g., "http://jaeger:4317"
    otlp_insecure: bool = os.getenv("OTLP_INSECURE", "true").lower() == "true"
    trace_sample_rate: float = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))  # 0.0 to 1.0
    
    # Prometheus Metrics
    metrics_port: int = int(os.getenv("METRICS_PORT", "9090"))
    metrics_path: str = os.getenv("METRICS_PATH", "/metrics")
    
    # Feature flags
    enable_tracing: bool = os.getenv("ENABLE_TRACING", "true").lower() == "true"
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    enable_profiling: bool = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
    
    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Load configuration from environment variables."""
        return cls()
    
    def __repr__(self) -> str:
        return (
            f"ObservabilityConfig(service={self.service_name}, "
            f"env={self.environment}, tracing={self.enable_tracing}, "
            f"metrics={self.enable_metrics})"
        )


# Global config instance
config = ObservabilityConfig.from_env()
