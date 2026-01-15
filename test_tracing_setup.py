#!/usr/bin/env python3
"""
Test script to verify OpenTelemetry tracing setup.

This script tests:
1. OpenTelemetry dependencies are installed
2. Tracing configuration is loaded
3. Manual span creation works
4. FastAPI instrumentation is available

Run: python test_tracing_setup.py
"""

import sys
import os

def test_imports():
    """Test that OpenTelemetry packages can be imported."""
    print("Testing OpenTelemetry imports...")
    
    try:
        from opentelemetry import trace
        print("✅ opentelemetry.trace imported")
    except ImportError as e:
        print(f"❌ Failed to import opentelemetry.trace: {e}")
        return False
    
    try:
        from opentelemetry.sdk.trace import TracerProvider
        print("✅ opentelemetry.sdk.trace imported")
    except ImportError as e:
        print(f"❌ Failed to import opentelemetry.sdk.trace: {e}")
        return False
    
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        print("✅ opentelemetry.exporter.otlp imported")
    except ImportError as e:
        print(f"❌ Failed to import opentelemetry.exporter.otlp: {e}")
        return False
    
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        print("✅ opentelemetry.instrumentation.fastapi imported")
    except ImportError as e:
        print(f"❌ Failed to import opentelemetry.instrumentation.fastapi: {e}")
        return False
    
    return True


def test_observability_module():
    """Test that observability module can be imported and initialized."""
    print("\nTesting observability module...")
    
    # Add backend to path if needed
    backend_path = os.path.join(os.path.dirname(__file__), 'backend')
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    
    try:
        from backend.infra.observability import (
            init_observability,
            get_tracer,
            instrument_fastapi,
        )
        print("✅ Observability module imported")
    except ImportError as e:
        print(f"❌ Failed to import observability module: {e}")
        return False
    
    # Test initialization
    try:
        init_observability(
            service_name="test-service",
            log_level="INFO",
            enable_tracing=True,
            enable_metrics=True,
        )
        print("✅ Observability initialized")
    except Exception as e:
        print(f"❌ Failed to initialize observability: {e}")
        return False
    
    # Test tracer
    try:
        tracer = get_tracer()
        print(f"✅ Got tracer: {tracer}")
    except Exception as e:
        print(f"❌ Failed to get tracer: {e}")
        return False
    
    return True


def test_manual_span():
    """Test creating a manual span."""
    print("\nTesting manual span creation...")
    
    try:
        from backend.infra.observability import get_tracer
        
        tracer = get_tracer()
        
        # Create a test span
        with tracer.start_as_current_span("test_operation") as span:
            span.set_attribute("test.attribute", "test_value")
            span.set_attribute("test.number", 42)
            print("✅ Created span with attributes")
        
        print("✅ Manual span test passed")
        return True
    except Exception as e:
        print(f"❌ Manual span test failed: {e}")
        return False


def test_fastapi_instrumentation():
    """Test FastAPI instrumentation."""
    print("\nTesting FastAPI instrumentation...")
    
    try:
        from fastapi import FastAPI
        from backend.infra.observability import instrument_fastapi
        
        app = FastAPI()
        instrument_fastapi(app)
        
        print("✅ FastAPI instrumentation successful")
        return True
    except Exception as e:
        print(f"❌ FastAPI instrumentation failed: {e}")
        return False


def test_config():
    """Test observability configuration."""
    print("\nTesting observability configuration...")
    
    try:
        from backend.infra.observability.config import ObservabilityConfig
        
        config = ObservabilityConfig()
        
        print(f"  Service Name: {config.service_name}")
        print(f"  Environment: {config.environment}")
        print(f"  Tracing Enabled: {config.enable_tracing}")
        print(f"  OTLP Endpoint: {config.otlp_endpoint or 'Not set'}")
        print(f"  Trace Sample Rate: {config.trace_sample_rate}")
        
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("OpenTelemetry Tracing Setup Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Import Test", test_imports()))
    results.append(("Observability Module Test", test_observability_module()))
    results.append(("Manual Span Test", test_manual_span()))
    results.append(("FastAPI Instrumentation Test", test_fastapi_instrumentation()))
    results.append(("Configuration Test", test_config()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Tracing setup is working correctly.")
        print("\nNext steps:")
        print("1. Start Jaeger: docker-compose up -d jaeger")
        print("2. Start backend: docker-compose up -d backend")
        print("3. Make some API requests")
        print("4. View traces at http://localhost:16686")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\nTo fix:")
        print("1. Install dependencies: pip install -r backend/requirements.txt")
        print("2. Check that backend/ is in PYTHONPATH")
        print("3. Verify OpenTelemetry packages are installed")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
