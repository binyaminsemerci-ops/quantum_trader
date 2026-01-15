# Distributed Tracing Setup

## Overview

The Quantum Trader platform now includes distributed tracing using OpenTelemetry and Jaeger. This allows you to trace requests across all microservices and understand the flow of operations through the system.

## Architecture

- **OpenTelemetry**: Industry-standard instrumentation for distributed tracing
- **Jaeger**: Open-source trace collection and visualization platform
- **OTLP Protocol**: OpenTelemetry Protocol for exporting traces

## Components

### 1. Tracing Infrastructure (`backend/infra/observability/`)

- `tracing.py`: Core tracing module with OpenTelemetry integration
- `config.py`: Environment-driven configuration
- `__init__.py`: Unified observability interface
- Graceful degradation if OpenTelemetry is not installed

### 2. Jaeger All-in-One (Docker Compose)

- Container: `quantum_jaeger`
- UI Port: `16686` - Access at http://localhost:16686
- OTLP gRPC: `4317` - For trace collection
- OTLP HTTP: `4318` - Alternative trace collection endpoint

### 3. Instrumented Services

- **Backend API** (`backend/main.py`): Main FastAPI service
- **AI Engine** (`microservices/ai_engine/main.py`): AI model inference service
- **Execution Service** (`microservices/execution/main.py`): Trade execution service

## Quick Start

### 1. Install Dependencies

OpenTelemetry dependencies are included in `backend/requirements.txt`:

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Add to your `.env` file (or use docker-compose environment):

```bash
# Service identification
SERVICE_NAME=quantum-backend
SERVICE_VERSION=1.0.0
ENVIRONMENT=development

# Enable tracing
ENABLE_TRACING=true

# Jaeger OTLP endpoint
OTLP_ENDPOINT=http://jaeger:4317
OTLP_INSECURE=true

# Trace sampling (1.0 = 100% of requests)
TRACE_SAMPLE_RATE=1.0
```

### 3. Start Services with Tracing

```bash
# Start all services including Jaeger
docker-compose --profile observability --profile dev up -d

# Or start specific services
docker-compose up -d backend jaeger
```

### 4. Access Jaeger UI

Open your browser to: http://localhost:16686

You should see traces from:
- `quantum-backend`
- `ai-engine`
- `execution` (if running)

## Using Tracing in Code

### Automatic Instrumentation

FastAPI routes are automatically traced when you call `instrument_fastapi(app)`:

```python
from backend.infra.observability import init_observability, instrument_fastapi

# Initialize at startup
init_observability(
    service_name="my-service",
    log_level="INFO",
    enable_tracing=True,
)

# Instrument FastAPI app
instrument_fastapi(app)
```

### Manual Tracing

Add custom spans to trace specific operations:

```python
from backend.infra.observability import get_tracer

tracer = get_tracer()

# Create a span
with tracer.start_as_current_span("process_signal") as span:
    span.set_attribute("symbol", "BTCUSDT")
    span.set_attribute("confidence", 0.85)
    
    # Your code here
    result = process_signal(data)
    
    span.set_attribute("result", result)
```

### Span Attributes

Add context to your traces with attributes:

```python
span.set_attribute("user.id", user_id)
span.set_attribute("trade.symbol", "BTCUSDT")
span.set_attribute("trade.size", 0.1)
span.set_attribute("ai.confidence", 0.85)
span.set_attribute("error", False)
```

## Trace Examples

### Example 1: API Request Flow

```
HTTP GET /api/signals
├─ query_signals (database)
│  ├─ connect_db
│  └─ execute_query
├─ filter_signals (business logic)
└─ format_response
```

### Example 2: Trade Execution Flow

```
POST /api/trades/execute
├─ validate_request
├─ ai-engine: get_signal
│  ├─ load_models
│  └─ predict
├─ execution: place_order
│  ├─ check_balance
│  ├─ calculate_size
│  └─ submit_to_exchange
└─ store_trade
```

## Troubleshooting

### No Traces Appearing

1. **Check Jaeger is running:**
   ```bash
   docker ps | grep jaeger
   curl http://localhost:16686/api/services
   ```

2. **Verify OTLP endpoint:**
   ```bash
   # Check environment variable
   docker exec quantum_backend env | grep OTLP_ENDPOINT
   ```

3. **Check logs for tracing initialization:**
   ```bash
   docker logs quantum_backend | grep -i "tracing\|opentelemetry"
   ```

4. **Verify OpenTelemetry installed:**
   ```bash
   docker exec quantum_backend python -c "import opentelemetry; print('OK')"
   ```

### Tracing Disabled Warning

If you see "OpenTelemetry not available" warnings, install dependencies:

```bash
pip install opentelemetry-api opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-exporter-otlp
```

### High Trace Volume

Reduce sampling rate to trace only a percentage of requests:

```bash
TRACE_SAMPLE_RATE=0.1  # 10% of requests
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | `unknown-service` | Service name in traces |
| `SERVICE_VERSION` | `1.0.0` | Service version |
| `ENVIRONMENT` | `development` | Environment (dev/staging/prod) |
| `ENABLE_TRACING` | `true` | Enable/disable tracing |
| `OTLP_ENDPOINT` | `None` | OTLP collector endpoint |
| `OTLP_INSECURE` | `true` | Use insecure gRPC |
| `TRACE_SAMPLE_RATE` | `1.0` | Sampling rate (0.0-1.0) |

### Docker Compose Profiles

- `observability`: Starts Jaeger and tracing infrastructure
- `dev`: Development profile (includes observability)
- `prod`: Production profile (includes observability)

## Best Practices

1. **Name spans meaningfully**: Use descriptive names like `process_signal` not `func1`
2. **Add relevant attributes**: Include identifiers (symbol, user_id) for filtering
3. **Don't over-instrument**: Trace important operations, not every function
4. **Use sampling in production**: Set `TRACE_SAMPLE_RATE=0.1` to reduce overhead
5. **Monitor Jaeger resources**: High trace volume can consume significant storage

## Advanced: Production Deployment

For production, consider:

1. **Separate Jaeger deployment**: Use dedicated Jaeger collector and storage
2. **Storage backend**: Configure Elasticsearch or Cassandra for persistence
3. **Authentication**: Secure Jaeger UI with authentication
4. **Sampling strategy**: Implement head-based or tail-based sampling
5. **Retention policies**: Set trace retention based on storage capacity

### Production Example

```yaml
# Use Jaeger collector with Elasticsearch backend
jaeger-collector:
  image: jaegertracing/jaeger-collector:1.52
  environment:
    - SPAN_STORAGE_TYPE=elasticsearch
    - ES_SERVER_URLS=http://elasticsearch:9200
  ports:
    - "4317:4317"  # OTLP gRPC

jaeger-query:
  image: jaegertracing/jaeger-query:1.52
  environment:
    - SPAN_STORAGE_TYPE=elasticsearch
    - ES_SERVER_URLS=http://elasticsearch:9200
  ports:
    - "16686:16686"
```

## Resources

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [OpenTelemetry Python API](https://opentelemetry-python.readthedocs.io/)
- [FastAPI OpenTelemetry Integration](https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation/opentelemetry-instrumentation-fastapi)

## Support

For issues or questions:
1. Check logs: `docker logs quantum_backend`
2. Verify configuration: `docker exec quantum_backend env | grep -i trace`
3. Test connectivity: `curl http://localhost:16686/api/services`
4. Review this documentation and OpenTelemetry docs
