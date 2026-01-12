# 🔍 OpenTelemetry Tracing - Systemd Native Deployment

**ARCHITECTURE**: Native systemd (NO Docker)

This is the **systemd-compatible version** of the tracing setup from `copilot/add-tracing-to-workspace-again`, adapted to run on native Ubuntu with systemd services.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│  Quantum Trader Microservices (systemd)    │
│  ├─ ai-engine.service                      │
│  ├─ execution.service                      │
│  ├─ risk-safety.service                    │
│  └─ ... (7 total services)                 │
└─────────────────┬───────────────────────────┘
                  │ OpenTelemetry SDK
                  │ OTLP gRPC (port 4317)
                  ▼
┌─────────────────────────────────────────────┐
│  Jaeger All-in-One (systemd)               │
│  ├─ OTLP Collector (4317/4318)             │
│  ├─ Storage (in-memory)                    │
│  └─ Query UI (16686)                       │
└─────────────────────────────────────────────┘
```

## 📦 What's Different from Docker Version?

| Component | Docker (Copilot) | Systemd (This) |
|-----------|------------------|----------------|
| **Jaeger** | `jaegertracing/all-in-one:1.52` container | Native binary `/opt/jaeger/jaeger-all-in-one` |
| **Backend** | `quantum_backend` container | `quantum-*.service` systemd units |
| **Network** | Docker bridge `quantum_trader` | localhost (127.0.0.1) |
| **OTLP Endpoint** | `http://jaeger:4317` | `http://localhost:4317` |
| **Management** | `docker-compose up jaeger` | `systemctl start jaeger` |

## 🚀 Installation

### 1. Install Jaeger Native

```bash
# On VPS (Hetzner 46.224.116.254)
sudo bash ops/tracing/install_jaeger_native.sh
```

This installs:
- Jaeger binary at `/opt/jaeger/jaeger-all-in-one`
- Systemd service at `/etc/systemd/system/jaeger.service`
- UI config at `/etc/jaeger/ui-config.json`

### 2. Configure Microservices

Each microservice needs OpenTelemetry environment variables:

```bash
# /home/qt/quantum_trader/.env
ENABLE_TRACING=true
OTLP_ENDPOINT=http://localhost:4317
OTLP_INSECURE=true
TRACE_SAMPLE_RATE=1.0
SERVICE_NAME=ai-engine  # Different for each service
```

### 3. Verify

```bash
# Check Jaeger service
systemctl status jaeger

# Check microservice logs for tracing
journalctl -u quantum-ai-engine -f | grep -i "tracing\|otlp"

# Open Jaeger UI (via SSH tunnel)
ssh -L 16686:localhost:16686 root@46.224.116.254
# Then open: http://localhost:16686
```

## 🔧 Service Management

```bash
# Start/stop Jaeger
sudo systemctl start jaeger
sudo systemctl stop jaeger
sudo systemctl restart jaeger

# View logs
journalctl -u jaeger -f

# Check ports
ss -tlnp | grep -E "4317|4318|16686"
```

## 🎯 Tracing Workflow

1. **Microservice starts** → Initializes OpenTelemetry SDK
2. **Request arrives** → SDK creates span with trace ID
3. **SDK exports** → Sends span to Jaeger via OTLP (gRPC 4317)
4. **Jaeger stores** → In-memory storage (default)
5. **User queries** → Jaeger UI at http://localhost:16686

## 📊 Example: Tracing AI Prediction

```python
# backend/microservices/ai_engine/main.py
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@app.post("/predict")
async def predict(symbol: str):
    with tracer.start_as_current_span("ai_predict") as span:
        span.set_attribute("symbol", symbol)
        
        # Call model
        result = await model.predict(symbol)
        
        span.set_attribute("confidence", result["confidence"])
        return result
```

In Jaeger UI you'll see:
- Service: `ai-engine`
- Operation: `ai_predict`
- Tags: `symbol=BTCUSDT`, `confidence=0.85`

## 🐛 Troubleshooting

### Tracing not working?

```bash
# 1. Check Jaeger is running
systemctl is-active jaeger
curl -s http://localhost:16686 | head -10

# 2. Check microservice has OpenTelemetry
journalctl -u quantum-ai-engine | grep "OpenTelemetry initialized"

# 3. Check OTLP endpoint is reachable
curl -v http://localhost:4317

# 4. Check for errors
journalctl -u jaeger | grep -i error
journalctl -u quantum-ai-engine | grep -i "otlp\|tracing"
```

### No traces in Jaeger UI?

- Check `TRACE_SAMPLE_RATE=1.0` (100% sampling)
- Verify `OTLP_ENDPOINT=http://localhost:4317` (not `http://jaeger:4317`)
- Restart microservice: `sudo systemctl restart quantum-ai-engine`

## 🔐 Production Considerations

### Resource Limits

Jaeger systemd service has:
- `MemoryMax=512M`
- `CPUQuota=50%`

Adjust in `/etc/systemd/system/jaeger.service` if needed.

### Persistent Storage

Default: in-memory (data lost on restart)

For production persistence, add Cassandra or Elasticsearch backend:

```bash
# Install Cassandra native
apt install cassandra

# Update jaeger.service ExecStart
ExecStart=/opt/jaeger/jaeger-all-in-one \
  --span-storage.type=cassandra \
  --cassandra.servers=localhost:9042
```

### Firewall

Jaeger UI (16686) is NOT exposed externally. Access via SSH tunnel:

```bash
# From local machine
ssh -L 16686:localhost:16686 root@46.224.116.254

# Open browser: http://localhost:16686
```

## 📚 References

- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [OBSERVABILITY_README.md](../../docs/OBSERVABILITY_README.md) - Microservice instrumentation guide

## ✅ Migration from Docker

If you have the old Docker setup:

```bash
# 1. Stop Docker Jaeger
docker stop quantum_jaeger
docker rm quantum_jaeger

# 2. Install native Jaeger
sudo bash ops/tracing/install_jaeger_native.sh

# 3. Update .env (change jaeger → localhost)
sed -i 's|http://jaeger:4317|http://localhost:4317|g' /home/qt/quantum_trader/.env

# 4. Restart microservices
sudo systemctl restart quantum-trader.target
```

---

**SYSTEMD ONLY** ✅  
This deployment does NOT use Docker.
