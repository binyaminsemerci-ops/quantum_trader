FROM python:3.11-slim

LABEL maintainer="Quantum Trader Team"
LABEL description="Meta-Regime Correlator - Phase 4R"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import redis; r=redis.from_url('redis://redis:6379/0'); r.ping()" || exit 1

# Run service
CMD ["python", "meta_regime_service.py"]
