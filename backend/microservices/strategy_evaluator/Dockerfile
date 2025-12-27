FROM python:3.11-slim

LABEL maintainer="Quantum Trader Team"
LABEL description="Phase 9: Meta-Cognitive Strategy Evaluator"

WORKDIR /app

# Copy service file
COPY backend/microservices/strategy_evaluator/evaluator_service.py .

# Install dependencies
RUN pip install --no-cache-dir \
    redis==7.1.0 \
    numpy==1.24.3

# Create sandbox directory for strategy variants
RUN mkdir -p /app/sandbox_strategies

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='redis', decode_responses=True); r.ping()" || exit 1

CMD ["python", "-u", "evaluator_service.py"]
