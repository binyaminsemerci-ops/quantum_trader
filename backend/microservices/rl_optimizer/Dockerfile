FROM python:3.11-slim

LABEL maintainer="Quantum Trader AI System"
LABEL description="Phase 8: Reinforcement Learning Optimization Loop"

WORKDIR /app

# Copy service file
COPY backend/microservices/rl_optimizer/optimizer_service.py .

# Install dependencies
RUN pip install --no-cache-dir \
    redis==7.1.0 \
    numpy==1.24.3

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

# Run service
CMD ["python", "-u", "optimizer_service.py"]
