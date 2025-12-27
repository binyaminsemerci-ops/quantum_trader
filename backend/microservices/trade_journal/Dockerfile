FROM python:3.11-slim

LABEL maintainer="Quantum Trader AI System"
LABEL description="Phase 7: Trade Journal & Performance Analytics"

WORKDIR /app

# Copy service file from correct context path
COPY backend/microservices/trade_journal/journal_service.py .

# Install dependencies
RUN pip install --no-cache-dir \
    redis==7.1.0

# Create reports directory
RUN mkdir -p /app/reports

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

# Run service
CMD ["python", "-u", "journal_service.py"]
