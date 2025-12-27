FROM python:3.11-slim

WORKDIR /app

# Copy microservices folder for intelligent leverage engine
COPY microservices/ ./microservices/

# Copy executor service
COPY backend/microservices/auto_executor/executor_service.py .

# Install dependencies (add numpy for leverage engine)
RUN pip install --no-cache-dir \
    python-binance==1.0.19 \
    redis==7.1.0 \
    numpy==1.24.3

# Environment variables
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='redis'); exit(0 if r.ping() else 1)"

# Run executor service
CMD ["python", "executor_service.py"]
