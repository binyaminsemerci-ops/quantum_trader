FROM python:3.11-slim

WORKDIR /app

# Copy evolution service
COPY evolution_service.py .

# Install dependencies
RUN pip install --no-cache-dir redis==7.1.0 numpy==1.24.3

# Create memory bank directory
RUN mkdir -p /app/memory_bank

# Health check - verify Redis connection
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import redis, os; redis.Redis(host=os.getenv('REDIS_HOST','redis')).ping()" || exit 1

# Run evolution service
CMD ["python", "evolution_service.py"]
