FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY . /app/microservices/training_worker

# Create models directory
RUN mkdir -p /app/models

WORKDIR /app/microservices/training_worker

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import redis; r=redis.from_url('redis://redis:6379/0'); r.ping()"

CMD ["python", "retrain_worker.py"]
