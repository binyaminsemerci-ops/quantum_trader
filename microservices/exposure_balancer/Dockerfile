# Exposure Balancer Service - Phase 4P
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    redis \
    numpy \
    structlog

# Create log directory
RUN mkdir -p /app/logs

# Copy exposure balancer module
COPY microservices/exposure_balancer /app/microservices/exposure_balancer

# Set Python path
ENV PYTHONPATH=/app

# Run service
CMD ["python3", "/app/microservices/exposure_balancer/service.py"]
