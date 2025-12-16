# Dockerfile for Monitoring Health Service
# Sprint 2 - Service #6

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/

# Expose API port
EXPOSE 8080

# Environment variables
ENV REDIS_URL=redis://redis:6379
ENV LOG_LEVEL=INFO
ENV HEALTH_CHECK_INTERVAL=60
ENV BASE_SERVICE_URL=http://localhost:8000

# Run service
CMD ["python", "-m", "backend.services.monitoring_health_service.main"]
