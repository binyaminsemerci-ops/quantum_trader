FROM python:3.11-slim

WORKDIR /app

# Copy alert service
COPY backend/microservices/governance_alerts/alert_service.py .

# Install dependencies
RUN pip install --no-cache-dir \
    redis==7.1.0 \
    requests==2.31.0 \
    psutil==5.9.6

# Create logs directory
RUN mkdir -p /app/logs

# Environment variables
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import psutil; exit(0 if psutil.cpu_percent() >= 0 else 1)"

# Run alert service
CMD ["python", "alert_service.py"]
