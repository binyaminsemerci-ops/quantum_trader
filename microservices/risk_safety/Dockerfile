# Risk & Safety Service - Docker Image

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY microservices/risk_safety/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY microservices/risk_safety /app/microservices/risk_safety

# Copy shared backend modules (ESS, PolicyStore, EventBus)
COPY backend/core /app/backend/core
COPY backend/utils /app/backend/utils

# Create logs directory
RUN mkdir -p logs data/event_buffers/risk_safety

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8003/health').raise_for_status()"

# Run service (use module path from /app root)
WORKDIR /app
CMD ["python", "-m", "uvicorn", "microservices.risk_safety.main:app", "--host", "0.0.0.0", "--port", "8003"]
