# AI Engine Service Dockerfile  
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly (no requirements.txt needed when mounted)
RUN pip install --no-cache-dir \
    fastapi uvicorn redis structlog \
    xgboost lightgbm torch numpy pandas scikit-learn \
    aiohttp httpx python-dotenv prometheus-client \
    psutil requests

# Create data directories
RUN mkdir -p logs data/event_buffers/ai_engine models

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"

# Run service from /app root so imports work correctly
CMD ["python", "-m", "uvicorn", "microservices.ai_engine.main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]
