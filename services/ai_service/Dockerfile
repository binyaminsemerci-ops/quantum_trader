# AI Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ backend/
COPY ai_engine/ ai_engine/
COPY services/ai_service/ services/ai_service/

# Create models directory
RUN mkdir -p ai_engine/models

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run service
CMD ["python", "-u", "services/ai_service/run_ai_service.py"]
