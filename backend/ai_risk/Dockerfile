FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install core dependencies for Risk Brain
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    redis \
    pydantic \
    httpx \
    numpy \
    pandas \
    structlog

# Copy backend code
COPY backend /app/backend

# Set environment
ENV PYTHONPATH=/app
ENV ENABLE_RISK_BRAIN=true
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Expose port
EXPOSE 8012

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8012/health || exit 1

# Run Risk Brain service
CMD ["python", "-m", "uvicorn", "backend.ai_risk.service:app", "--host", "0.0.0.0", "--port", "8012"]
