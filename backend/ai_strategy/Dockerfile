FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install core dependencies for Strategy Brain
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    redis \
    pydantic \
    httpx \
    numpy \
    pandas

# Copy backend code
COPY backend /app/backend
COPY models /app/models 2>/dev/null || true
COPY data /app/data 2>/dev/null || true

# Set environment
ENV PYTHONPATH=/app
ENV ENABLE_STRATEGY_BRAIN=true
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Expose port
EXPOSE 8011

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8011/health || exit 1

# Run Strategy Brain service
CMD ["python", "-m", "uvicorn", "backend.ai_strategy.service:app", "--host", "0.0.0.0", "--port", "8011"]
