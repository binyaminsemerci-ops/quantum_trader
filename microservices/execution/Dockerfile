# Execution Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY . .

# Copy shared backend modules (temporary until modules are fully extracted)
COPY ../../backend /app/backend

# Create data directories
RUN mkdir -p logs data/event_buffers/execution

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8002/health')"

# Run service
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "info"]
