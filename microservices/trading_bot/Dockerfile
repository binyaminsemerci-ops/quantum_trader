FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY microservices/trading_bot/requirements.txt /app/microservices/trading_bot/

# Install Python packages
RUN pip install --no-cache-dir -r /app/microservices/trading_bot/requirements.txt

# Copy trading bot code
COPY microservices/trading_bot/*.py /app/microservices/trading_bot/

# Copy backend modules (for EventBus and RL Agent)
COPY backend/utils/ /app/backend/utils/
COPY backend/services/ /app/backend/services/
COPY backend/__init__.py /app/backend/

# Create __init__.py files
RUN touch /app/microservices/__init__.py

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')"

# Run application
CMD ["uvicorn", "microservices.trading_bot.main:app", "--host", "0.0.0.0", "--port", "8003"]
