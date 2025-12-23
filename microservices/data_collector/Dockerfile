FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    aiohttp \
    websockets \
    redis \
    pandas \
    numpy \
    orjson \
    ujson \
    tenacity

# Copy backend infrastructure (for redis_manager)
COPY backend/infrastructure /app/backend/infrastructure

# Copy source files
COPY microservices/data_collector /app/

# Default command (can be overridden)
CMD ["python3", "exchange_stream_bridge.py"]
