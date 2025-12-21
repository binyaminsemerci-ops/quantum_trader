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
    ujson

# Copy source files
COPY . /app/

# Default command (can be overridden)
CMD ["python3", "exchange_stream_bridge.py"]
