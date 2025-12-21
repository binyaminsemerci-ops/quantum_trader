FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    numpy \
    redis \
    aiohttp

# Copy source files
COPY . /app/

# Default command
CMD ["python3", "adaptive_leverage_engine.py", "--test"]
