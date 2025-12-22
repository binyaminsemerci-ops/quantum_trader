FROM python:3.11-slim

WORKDIR /app/microservices/exitbrain_v3_5

# Install dependencies
RUN pip install --no-cache-dir \
    numpy \
    redis \
    aiohttp

# Copy source files
COPY . .

# Default command
CMD ["python3", "adaptive_leverage_engine.py", "--test"]
