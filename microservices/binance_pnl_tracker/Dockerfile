FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    aiohttp \
    redis

# Copy tracker script
COPY binance_pnl_tracker.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

CMD ["python3", "-u", "binance_pnl_tracker.py"]
