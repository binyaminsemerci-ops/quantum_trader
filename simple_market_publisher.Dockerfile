FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir     redis     python-binance

# Copy publisher script
COPY simple_market_publisher.py /app/

# Run the publisher
CMD python3 simple_market_publisher.py
