FROM python:3.11-slim

WORKDIR /app

# Copy backend dependencies first (caching)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy position monitor service
COPY microservices/position_monitor /app/microservices/position_monitor

# Copy backend modules (needed for dependencies)
COPY backend /app/backend

# Copy shared modules (needed for logging)
COPY shared /app/shared

# Copy config module (needed for position monitor)
COPY config /app/config

# Copy execution modules (for Exit Brain v3)
COPY microservices/execution/exit_brain_v3 /app/microservices/execution/exit_brain_v3

# Copy Exit Brain v3.5 with ILFv2
COPY microservices/exitbrain_v3_5 /app/microservices/exitbrain_v3_5

ENV PYTHONPATH=/app:/app/microservices
ENV EXIT_BRAIN_V3_ENABLED=true

CMD ["python", "-m", "microservices.position_monitor.main"]
