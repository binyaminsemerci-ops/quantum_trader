FROM python:3.11-slim

WORKDIR /app

# Copy backend requirements
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy RL sizing agent
COPY microservices/rl_sizing_agent /app/microservices/rl_sizing_agent

# Copy backend modules (needed for dependencies)
COPY backend /app/backend

ENV PYTHONPATH=/app
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

CMD ["python", "-m", "microservices.rl_sizing_agent.pnl_feedback_listener"]
