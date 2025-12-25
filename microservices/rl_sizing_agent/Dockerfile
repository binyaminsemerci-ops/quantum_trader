FROM python:3.11-slim

WORKDIR /app

# Install basic requirements
RUN pip install --no-cache-dir redis torch numpy structlog

# Copy RL sizing agent
COPY . /app/microservices/rl_sizing_agent

ENV PYTHONPATH=/app
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

CMD ["python", "-m", "microservices.rl_sizing_agent.pnl_feedback_listener"]
