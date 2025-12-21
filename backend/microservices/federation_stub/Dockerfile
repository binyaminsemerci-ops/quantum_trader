FROM python:3.11-slim

WORKDIR /app

COPY federation_stub_service.py .

RUN pip install redis

HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import redis; r=redis.Redis(host='redis'); r.ping()" || exit 1

CMD ["python", "federation_stub_service.py"]
