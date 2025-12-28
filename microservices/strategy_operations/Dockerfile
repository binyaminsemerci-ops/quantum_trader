FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install redis torch
CMD ["python3", "strategy_ops.py"]
