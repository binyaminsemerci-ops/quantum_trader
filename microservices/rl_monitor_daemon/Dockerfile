FROM python:3.10-slim
WORKDIR /app
COPY rl_monitor.py .
RUN pip install redis matplotlib requests
CMD ["python3", "rl_monitor.py"]
