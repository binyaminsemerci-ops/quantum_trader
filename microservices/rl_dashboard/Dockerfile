FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir flask flask-socketio redis
EXPOSE 8025
CMD ["python3", "app.py"]
