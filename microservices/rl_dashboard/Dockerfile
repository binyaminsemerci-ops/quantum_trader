FROM python:3.10-slim
WORKDIR /app
COPY . .

# Install docker CLI for stats collection
RUN apt-get update && \
    apt-get install -y docker.io && \
    rm -rf /var/lib/apt/lists/*

RUN pip install flask flask_socketio redis
EXPOSE 8025
CMD ["python3", "app.py"]
