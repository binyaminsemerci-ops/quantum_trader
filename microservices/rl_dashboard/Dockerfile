FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install flask flask_socketio redis
EXPOSE 8025
CMD ["python3", "app.py"]
