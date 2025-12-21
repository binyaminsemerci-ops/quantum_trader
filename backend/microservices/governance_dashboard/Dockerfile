FROM python:3.11-slim

WORKDIR /app

# Copy application
COPY backend/microservices/governance_dashboard/app.py .

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.125.0 \
    uvicorn==0.38.0 \
    redis==7.1.0 \
    httpx==0.28.1

# Expose port
EXPOSE 8501

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8501", "--reload"]
