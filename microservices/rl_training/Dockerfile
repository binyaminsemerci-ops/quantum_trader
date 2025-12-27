FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY microservices/rl_training/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY microservices/rl_training/ /app/microservices/rl_training/

# Copy shared backend core (if needed)
COPY backend/core/ /app/backend/core/

# Expose port
EXPOSE 8005

# Run service
CMD ["python", "-m", "uvicorn", "microservices.rl_training.main:app", "--host", "0.0.0.0", "--port", "8005"]
