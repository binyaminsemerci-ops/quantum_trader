FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY microservices/clm/requirements.txt /app/microservices/clm/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r microservices/clm/requirements.txt

# Copy project files
COPY backend/ /app/backend/
COPY microservices/clm/ /app/microservices/clm/
COPY ai_engine/ /app/ai_engine/

# Create directories for models and metadata
RUN mkdir -p /app/models /app/data/clm_v3/registry

# Set Python path
ENV PYTHONPATH=/app

# Run CLM service
CMD ["python", "-m", "microservices.clm.main"]
