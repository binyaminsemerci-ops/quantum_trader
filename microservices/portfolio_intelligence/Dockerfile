FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY microservices/portfolio_intelligence/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy service code
COPY microservices/portfolio_intelligence/ /app/microservices/portfolio_intelligence/
COPY backend/core/ /app/backend/core/

# Expose port
EXPOSE 8004

# Run service
CMD ["python", "-m", "uvicorn", "microservices.portfolio_intelligence.main:app", "--host", "0.0.0.0", "--port", "8004"]
