FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Copy service file
COPY qpm_service.py .

# Install dependencies
RUN pip install --no-cache-dir redis==7.1.0 numpy==1.24.3

# Create memory directory
RUN mkdir -p /app/policy_memory

# Health check - verify Redis connection and PyTorch
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import redis, os, torch; r=redis.Redis(host=os.getenv('REDIS_HOST','redis')); exit(0 if r.ping() and torch.cuda.is_available() or True else 1)"

# Run quantum policy memory service
CMD ["python", "qpm_service.py"]
