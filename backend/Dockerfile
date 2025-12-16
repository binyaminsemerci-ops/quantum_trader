FROM python:3.11-slim

# Install system dependencies for sklearn/numpy/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Arbeidsmappe
WORKDIR /app

# Kopier kun backend requirement filer FØRST (for Docker layer caching)
COPY backend/requirements.txt ./

# Install med pip cache - MYE raskere rebuilds!
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Kopier backend kode inn i image
COPY backend/ ./backend/
COPY ai_engine/ ./ai_engine/
COPY database/ ./database/
COPY config/ ./config/

# Kopier root scripts folder (innehåller testnet_trading.py)
COPY scripts/ ./scripts/

# Kopier root models folder (XGBoost features)
COPY models/ ./models/

# Ensure models and logs directories exist
RUN mkdir -p /app/ai_engine/models /app/logs

# Keep working directory at repository root for package import path
WORKDIR /app

# Sørg for at Python finner backend-modulen
ENV PYTHONPATH=/app

# Default mode er prod, men kan overstyres ved build
ARG MODE=prod
ENV MODE=${MODE}

# Prod = uvicorn, Dev = tail
CMD if [ "$MODE" = "prod" ]; then \
        uvicorn backend.main:app --host 0.0.0.0 --port 8000; \
    else \
        tail -f /dev/null; \
    fi
