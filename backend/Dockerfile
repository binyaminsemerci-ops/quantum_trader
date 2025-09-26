FROM python:3.11-slim

# Arbeidsmappe
WORKDIR /app

# Installer dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# Kopier kode
COPY . .

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
