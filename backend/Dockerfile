FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_OPTIONAL=false

COPY backend/requirements.txt backend/requirements-optional.txt ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && if [ "$INSTALL_OPTIONAL" = "true" ]; then pip install -r requirements-optional.txt; fi

COPY backend /app/backend
COPY ai_engine /app/ai_engine
COPY config /app/config
COPY scripts /app/scripts
COPY readme.md /app/readme.md
COPY DEVELOPMENT.md /app/DEVELOPMENT.md
COPY TODO.md /app/TODO.md

FROM base AS builder

RUN python -m compileall backend ai_engine config || true

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY --from=base /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=base /usr/local/bin /usr/local/bin
COPY --from=base /app /app

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -m http.client localhost 8000 || exit 1

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
