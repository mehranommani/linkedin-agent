FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# App code
COPY backend/ /app/backend/
COPY scripts/ /app/scripts/

# Data dir
RUN mkdir -p /app/data

ENV PYTHONPATH=/app
ENV DATABASE_PATH=/app/data/linkedin_agent.duckdb

EXPOSE 9000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "9000"]
