# Backend image for the Hybrid RAG API
FROM python:3.11-slim

# Python behaves better in containers with these
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first, so this layer is cached and rebuilds stay fast
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -r requirements-dev.txt

# Model cache lives here so it is not re-downloaded on every restart
ENV HF_HOME=/models

COPY app ./app
COPY tests ./tests

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]