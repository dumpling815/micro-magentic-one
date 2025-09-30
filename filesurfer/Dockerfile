FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true

COPY common ./common

COPY filesurfer ./filesurfer
COPY filesurfer/requirements.txt ./filesurfer/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir -r filesurfer/requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

CMD ["python3", "-m", "uvicorn", "filesurfer.app:app", "--host", "0.0.0.0", "--port", "8080"]