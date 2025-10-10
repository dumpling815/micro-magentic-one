# 원래 Agentic AI의 목적에 따르면, Agent가 구성한 코드를 유저의 컴퓨터에서 실행해야함.
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 


COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true

COPY common ./common

COPY coder ./coder
COPY coder/requirements.txt ./coder/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir -r coder/requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

CMD ["python3", "-m", "uvicorn", "coder.app:app", "--host", "0.0.0.0", "--port", "8080"]