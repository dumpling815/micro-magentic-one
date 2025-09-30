# 원래 Agentic AI의 목적에 따르면, Agent가 구성한 코드를 유저의 컴퓨터에서 실행해야함.
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true

COPY common ./common

COPY user ./user
COPY user/requirements.txt ./user/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir -r user/requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

CMD ["python3", "-m", "uvicorn", "user.app:app", "--host", "0.0.0.0", "--port", "8080"]