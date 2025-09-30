# Websurfer는 내부에서 브라우저 자동화 (Playwright/Chromium) 사용하므로, Playwright python 베이스 이미지 사용
FROM mcr.microsoft.com/playwright/python:v1.47.0-jammy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HEADLESS=true

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true

COPY common ./common

COPY websurfer ./websurfer
COPY websurfer/requirements.txt ./websurfer/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir -r websurfer/requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

# 해당 베이스 이미지에는 pwuser가 있ㅡ므로, 소유권 맞춰주고 사용자 전환.
RUN chown -R pwuser:pwuser /app
USER pwuser

CMD ["python3", "-m", "uvicorn", "websurfer.app:app", "--host", "0.0.0.0", "--port", "8080"]