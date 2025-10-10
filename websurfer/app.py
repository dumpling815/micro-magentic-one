# 현재, WebSurfer의 경우는 브라우저(Chromium + Playwright)를 띄우고 모델이 스크린샷과 tool call을 처리해야함
# 따라서, 텍스트 전용 LLM으로는 Websurfer가 정상적으로 작동하지 않을 가능성이 높음.
# 빠르게 띄우기 위해서는 OpenAIChatCompletionClient를 사용하고, 이후에 OSS 로컬 멀티모달 대안을 검토해야함.

from fastapi import FastAPI, Body, HTTPException
import os, time
from common.request_schema import InvokeBody, InvokeResult, Msg

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken  # Supports task cancellation while async processing
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
#from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient # If possible, we'll use oss model in ollama

app = FastAPI(title="Magentic-One Web Surfer")

# --- ENV ---
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")  # e.g. openai, ollama
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b") # or "llava:13b"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "20"))
HEADLESS = os.getenv("HEADLESS", "true").lower() in ("true", "1", "yes") # GUI 사용 안 하는 경우 -> 리소스 절약, GUI 없는 서버환경에서 실행 가능.
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR") # Websurfer가 다운로드하는 파일 저장 경로를 지정.

# --- Lazy Singleton ---
_client = None
_agent = None
def get_agent() -> MultimodalWebSurfer:
    global _client, _agent
    if _agent is None:
        if _client in None:
            # if MODEL_PROVIDER == "openai":
            #     _client = OpenAIChatCompletionClient(model=OPENAI_MODEL, timeout=REQUEST_TIMEOUT)
            if MODEL_PROVIDER == "ollama":
                _client = OllamaChatCompletionClient(model=OLLAMA_MODEL, host=OLLAMA_HOST, timeout=REQUEST_TIMEOUT)
            else:
                raise RuntimeError(f"Unsupported model provider: {MODEL_PROVIDER}")
        _agent = MultimodalWebSurfer(
            name="websurfer",
            model_client=_client,
            downloads_folder=DOWNLOAD_DIR,
            headless=HEADLESS,
        )
    return _agent

# --- Endpoinsts ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": MODEL_PROVIDER,
        "openai_model": OPENAI_MODEL,
        "ollama_model": OLLAMA_MODEL,
        "headless": HEADLESS,
    }

@app.get("/ready")
def ready():
    try:
        _ = get_agent()
        return {"ready": True, "message": "WebSurfer agent is ready."}
    except Exception as e:
        raise HTTPException(503, f"not ready: {e}")
    
@app.post("/invoke", response_model=InvokeResult)
async def invoke(body: InvokeBody = Body(...)):
    start_time_perf = time.perf_counter()
    # JSON -> AutoGen message objects
    py_msgs = []
    for msg in body.messages:
        if msg.type == "TextMessage":
            py_msgs.append(TextMessage(content=msg.content, source=msg.source))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported message type: {msg.type}")
    
    agent = get_agent()

    try:
        response = await agent.on_messages(py_msgs, CancellationToken())
    except Exception as e:
        return InvokeResult(
            status="fail",
            response=response,
            elapsed={"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)},
        )
    return InvokeResult(
        status="ok",
        response=response,
        elapsed={"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)},
    )