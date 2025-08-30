from fastapi import FastAPI, Body, HTTPException
import time, os
from RequestSchema import InvokeBody, InvokeResult, Msg

app = FastAPI(title="Magentic-One Coder Agent")

# --- AutoGen imports ---
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core import CancellationToken  # Supports task cancellation while async processing
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import Response

# --- Env ---
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")  # e.g. openai, ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
REQUEST_TIMEOUT  = float(os.getenv("REQUEST_TIMEOUT", "30"))


# --- Lazy Singleton ---
_client = None
_coder = None
def get_coder() -> MagenticOneCoderAgent:
    global _client, _coder
    if _coder is None:
        if _client is None:
            if MODEL_PROVIDER == "ollama":
                _client = OllamaChatCompletionClient(model=OLLAMA_MODEL, host=OLLAMA_HOST, timeout=REQUEST_TIMEOUT)
            else:
                raise RuntimeError(f"Unsupported model provider: {MODEL_PROVIDER}")
        _coder = MagenticOneCoderAgent(name="coder", model_client=_client) # Description과 System message는 해당 클래스에 구현되어 있음.
    return _coder

# --- Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": MODEL_PROVIDER,
        "ollama_model": OLLAMA_MODEL,
        "ollama_host": OLLAMA_HOST,
        "request_timeout": REQUEST_TIMEOUT
    }

@app.get("/ready")
def ready():
    try:
        _ = get_coder()
        return {"ready": True}
    except Exception as e:
        raise HTTPException(503, f"not ready: {str(e)}")

@app.post("/invoke", response_model=InvokeResult)
async def invoke(body: InvokeBody = Body(...)):
    start_time_perf = time.perf_counter()
    py_msgs =[]
    for m in body.messages:
        if m.type != "TextMessage": # 이후에 ChatMessage로 변경하여 더 다양한 메시지 타입 지원.
            raise HTTPException(status_code=400, detail=f"Unsupported message type: {m.type}")
        py_msgs.append(TextMessage(content=m.content, source=m.source))
    
    coder = get_coder()

    try:
        response: Response = await coder.on_messages(py_msgs, CancellationToken())
        if not isinstance(response, Response):
            raise ValueError("Coder agent did not return a valid Response object.")
        
        return InvokeResult(
            status="ok", 
            response=response, 
            elapsed={"code_generation_latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
        )
    except Exception as e:
        return InvokeResult(
            status="fail", 
            response=response, 
            elapsed={"code_generation_latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
        )