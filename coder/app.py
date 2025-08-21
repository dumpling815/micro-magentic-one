from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Literal
import time, os
from RequestSchema import InvokeBody, InvokeResult

app = FastAPI(title="Magentic-One Coder Agent")

# --- AutoGen imports ---
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core import CancellationToken  # Supports task cancellation while async processing
from autogen_agentchat.messages import TextMessage

# --- Env ---
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")  # e.g. openai, ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
REQUEST_TIMEOUT  = float(os.getenv("REQUEST_TIMEOUT", "30"))


# --- Lazy Singleton ---
_client = None
_agent = None
def get_agent() -> MagenticOneCoderAgent:
    global _client, _agent
    if _agent is None:
        if _client is None:
            if MODEL_PROVIDER == "ollama":
                _client = OllamaChatCompletionClient(model=OLLAMA_MODEL, host=OLLAMA_HOST, timeout=REQUEST_TIMEOUT)
            else:
                raise RuntimeError(f"Unsupported model provider: {MODEL_PROVIDER}")
        _agent = MagenticOneCoderAgent(name="coder", model_client=_client)
    return _agent

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
        _ = get_agent()
        return {"ready": True}
    except Exception as e:
        raise HTTPException(503, f"not ready: {str(e)}")

@app.post("/invoke", response_model=InvokeResult)
async def invoke(body: InvokeBody = Body(...)):
    start_time_perf = time.perf_counter()
    py_msgs =[]
    for m in body.messages:
        if m.type != "TextMessage":
            raise HTTPException(status_code=400, detail=f"Unsupported message type: {m.type}")
        py_msgs.append(TextMessage(content=m.content, source=m.source))
    
    agent = get_agent()

    try:
        response = await agent.on_messages(py_msgs, CancellationToken())
        chat_msg = getattr(response, "chat_message", None)
        content = getattr(chat_msg, "content", "") if chat_msg else str(response)
        return {
            "status": "ok",
            "message": {"type": "TextMessage", "source": "coder", "content": content},
            "elapsed": {"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)},
        }
    except Exception as e:
        return {
            "status": "fail",
            "message": None,
            "elapsed": {"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
        }