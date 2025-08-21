from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Literal
import time, os
from RequestSchema import InvokeBody, InvokeResult

# == AutoGen imports ==
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken # Supports task cancellation while async processing


app = FastAPI(title="Magentic-One File Surfer")


# == env ==
SAFE_ROOT = os.getenv("SAFE_ROOT", "/data") # To control the file system access for filesurfer agent
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL = os.getenv("MODEL", "gpt-oss:20b")
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "20"))


# == Lazy Singleton ==
_client = None
_agent = None
def get_agent() -> FileSurfer:
    global _client, _agent
    if _agent is None:
        if _client is None:
            _client = OllamaChatCompletionClient(model=MODEL, host=OLLAMA_HOST, timeout=TIMEOUT)
        _agent = FileSurfer(name="filesurfer", model_client=_client, base_path=SAFE_ROOT)
    return _agent

@app.get("/health")
def health_check():
    return {"status": "ok", "model": MODEL, "ollama_host": OLLAMA_HOST, "safe_root": SAFE_ROOT}

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
    
    agent = get_agent() # FileSurfer agent
    try:
        response = await agent.on_messages(py_msgs, CancellationToken())
    except Exception as e:
        return {"status":"fail",
                "message": None,
                "elapsed":{"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
        }
    
    chat_msg = getattr(response, "chat_message", None)
    content = getattr(chat_msg, "content", "") if chat_msg else str(response)

    return {
        "status": "ok",
        "message": {"type": "TextMessage", "source":"filesurfer", "content": content},
        "elapsed": {"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)},
    }