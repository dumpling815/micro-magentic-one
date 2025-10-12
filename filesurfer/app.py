from fastapi import FastAPI, Body
import time, os
from common.request_schema import InvokeBody, InvokeResult

# == AutoGen imports ==
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.base import Response
from autogen_core import CancellationToken # Supports task cancellation while async processing


app = FastAPI(title="Magentic-One File Surfer")


# == env ==
FILESURFER_ROOT = os.getenv("FILESURFER_ROOT") # To control the file system access for filesurfer agent
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT"))


# == Lazy Singleton ==
_client = None
_agent = None
def get_agent() -> FileSurfer:
    global _client, _agent
    if _agent is None:
        if _client is None:
            _client = OllamaChatCompletionClient(model=OLLAMA_MODEL, host=OLLAMA_HOST, timeout=TIMEOUT)
        _agent = FileSurfer(name="filesurfer", model_client=_client, base_path=FILESURFER_ROOT)
    return _agent

@app.get("/health")
def health_check():
    return {"status": "ok", "model": OLLAMA_MODEL, "ollama_host": OLLAMA_HOST, "safe_root": FILESURFER_ROOT}

@app.post("/invoke", response_model=InvokeResult)
async def invoke(body: InvokeBody = Body(...)):
    start_time_perf = time.perf_counter()
    # JSON -> AutoGen message objects
    # py_msgs = []

    # for msg in body.messages:
    #     if msg.type == "TextMessage":
    #         py_msgs.append(TextMessage(content=msg.content, source=msg.source))
    #     else:
    #         raise HTTPException(status_code=400, detail=f"Unsupported message type: {msg.type}")
    
    agent = get_agent() # FileSurfer agent
    try:
        response: Response = await getattr(agent,body.method)(body.messages, CancellationToken())
        #response = await agent.on_messages(py_msgs, CancellationToken())
    except Exception as e:
        print(f"Exception occured: {e}")
        return InvokeResult(
            status="fail",
            response = None,
            elapsed={"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)},
        )
    return InvokeResult(
        status="ok",
        response = response,
        elapsed={"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)},
    )