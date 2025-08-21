# 원래 Agentic AI의 목적에 따르면, Agent가 구성한 코드를 유저의 컴퓨터에서 실행해야함.
# 그러나, 이 실험에서 Agent가 생성한 코드가 시스템에 어떤 영향을 끼칠지 알 수 없으므로, 유저의 컴퓨터를 컨테이너로 가정.
# 따라서, Agent가 생성한 코드를 실행하는 ComputerTerminal agent는 유저 컨테이너 안에서 실행됨.
# 즉, user 디렉토리의 app.py는 최초 유저의 요청을 발생시키는 역할과 동시에, ComputerTerminal agent의 역할도 수행함.

from fastappi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
import time, os, httpx
from RequestSchema import InvokeBody, InvokeResult
from ..orchestrator import Plan, FinalResult

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken  # Supports task cancellation while async processing
from autogen_ext.code_executors.local import LocalCommandLineExecutor
# 해당 모듈은 local이긴 하지만, user의 컴퓨터 자체를 컨테이너로 띄우기 때문에 실험 환경 시스템 안정성을 해치지 않음.

app = FastAPI(title="Magentic-One User")

# --- Env ---
WORKDIR = os.getenv("WORKDIR", "/workspace")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))
ALLOW_CMDS = os.getenv("ALLOW_CMDS") # e.g. python, pip, ls, cat, cd
DENY_CMDS = os.getenv("DENY_CMDS") # e.g. rm, sudo

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000/invoke")

# --- Lazy Singleton ---
# user 컴퓨터에서는 직접적으로 LLM을 사용하지 않음
# 오로지 요청을 전달하고, 받은 응답을 ComputerTerminal이 수행하는 것. (ComputerTerminal은 LLM 사용하지 않음)
_agent = None
def get_agent() -> LocalCommandLineExecutor:
    global _agent
    if _agent is None:
        _agent = LocalCommandLineExecutor(
            name="user",
            workdir=WORKDIR,
            allow_cmds=[cmd.strip() for cmd in ALLOW_CMDS.split(",")] if ALLOW_CMDS else None,
            deny_cmds=[cmd.strip() for cmd in DENY_CMDS.split(",")] if DENY_CMDS else None,
        )
    return _agent


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status":"ok","workdir":WORKDIR,"allow":ALLOW_CMDS,"deny":DENY_CMDS}

@app.get("/ready")
def ready():
    try:
        _ = get_agent()
        return {"ready": True}
    except Exception as e:
        raise HTTPException(503, f"not ready: {e}")
    
@app.post("/invoke", response_model=InvokeResult)
async def invoke(body: InvokeBody = Body(...)):
    start_time_perf = time.perf_counter()
    py_msgs = []
    
    for m in body.messages:
        if m.type != "TextMessage":
            raise HTTPException(status_code=400, detail=f"Unsupported message type: {m.type}")
        py_msgs.append(TextMessage(content=m.content, source=m.source))
    
    agent = get_agent() # LocalCommandLineExecutor agent
    
    try:
        response = await agent.on_messages(py_msgs, CancellationToken())
        chat_msg = getattr(response, "chat_message", None)
        content = getattr(chat_msg, "content", "") if chat_msg else str(response)
        return {
            "status": "ok",
            "message": {"type": "TextMessage", "source": "ComputerTerminal", "content": content},
            "elapsed": {"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)},
        }
    except Exception as e:
        return {
            "status": "fail",
            "message": None,
            "elapsed": {"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)},
        }
    

# --- Request to Orchestrator ---
@app.post("/start", response_model=FinalResult)
async def start(plan: Plan = Body(...)):
    """
    Start the orchestrator with the given plan.
    """
    start_time_perf = time.perf_counter()
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.post(
                ORCHESTRATOR_URL,
                json=plan.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
            return FinalResult(**result)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))