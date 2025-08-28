# 원래 Agentic AI의 목적에 따르면, Agent가 구성한 코드를 유저의 컴퓨터에서 실행해야함.
# 그러나, 이 실험에서 Agent가 생성한 코드가 시스템에 어떤 영향을 끼칠지 알 수 없으므로, 유저의 컴퓨터를 컨테이너로 가정.
# 따라서, Agent가 생성한 코드를 실행하는 ComputerTerminal agent는 유저 컨테이너 안에서 실행됨.
# 즉, user 디렉토리의 app.py는 최초 유저의 요청을 발생시키는 역할과 동시에, ComputerTerminal agent의 역할도 수행함.

from fastapi import FastAPI, HTTPException, Body
import time, os, httpx
from RequestSchema import InvokeBody, InvokeResult, Msg
from ..orchestrator import FinalResult

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken  # Supports task cancellation while async processing
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
# 해당 모듈은 local이긴 하지만, user의 컴퓨터 자체를 컨테이너로 띄우기 때문에 실험 환경 시스템 안정성을 해치지 않음.

app = FastAPI(title="Magentic-One User")

# --- Env ---
WORKDIR = os.getenv("WORKDIR", "/workspace")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
ALLOW_CMDS = os.getenv("ALLOW_CMDS") # e.g. python, pip, ls, cat, cd
DENY_CMDS = os.getenv("DENY_CMDS") # e.g. rm, sudo

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")

# --- Lazy Singleton ---
# user 컴퓨터에서는 직접적으로 LLM을 사용하지 않음
# 오로지 요청을 전달하고, 받은 응답을 ComputerTerminal이 수행하는 것. (ComputerTerminal은 LLM 사용하지 않음)
_executor = None
def get_executor() -> LocalCommandLineCodeExecutor:
    global _executor
    if _executor is None:
        _executor = LocalCommandLineCodeExecutor(
            timeout = REQUEST_TIMEOUT,
            work_dir=WORKDIR,
            allow_cmds=[cmd.strip() for cmd in ALLOW_CMDS.split(",")] if ALLOW_CMDS else None,
            deny_cmds=[cmd.strip() for cmd in DENY_CMDS.split(",")] if DENY_CMDS else None,
        )
    return _executor


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status":"ok","workdir":WORKDIR,"allow":ALLOW_CMDS,"deny":DENY_CMDS}

@app.get("/ready")
def ready():
    try:
        _ = get_executor()
        return {"ready": True}
    except Exception as e:
        raise HTTPException(503, f"not ready: {e}")
    
@app.post("/invoke", response_model=InvokeResult)
async def invoke(body: InvokeBody = Body(...)):
    """
    Handle incoming messages, process them using LocalCommandLineExecutor, and return the response.
    """
    start_time_perf = time.perf_counter()
    code_blocks = []
    
    for m in body.messages:
        if m.type != "TextMessage":
            raise HTTPException(status_code=400, detail=f"Unsupported message type: {m.type}")
        code_blocks.append(TextMessage(content=m.content, source=m.source))
    
    executor = get_executor() # LocalCommandLineExecutor agent
    # LocalCommandLineExecutor는 Code Block을 포함한 Message를 받아서 해당 코드를 Local 환경에서 실행 후
    # Execution output을 포함한 Message를 반환함.
    # https://microsoft.github.io/autogen/0.2/docs/tutorial/code-executors/ 참조.
    
    try:
        code_result = await executor.execute_code_blocks(code_blocks=code_blocks, cancellation_token=CancellationToken())
        return InvokeResult(
            status="ok", 
            message=Msg(type="TextMessage", source="computerterminal", content=code_result), 
            elapsed={"Code Execution latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
        )
    except Exception as e:
        return InvokeResult(
            status="fail", 
            message=Msg(type="TextMessage", source="computerterminal", content=code_result), 
            elapsed={"Code Execution latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
        )
    

# --- Request to Orchestrator ---
@app.post("/start", response_model=FinalResult)
async def start(msg: Msg = Body(...)):
    """
    Start the orchestrator.
    Input will be a message from the user, which will be used to create a plan at orchestrator.
    The orchestrator will handle the rest of the process and return the final result.
    """
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            body = InvokeBody(messages=[msg], options={})
            response = await client.post(
                ORCHESTRATOR_URL+"/invoke",
                json=body.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
            return FinalResult(**result)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))