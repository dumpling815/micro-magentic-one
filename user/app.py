# 원래 Agentic AI의 목적에 따르면, Agent가 구성한 코드를 유저의 컴퓨터에서 실행해야함.
# 그러나, 이 실험에서 Agent가 생성한 코드가 시스템에 어떤 영향을 끼칠지 알 수 없으므로, 유저의 컴퓨터를 컨테이너로 가정.
# 따라서, Agent가 생성한 코드를 실행하는 ComputerTerminal agent는 유저 컨테이너 안에서 실행됨.
# 즉, user 디렉토리의 app.py는 최초 유저의 요청을 발생시키는 역할과 동시에, ComputerTerminal agent의 역할도 수행함.

from fastapi import FastAPI, HTTPException, Body
import time, os, httpx
from common.request_schema import InvokeBody, InvokeResult

from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken  # Supports task cancellation while async processing
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
# 해당 모듈은 local이긴 하지만, user의 컴퓨터 자체를 컨테이너로 띄우기 때문에 실험 환경 시스템 안정성을 해치지 않음.

app = FastAPI(title="Magentic-One User")

# --- Env ---
EXECUTER_WORKDIR = os.getenv("WORKDIR")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT"))
CLEANUP_TEMP_FILES = os.getenv("CLEANUP_TEMP_FILES").lower() == "true"
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL")

# --- Lazy Singleton ---
# user 컴퓨터에서는 직접적으로 LLM을 사용하지 않음
# 오로지 요청을 전달하고, 받은 응답을 ComputerTerminal이 수행하는 것. (ComputerTerminal은 LLM 사용하지 않음)
_executor = None
def get_executor() -> LocalCommandLineCodeExecutor:
    global _executor
    if _executor is None:
        _executor = LocalCommandLineCodeExecutor(
            timeout = REQUEST_TIMEOUT,
            work_dir=EXECUTER_WORKDIR,
            cleanup_temp_files=CLEANUP_TEMP_FILES, # Set to False for debugging purposes
        )
    return _executor

# 파싱 부분을 찾지 못했을 때, 코드 블록 파싱을 직접 구현해봄.
# CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"
# def code_block_parser(msg: TextMessage) -> CodeBlock:
#     # This function gets the code block from the response message and return the pure python code.
#     # Leverages Regular Expression.
#     regex = re.compile(CODE_BLOCK_PATTERN, re.DOTALL)
#     text = msg.content
#     matches = regex.search(text)
#     if matches:
#         language = (matches.group(1) or "unknown").lower()
#         code = matches.group(2).strip("\n")
#         return CodeBlock(code=code, language=language)
#     else:
#         raise ValueError("No code block found in the message.")


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/ready")
def ready():
    try:
        _ = get_executor()
        return {"ready": True}
    except Exception as e:
        raise HTTPException(503, f"not ready: {e}")
    
@app.post("/invoke", response_model=InvokeResult)
async def execute(body: InvokeBody = Body(...)):
    """
    Handle incoming messages, process them using LocalCommandLineExecutor, and return the response.
    """
    start_time_perf = time.perf_counter()
    
    #py_msgs =[]
    # for m in body.messages:
    #     if m.type != "TextMessage":
    #         raise HTTPException(status_code=400, detail=f"Unsupported message type: {m.type}")
    #     py_msgs.append(TextMessage(content=m.content, source=m.source))
    
    executor = get_executor() # LocalCommandLineExecutor agent
    # LocalCommandLineExecutor는 Code Block을 포함한 Message를 받아서 해당 코드를 Local 환경에서 실행 후
    # Execution output을 포함한 Message를 반환함.
    # https://microsoft.github.io/autogen/0.2/docs/tutorial/code-executors/ 참조.
    
    try:
        response: Response = await getattr(executor,body.method)(body.messages,CancellationToken())
        #response: Response = await executor.on_messages(py_msgs, CancellationToken())
        # code_result = await executor.execute_code_blocks(code_blocks=code_blocks, cancellation_token=CancellationToken())
        # code_result는 CommandLineCodeResult 클래스 : {exit_code: int, output: str} 형태.
        # code executor의 on_messages() 메서드는 메시지에서 코드 블록을 추출하는 함수 extract_code_blocks_from_messages()와
        # execute_code_blocks() 메서드를 내부적으로 호출함.
    except Exception as e:
        print(f"Exception occured: {e}")
        return InvokeResult(
            status="fail", 
            response=None, 
            elapsed={"execution_latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
        )
    return InvokeResult(
        status="ok", 
        response=response,
        elapsed={"execution_latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
    )
    

# --- Request to Orchestrator ---
@app.post("/start", response_model=InvokeResult)
async def run(request: dict = Body(...)):
    """
    Run the orchestrator(Entire Magentic-One System).
    The orchestrator will handle the rest of the process and return the final result.
    """
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        query = request.get("query")
        if not query:
            raise HTTPException(400, "Missing 'query' field")

        user_msg = TextMessage(source="user", content=query)
        body = InvokeBody(method="run", messages=[user_msg])

        try:
            #body = InvokeBody([msg], options={})
            invoke_result: httpx.Response = await client.post(
                ORCHESTRATOR_URL+"/invoke",
                json=body.model_dump(),
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        # Deserialization
        invoke_result = invoke_result.json()
        invoke_result = InvokeResult(**invoke_result)

        return invoke_result