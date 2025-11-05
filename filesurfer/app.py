from fastapi import FastAPI, Body
import time, os, logging
from common.request_schema import InvokeBody, InvokeResult

# == AutoGen imports ==
from autogen_agentchat.messages import TextMessage
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

logger = logging.getLogger("filesurfer")
logging.basicConfig(
    filename='filesurfer.log', # StreamHandler 대신 FileHandler 사용 기본 모드 'a'
    encoding='utf-8', 
    level=logging.INFO, # INFO 레벨부터 출력될 수 있도록 설정.(기본 수준 이상의 로그만 출력)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    ) 


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
    
    logger.info(f"FileSurfer invoke called. method: {body.method}, num_messages: {len(body.messages)}")
    agent = get_agent() # FileSurfer agent
    logger.info(f"Coder instance: {agent}")

    
    if body.method == "on_reset":
        try:
            await agent.on_reset(CancellationToken())
        except Exception as e:
            logger.exception(f"Exception while reset filesurfer: {e}")
            # on_reset은 return 없음.
        logger.info("FileSurfer reset ok.")
        return InvokeResult(
            status="ok", 
            response={
                "chat_message": TextMessage(source="filesurfer", content="reset ok")
            }, 
            elapsed={"execution_latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
        )
    else:
        try:
            # 다른 메소드들은 messages 인자 안받는 경우도 있지만, 현재는 무시. TODO: 필요 인자 다른 메소드마다 로직 분기 필요.
            response: Response = await getattr(agent,body.method)(body.messages, CancellationToken())
            #response = await agent.on_messages(py_msgs, CancellationToken())
        except Exception as e:
            logger.exception(f"Exception occured: {e}")
            return InvokeResult(
                status="fail",
                response={
                    "chat_message": TextMessage(source="filesurfer", content=f"FileSurfer Exception: {e}")
                },
                elapsed={"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)},
            )
        logger.info(f"FileSurfer invoke {body.method} completed.")
        logger.info(f"Response chat_message: {response.chat_message}")
        return InvokeResult(
            status="ok",
            response = {"chat_message":response.chat_message,"inner_messages":response.inner_messages},
            elapsed={"latency_ms": int((time.perf_counter() - start_time_perf) * 1000)},
        )