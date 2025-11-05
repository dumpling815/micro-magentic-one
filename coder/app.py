from fastapi import FastAPI, Body, HTTPException
import time, os, logging, sys
from common.request_schema import InvokeBody, InvokeResult

app = FastAPI(title="Magentic-One Coder Agent")

# --- AutoGen imports ---
from autogen_agentchat.messages import TextMessage
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core import CancellationToken  # Supports task cancellation while async processing
from autogen_agentchat.base import Response

# --- Env ---
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")  # e.g. openai, ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
REQUEST_TIMEOUT  = float(os.getenv("REQUEST_TIMEOUT"))

logger = logging.getLogger("coder")

# File handler
file_handler = logging.FileHandler("coder.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Console handler (optional but useful in k8s)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Common formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Make sure uvicorn's root logger doesn't swallow/override these
#logger.propagate = False


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

def write_log(message: str):
    logger.info(message)

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
    # py_msgs =[]
    # for m in body.messages:
    #     if m.type != "TextMessage": # 이후에 ChatMessage로 변경하여 더 다양한 메시지 타입 지원.
    #         raise HTTPException(status_code=400, detail=f"Unsupported message type: {m.type}")
    #     py_msgs.append(TextMessage(content=m.content, source=m.source))
    msgs = body.messages or []
    logger.info(f"Invoke called with method: {body.method}, messages count: {len(msgs)}")
    coder = get_coder()
    logger.info(f"Coder instance: {coder}")

    if body.method == "on_reset":
        try:
            await coder.on_reset(CancellationToken())
        except Exception as e:
            logger.exception(f"Exception while reset coder: {e}")
            # on_reset은 return 없음.
        logger.info("Coder reset ok.")
        return InvokeResult(
            status="ok", 
            response={
                "chat_message": TextMessage(source="coder", content="reset ok")
            }, 
            elapsed={"execution_latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
        )
    else:
        try:
            # 다른 메소드들은 messages 인자 안받는 경우도 있지만, 현재는 무시. TODO: 필요 인자 다른 메소드마다 로직 분기 필요.
            response: Response = await getattr(coder,body.method)(body.messages,CancellationToken())
            #response: Response = await coder.on_messages(py_msgs, CancellationToken())
        except Exception as e:
            logger.exception(f"Exception occured: {e}")
            return InvokeResult(
                status="fail", 
                response={
                    "chat_message": TextMessage(source="coder", content=f"coder Exception: {e}")
                }, 
                elapsed={"code_generation_latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
            )
        write_log(f"Coder invoke {body.method} completed.")
        write_log(f"Response chat_message: {response.chat_message}")
        return InvokeResult(
            status="ok", 
            response={"chat_message":response.chat_message,"inner_messages":response.inner_messages}, 
            elapsed={"code_generation_latency_ms": int((time.perf_counter() - start_time_perf) * 1000)}
        )