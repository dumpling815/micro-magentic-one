from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
import time, os, httpx, asyncio
from RequestSchema import Msg, InvokeBody

# --- AutoGen imports ---
from autogen_agentchat.messages import TextMessage, ChatMessage
from autogen_core import CancellationToken  # Supports task cancellation while async processing
try:
    from autogen_agentchat.teams._group_chat.magentic_one._magentic_one_orchestrator import MagenticOneOrchestrator
except Exception:
    from autogen_agentchat.teams.magentic_one import MagenticOneOrchestrator # 리팩토링 되었을 경우
from autogen_ext.models.ollama import OllamaChatCompletionClient


app = FastAPI(title="Magentic-One Orchestrator")


# --- Env ---
URL_FILESURFER       = os.getenv("URL_FILESURFER", "http://filesurfer:8000")
URL_WEBSURFER        = os.getenv("URL_WEBSURFER", "http://websurfer:8000")
URL_CODER            = os.getenv("URL_CODER", "http://coder:8000")
URL_COMPUTERTERMINAL             = os.getenv("URL_User", "http://computerterminal:8000")

MODEL_PROVIDER       = os.getenv("MODEL_PROVIDER", "ollama")  # e.g. ollama, openai, etc.
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_HOST          = os.getenv("OLLAMA_HOST", "http://ollama:11434")  

REQUEST_TIMEOUT      = float(os.getenv("REQUEST_TIMEOUT", "30"))
RETRIES              = int(os.getenv("RETRIES", "1"))             # 실패 시 추가 재시도 횟수
MAX_STEPS            = int(os.getenv("MAX_STEPS", "8"))              # 최대 스텝 수    

AUTH_TOKEN           = os.getenv("AUTH_TOKEN")  # 서비스 공통 토큰이 잇다면 Authorization 헤더로 전파

APPEND_PREVIOUS      = os.getenv("APPEND_PREVIOUS", "true").lower() == "true"  # 이전 스텝 응답 메시지를 다음 입력에 누적
SERVICE_ENDPOINTS: dict[str, str] = {
    "websurfer": URL_WEBSURFER,
    "filesurfer": URL_FILESURFER,
    "coder": URL_CODER,
    "computerterminal": URL_COMPUTERTERMINAL
}       

class Plan(BaseModel):
    route: list[Literal["websurfer","filesurfer","coder","computerterminal"]] = Field(
        ..., example=["websurfer","filesurfer","coder","computerterminal"]
    )
    input: InvokeBody

class Step(BaseModel):
    service: Literal["websurfer","filesurfer","coder","computerterminal"]
    status: Literal["ok","fail"]
    message: Optional[Msg] = None
    latency_ms: int

class FinalResult(BaseModel):
    status: Literal["ok","fail"]
    steps: list[Step]
    total_latency_ms: int

# --- HTTP Participant ---
class HttpParticipant:
    """
    AutoGen 참가자처럼 동작하는 경량 래퍼
    - on_messages(messages, Cancellation Token) 인터페이스 제공
    - 내부에서 마이크로서비스의 /invoke HTTP 엔드포인터 호출
    """
    def __init__(self, name: str, endpoint: str, timeout: float = REQUEST_TIMEOUT):
        self.name = name
        self.endpoint = endpoint
        self.timeout = timeout
    async def on_messages(self, messages: list[ChatMessage], cacellation_token: Optional[CancellationToken] = None) -> ChatMessage:
        # ChatMessage를 custom contract 형식인 Msg로 직렬화
        payload = {
            "messages": [
                {
                    "type": "TextMessage",
                    "source": (m.source or "orchestrator") if hasattr(m, "source") else "orchestrator",
                    "content": getattr(m, "content", ""),
                }
                for m in messages
            ],
            "options": {}
        }
        headers = {}
        if AUTH_TOKEN:
            headers["Authorization"] = AUTH_TOKEN
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.endpoint + "/invoke", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
        
        msg = result.get("message") or {"content": ""}
        return TextMessage(content=msg.get("content",""), source=self.name)

# --- Lazy Singleton ---
_client = None
_agent = None
def get_agent() -> MagenticOneOrchestrator:
    global _client, _agent
    if _agent is None:
        if _client is None:
            _client = OllamaChatCompletionClient(
                model=OLLAMA_MODEL,
                host=OLLAMA_HOST,
                timeout=REQUEST_TIMEOUT,
                retries=RETRIES
            )
        participants = {
            "filesurfer": HttpParticipant("filesurfer", SERVICE_ENDPOINTS["filesurfer"]),
            "websurfer": HttpParticipant("websurfer", SERVICE_ENDPOINTS["websurfer"]),
            "coder": HttpParticipant("coder", SERVICE_ENDPOINTS["coder"]),
            "computerterminal": HttpParticipant("computerterminal", SERVICE_ENDPOINTS["computerterminal"]),
        }
        _agent = MagenticOneOrchestrator(
            name =  "Magentic-One Orchestrator",
            model_client = _client,
            participant_names = list(participants.keys()),
        )
    return _agent

async def call_service(url: str, payload: dict) -> dict:
    headers = {}
    if AUTH_TOKEN:
        headers["Authorization"] = AUTH_TOKEN
    
    last_exc = None
    for attempt in range(RETRIES + 1):
        start_time_perf = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            return {"status": response.get("status", "fail"), "message": response.get("message"), "latency_ms": int((time.perf_counter()-start_time_perf)*1000)}
        except Exception as e:
            last_exc = e
            await asyncio.sleep(0.2 * (attempt + 1))
    raise HTTPException(502, f"call failed: {url} ({last_exc})")

# --- Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "ok",
        "endpoints": SERVICE_ENDPOINTS,
        "orchestrator": {"provider": "MODEL_PROVIDER", "model": OLLAMA_MODEL, "host": OLLAMA_HOST},    
    }

@app.get("/ready")
def ready():
    try:
        _ = get_agent()
        return {"ready": True}
    except Exception as e:
        raise HTTPException(503, f"not ready: {e}")

class OrchestrateInput(BaseModel):
    input: InvokeBody

@app.post("/invoke", response_model=FinalResult)
async def orchestrate(body: OrchestrateInput = Body(...)):
    """
    Magentic-One Orchestrator    """
    start_time_perf = time.perf_counter()
    plan = body.input
    steps = []
    
    orchestrator = get_agent()

    msgs: list[ChatMessage] = [
        TextMessage(content=msg.content, source=msg.source) for msg in body.input.messages
    ]
    steps: list[Step] = []
    ct = CancellationToken()

    for _ in range(MAX_STEPS):
        # Get the next service in the route
        response = await orchestrator.on_messages(msgs, ct)
        content = getattr(response, "content", "")
        source = getattr(response, "source", "orchestrator")

        msg = Msg(type="TextMessage", source=source, content=content)
        msgs.append(msg)

        # 간단한 종료 휴리스틱: orchestrator가 'stop' 신호를 content에 담는 경우
        # (실제 구현체에 맞춰 종료 조건/메타 신호를 파싱하도록 수정 가능)
        if isinstance(content, str) and content.strip().lower().startswith("[done]"):
            break

        steps.append(Step(service=source if source in SERVICE_ENDPOINTS else "unknown", status="ok", message=msg))

        if len(steps) >= MAX_STEPS:
            break
    
    overall = "ok" if steps else "fail"
    return FinalResult(status=overall, steps=steps, total_latency_ms=int((time.perf_counter() - start_time_perf) * 1000))
