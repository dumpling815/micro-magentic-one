from fastapi import FastAPI, Body, HTTPException
from typing import Sequence
import time, os, httpx
from common.request_schema import InvokeBody, InvokeResult, Msg

# --- AutoGen imports ---
from autogen_agentchat.messages import TextMessage, ChatMessage # ChatMessage는 TextMessage를 포함하는 Uninon. (다양한 메시지 타입 지원을 위해)
from autogen_core import CancellationToken  # Supports task cancellation while async processing
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat 
from autogen_agentchat.base import Response, TaskResult
from autogen_ext.models.ollama import OllamaChatCompletionClient


app = FastAPI(title="Magentic-One Orchestrator")


# --- Env ---
FILESURFER_URL       = os.getenv("FILESURFER_URL")
WEBSURFER_URL        = os.getenv("WEBSURFER_URL")
CODER_URL            = os.getenv("CODER_URL")
COMPUTERTERMINAL_URL = os.getenv("COMPUTERTERMINAL_URL")

MODEL_PROVIDER       = os.getenv("MODEL_PROVIDER")  # e.g. ollama, openai, etc.
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL")
OLLAMA_HOST          = os.getenv("OLLAMA_HOST")  

REQUEST_TIMEOUT      = int(os.getenv("REQUEST_TIMEOUT"))
RETRIES              = int(os.getenv("RETRIES"))             # 실패 시 추가 재시도 횟수
MAX_STEPS            = int(os.getenv("MAX_STEPS"))              # 최대 스텝 수    

AUTH_TOKEN           = os.getenv("AUTH_TOKEN")  # 서비스 공통 토큰이 잇다면 Authorization 헤더로 전파

APPEND_PREVIOUS      = os.getenv("APPEND_PREVIOUS").lower() == "true"  # 이전 스텝 응답 메시지를 다음 입력에 누적

SERVICE_ENDPOINTS: dict[str, str] = {
    "websurfer": WEBSURFER_URL,
    "filesurfer": FILESURFER_URL,
    "coder": CODER_URL,
    "computerterminal": COMPUTERTERMINAL_URL
}       

# --- Wrapper for Agent to leverage http ---
class HttpChatAgent(AssistantAgent): #
    """
    AutoGen 참가자처럼 동작하는 경량 래퍼
    - on_messages(messages, Cancellation Token) 인터페이스 제공
    - 내부에서 마이크로서비스의 /invoke HTTP 엔드포인터 호출
    - AssistantAgent에 구현되어있는 여러 메소드들 중 현재는 on_messages만 구현
    """
    def __init__(self, name: str, endpoint: str, timeout: float = REQUEST_TIMEOUT):
        self.name = name
        self.endpoint = endpoint
        self.timeout = timeout

    def name(self) -> str:
        return self.name
    
    def endpoint(self) -> str:
        return self.endpoint
    
    def description(self) -> str:
        return f"HTTP-based Assistant Agent for {self.name} at {self.endpoint}"
    
    async def on_messages(
            self,
            messages: Sequence[ChatMessage],
        ) -> Response:
        # TODO: 각 Agent들에 구현되어 있는 세부 latency 측정을 저장하는 부분은 구현 예정
        messages = [Msg(type="TextMessage", source="orchestrator", content= message) for message in messages]
        payload = InvokeBody(
            messages = messages,
            options = {}
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            invoke_result: httpx.Response = await client.post(self.endpoint + "/invoke", json=payload.model_dump())
            invoke_result.raise_for_status()
        
        # Deserialization
        invoke_result = invoke_result.json()
        invoke_result = InvokeResult(**invoke_result)

        response = invoke_result.response

        if not isinstance(response, Response):
            raise ValueError(f"Agent {self.name} did not return a valid Response object.")
        elif invoke_result.status != "ok":
            raise ValueError(f"Agent {self.name} returned an error status: {invoke_result.status}")
        return response

# --- Lazy Singleton ---
_client = None
_agent = None
def get_agent() -> MagenticOneGroupChat:
    # Magentic-One의 구성에 따르면 get_team()이 더 적절한 함수명 일 수도 있으나, 
    # 마이크로서비스 관점에서 orchestrator가 단일 서비스이면서 팀 전체를 관리하는 역할이므로 get_agent()로 명명 (이후 수정 가능)
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
            "filesurfer": HttpChatAgent("filesurfer", SERVICE_ENDPOINTS["filesurfer"]),
            #"websurfer": HttpChatAgent("websurfer", SERVICE_ENDPOINTS["websurfer"]), #WebSurfer는 기본적으로 브라우저와의 상호작용 필요하기 때문에 llm 모델이 이를 지원하지 않는 경우 사용 불가.
            "coder": HttpChatAgent("coder", SERVICE_ENDPOINTS["coder"]),
            "computerterminal": HttpChatAgent("computerterminal", SERVICE_ENDPOINTS["computerterminal"]),
        }
        _agent = MagenticOneGroupChat(
            name =  "Micro Magentic-One Orchestrator",
            model_client = _client,
            participant = list(participants.values()), # participant는 Magentic-One을 구성하는 하위 에이전트들의 목록.
        )
    return _agent

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

@app.post("/invoke", response_model=InvokeResult)
async def orchestrate(body: InvokeBody = Body(...)):
    """
    Magentic-One Orchestrator    
    """
    start_time_perf = time.perf_counter()
     # 현재는 messages의 첫 번째 메시지만 작업으로 간주. 향후 다중 메시지 지원 가능.
    task:str = body.messages[0].content if body.messages else "No task provided"
    orchestrator = get_agent()
    
    # task는 str, TextMessage, ChatMessage 모두 가능.
    result: TaskResult = await orchestrator.run(task=task, cancellation_token=CancellationToken())
    finish_time_perf = time.perf_counter()
    return InvokeResult(
        status="ok" if result.status == "completed" else "fail",
        response=result,
        elapsed={"orchestration_latency_ms": int((finish_time_perf-start_time_perf)*1000)}
    )
    