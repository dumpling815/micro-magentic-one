from fastapi import FastAPI, Body, HTTPException
from typing import Sequence, Any
import time, os, httpx
from common.request_schema import InvokeBody, InvokeResult

# --- AutoGen imports ---
from autogen_agentchat.messages import TextMessage, BaseChatMessage # ChatMessage는 TextMessage를 포함하는 Union. (다양한 메시지 타입 지원을 위해)
from autogen_core import CancellationToken  # Supports task cancellation while async processing
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.teams import MagenticOneGroupChat 
from autogen_agentchat.base import Response, TaskResult
from autogen_ext.models.ollama import OllamaChatCompletionClient


app = FastAPI(title="Magentic-One Orchestrator")


# --- Env ---
FILESURFER_URL       = os.getenv("FILESURFER_URL")
WEBSURFER_URL        = os.getenv("WEBSURFER_URL")
CODER_URL            = os.getenv("CODER_URL")
COMPUTERTERMINAL_URL = os.getenv("COMPUTERTERMINAL_URL")

WEBSURFER_DESCRIPTION = os.getenv("WEBSURFER_DESCRIPTION")
FILESURFER_DESCRIPTION = os.getenv("FILESURFER_DESCRIPTION")
CODER_DESCRIPTION    = os.getenv("CODER_DESCRIPTION")
COMPUTERTERMINAL_DESCRIPTION = os.getenv("COMPUTERTERMINAL_DESCRIPTION")

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
AGENT_DESCRIPTIONS: dict[str, str] = {
    "websurfer": WEBSURFER_DESCRIPTION,
    "filesurfer": FILESURFER_DESCRIPTION,
    "coder": CODER_DESCRIPTION,
    "computerterminal": COMPUTERTERMINAL_DESCRIPTION
}       

# --- Wrapper for Agent to leverage http ---
class HttpChatAgent(BaseChatAgent): 
    # BaseChatAgent를 상속하는 클래스는 아래 메소드를 구현하도록 강요받음 by @abstractmethod
    # 1. on_messages()
    # 2. produced_message_types()
    # 3. on_reset()

    """
    AutoGen 참가자처럼 동작하는 경량 래퍼
    - on_messages(messages, Cancellation Token) 인터페이스 제공
    - 내부에서 마이크로서비스의 /invoke HTTP 엔드포인터 호출
    - AssistantAgent에 구현되어있는 여러 메소드들 중 현재는 on_messages만 구현
    """
    def __init__(self, name: str, endpoint: str, description: str, timeout: float = REQUEST_TIMEOUT):
        self._name = name
        self._endpoint = endpoint
        self._description = description
        self._timeout = timeout

    async def _rpc(
            self,
            body: InvokeBody
        ) -> InvokeResult:
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient(headers=headers, timeout=self._timeout) as client:
            try:
                response: httpx.Response = await client.post(f"{self._endpoint}/invoke", json=body.model_dump())
                response.raise_for_status()
                return InvokeResult(**response.json())
            except httpx.HTTPStatusError as e:
                print(f"[{self._name}] HTTP {e.response.status_code}: {e.response.text}")
                raise
            except Exception as e:
                print(f"[{self._name}] Exception during RPC: {e}")
                raise 
    
    # 직렬화 보조 _rpc는 InvokeBody 형태로 받아야하기 때문.
    # @staticmethod
    # def _wire_messages(
    #     method: str, 
    #     messages: Sequence[BaseChatMessage],
    #     options: dict[str,Any]
    # ) -> InvokeBody:
    #     return InvokeBody(
    #         method = method,
    #         messages = messages
    #     )

    @property
    def name(self) -> str:
        return self._name

    @property    
    def endpoint(self) -> str:
        return self._endpoint
    
    @property
    def description(self) -> str:
        # TODO: description도 http로 받아오는게 맞지 않나? 왜냐하면 description은 orchestrator의 선택 근거 -> Magentic One의 Agent에 대한 description 필요.
        # 그런데 이런 사소한 부분까지 http로 받아오면 통신 오버헤드가 클 것 같음 => 하드코딩 <현재방식>
        return self._description
    
    # 현재는 BaseChatAgent의 @abstractmethod로 감싸진 필수 메서드만 구현.
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """Get the types of messages this agent can produce.

        Returns:
            Sequence of message types this agent can generate
        """
        types: list[type[BaseChatMessage]] = [TextMessage]
        return types
    
    async def on_messages(
            self,
            messages: Sequence[BaseChatMessage],
            cancellation_token: CancellationToken
        ) -> Response:
        body = InvokeBody(method="on_messages",messages=messages)
        result: InvokeResult = await self._rpc(body)
        if result.status != "ok":
            raise RuntimeError(f"{self.name}.on_messages failed: {result}")
        response = Response(chat_message=result.response["chat_message"],inner_messages=result.response["inner_messages"])
        return response

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        try: 
            await self._rpc(InvokeBody(method="on_reset"))
        except Exception as e:
            print(f"Exception occured while 'on_reset()': {e}")

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
            #"filesurfer": HttpChatAgent("filesurfer", SERVICE_ENDPOINTS["filesurfer"]),
            #"websurfer": HttpChatAgent("websurfer", SERVICE_ENDPOINTS["websurfer"]), 
            "coder": HttpChatAgent(name="coder",endpoint=SERVICE_ENDPOINTS["coder"],description=AGENT_DESCRIPTIONS["coder"]),
            "computerterminal": HttpChatAgent(name="computerterminal", endpoint=SERVICE_ENDPOINTS["computerterminal"],description=AGENT_DESCRIPTIONS["computerterminal"]),
        }
        _agent = MagenticOneGroupChat(
            name =  "Micro Magentic-One Orchestrator",
            model_client = _client,
            participants = list(participants.values()), # participant는 Magentic-One을 구성하는 하위 에이전트들의 목록.
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
    try:
        orchestrator = get_agent()
    except Exception as e:
        finish_time_perf = time.perf_counter()
        print(f"Exception occured: {e}")
        return InvokeResult(
            status="fail",
            response=None,
            elapsed={"orchestration_latency_ms": int((finish_time_perf-start_time_perf)*1000)}
        )
    
    # task는 str, TextMessage, ChatMessage 모두 가능.
    try:
        result: TaskResult = await orchestrator.run(task=task, cancellation_token=CancellationToken())
    except Exception as e:
        finish_time_perf = time.perf_counter()
        print(f"Execption occured while running orchestrator: {e}")
        return InvokeResult(
            status="fail",
            response=None,
            elapsed={"orchestration_latency_ms": int((finish_time_perf-start_time_perf)*1000)}
        )
    # yield TaskResult(messages=output_messages, stop_reason=stop_reason) : Groupchat의 결과.
    print(f"Stop reason: {result.stop_reason}")
    finish_time_perf = time.perf_counter()
    return InvokeResult(
        status="ok",
        response=result,
        elapsed={"orchestration_latency_ms": int((finish_time_perf-start_time_perf)*1000)}
    )
    