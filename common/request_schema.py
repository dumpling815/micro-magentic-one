from pydantic import BaseModel, Field
from typing import Any, Literal, Sequence
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.base import Response, TaskResult
# Response는 on_messages()의 결과, TaskResult는 run()의 결과

# Msg를 쓸 필요가 있을까? -> 없을 것 같음. => source는 어차피 BaseChatMessage에 있음.
# class Msg(BaseModel):
#     # type: Literal["TextMessage"] = "TextMessage" # e.g. text, code, file, image, etc. 하지만 현재는 "TextMessage"만 사용 (간편화)
#     source: Literal["user", "orchestrator", "filesurfer", "websurfer", "coder", "computerterminal"] # e.g. filesurfer agent, websurfer agent, orchestrator, etc.
#     content: BaseChatMessage
#     def __str__(self):
#         return f"{self.content.id} from {self.source}: {self.content}"

class InvokeBody(BaseModel):
    method: str | None = None
    messages: Sequence[BaseChatMessage] | None = None
    options: dict[str, Any] = Field(default_factory=dict)

class InvokeResult(BaseModel):
    status: Literal["ok","fail"]
    # autogen_agentchat.base의 Response는 pydantic 모델이 아니라 직렬화 보장하지 못해서 ㅇ로 대체
    response: dict[str, Any] | TaskResult | None = None
    elapsed: dict