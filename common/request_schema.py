from pydantic import BaseModel, Field
from typing import Any, Literal, Sequence
from autogen_agentchat.messages import ChatMessage
from autogen_agentchat.base import TaskResult
import importlib
# Response는 on_messages()의 결과, TaskResult는 run()의 결과

# Msg를 쓸 필요가 있을까? -> 없을 것 같음. => source는 어차피 BaseChatMessage에 있음.
# class Msg(BaseModel):
#     # type: Literal["TextMessage"] = "TextMessage" # e.g. text, code, file, image, etc. 하지만 현재는 "TextMessage"만 사용 (간편화)
#     source: Literal["user", "orchestrator", "filesurfer", "websurfer", "coder", "computerterminal"] # e.g. filesurfer agent, websurfer agent, orchestrator, etc.
#     content: BaseChatMessage
#     def __str__(self):
#         return f"{self.content.id} from {self.source}: {self.content}"
AUTOGEN_MESSAGE_MODULE_PATH = "autogen_agentchat.messages"

def deserialize_messages(messages: list[dict]) -> list:
    # http를 사용하다보면 응답은 JSON으로 Serialize 되어서 돌아옴. 
    # 이 경우 class에 대한 정보는 'type'에만 남고 사라짐 (Autogen message 클래스들이 내부적으로 type 어트리뷰트를 구현해 두었음)
    # 따라서 이 타입 힌트를 이용해서 Dictionary를 다시 Autogen Message로 복구하는 작업이 필요하고, 이 함수가 그 역할을 수행.
    module = importlib.import_module(AUTOGEN_MESSAGE_MODULE_PATH)
    result = []
    for message in messages:
        msg_type = message.get("type")
        cls = getattr(module, msg_type)
        result.append(cls(**message))
    return result

class InvokeBody(BaseModel):
    method: str | None = None
    messages: Sequence[ChatMessage] | None = None
    options: dict[str, Any] = Field(default_factory=dict)

class InvokeResult(BaseModel):
    status: Literal["ok","fail"]
    # http를 통한 와이어 부분에서는 어차피 json 형식을 따르기 위해 dictionary 형태로 return 됨. 필요시 deserialize 방식을 써서 대체 필요.
    # autogen_agentchat.base의 Response는 pydantic 모델이 아니라 직렬화 보장하지 못해서 ㅇ로 대체
    response: dict[str, Any] | None = None
    elapsed: dict