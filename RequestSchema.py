from pydantic import BaseModel, Field
from typing import Any, Literal

class Msg(BaseModel):
    type: Literal["TextMessage"] = "TextMessage" # e.g. text, code, file, image, etc. 하지만 현재는 "TextMessage"만 사용 (간편화)
    source: Literal["user", "orchestrator", "filesurfer", "websurfer", "coder", "computerterminal"] # e.g. filesurfer agent, websurfer agent, orchestrator, etc.
    content: Any # payload
    def __str__(self):
        return f"{self.type} from {self.source}: {self.content}"

class InvokeBody(BaseModel):
    messages: list[Msg]
    options: dict[str, Any] = Field(default_factory=dict)

class InvokeResult(BaseModel):
    status: Literal["ok","fail"]
    message: Msg | None = None
    elapsed: dict