from pydantic import BaseModel, Field
from typing import Any, Literal

class Msg(BaseModel):
    type: Literal["TextMessage"] # e.g. text, code, file, image, etc.
    source: Literal["user", "orchestrator", "filesurfer", "websurfer", "coder", "computerterminal"] # e.g. filesurfer agent, websurfer agent, orchestrator, etc.
    content: Any # payload

class InvokeBody(BaseModel):
    messages: list[Msg]
    options: dict[str, Any] = Field(default_factory=dict)

class InvokeResult(BaseModel):
    status: Literal["ok","fail"]
    message: Msg | None = None
    elapsed: dict