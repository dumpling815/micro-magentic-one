from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken  # Supports task cancellation while async processing
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from asyncio import run
from RequestSchema import Msg, InvokeBody, InvokeResult
from fastapi import HTTPException

_coder = None
_executor = None
def get_coder() -> MagenticOneCoderAgent:
    global _coder
    if _coder is None:
        _coder = MagenticOneCoderAgent(name="coder", model_client=OllamaChatCompletionClient(model="llama3.2:1b", host="http://localhost:11434", timeout=30)) # Description과 System message는 해당 클래스에 구현되어 있음.
    return _coder

def get_executor() -> LocalCommandLineCodeExecutor:
    global _executor
    if _executor is None:
        _executor = LocalCommandLineCodeExecutor(
            timeout = 30,
            work_dir="/workspace",
            allow_cmds=["python", "pip", "ls", "cat", "cd"],
            deny_cmds=["rm", "sudo"],
        )


async def test_func():
    coder = get_coder()
    content = input("Your request:")
    msg = Msg(source="user", content = content)
    body = InvokeBody(messages=[msg])
    py_msgs =[]
    for m in body.messages:
        if m.type != "TextMessage":
            raise HTTPException(status_code=400, detail=f"Unsupported message type: {m.type}")
        py_msgs.append(TextMessage(content=m.content, source=m.source))
    try:
        response = await coder.on_messages(py_msgs, CancellationToken())
        print(type((response.chat_message)))
        print(response.chat_message.content)
        print(f"Source:{response.chat_message.source}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run(test_func())