from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken  # Supports task cancellation while async processing
from autogen_core.code_executor import CodeBlock
from autogen_core.code_executor._func_with_reqs import to_code
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from asyncio import run
from RequestSchema import Msg, InvokeBody, InvokeResult
from fastapi import HTTPException
import re # for finding code block


CODE_BLOCK_PATTERN = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"

def code_block_parser(msg: TextMessage) -> CodeBlock:
    # This function gets the code block from the response message and return the pure python code.
    # Leverages Regular Expression.
    regex = re.compile(CODE_BLOCK_PATTERN, re.DOTALL)
    text = msg.content
    matches = regex.search(text)
    if matches:
        language = (matches.group(1) or "unknown").lower()
        code = matches.group(2).strip("\n")
        return CodeBlock(code=code, language=language)
    else:
        raise ValueError("No code block found in the message.")


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
            work_dir="/tmp/workspace",
            cleanup_temp_files=False,
        )
    return _executor


async def test_func():
    coder = get_coder()
    executor = get_executor()
    print(f"executor type: {type(executor)}")
    #content = input("Your request:")
    content = "build one simple python code with only one function which sums up two integer"
    msg = Msg(source="user", content = content)
    body = InvokeBody(messages=[msg])
    py_msgs =[]
    for m in body.messages:
        if m.type != "TextMessage":
            raise HTTPException(status_code=400, detail=f"Unsupported message type: {m.type}")
        py_msgs.append(TextMessage(content=m.content, source=m.source))
    try:
        response = await coder.on_messages(py_msgs, CancellationToken())
        print(type(response))
        print(type((response.chat_message)))
        print(response.chat_message.content)
        print(f"Source:{response.chat_message.source}")
        print(f"Working directory: {executor.work_dir.resolve()}")
        code_block = code_block_parser(response.chat_message)
        print(f"Parsed code:\n{code_block.code}")
        print(type(code_block))
        #print(f"#############FunctionWithRequirementsStr")
        #func = FunctionWithRequirementsStr(parsed)
        #print(type(func))
        #print(f"#############to code function")
        #code = to_code(func)
        code_blocks = [code_block]
        print(f"#############Executing the code block#############")
        code_execution_result = await executor.execute_code_blocks(code_blocks, CancellationToken())
        print(f"#############Code Execution Result")
        print(f"Exit Code: {code_execution_result.exit_code}\nOutput: {code_execution_result.output}\nCode file: {code_execution_result.code_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run(test_func())