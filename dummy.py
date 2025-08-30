from autogen_agentchat.teams._group_chat._magentic_one import MagenticOneGroupChat
from autogen_agentchat.base import ChatAgent
from pydantic import BaseModel
class dummy(BaseModel):
    content: int

class HttpChatAgent(ChatAgent):
    content: int

participants = {
        "Dummy 1": dummy(content=1),
        "Dummy 2": dummy(content=2),
        "Dummy 3": dummy(content=3),
        "Dummy 4": dummy(content= 4),
    }

#a = dummy(content=1)
#print(type(a))
print(list(participants.values())[0].content)
print("#############")
a = HttpChatAgent(content=1)
print(isinstance(a, ChatAgent))