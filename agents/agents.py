from langchain.agents import create_agent ,AgentState
from langchain.tools import tool
from pydantic import BaseModel
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

class Answer(BaseModel):
    summary: str
    confidence: float

class MyState(AgentState):
    user_id: str
    call_count: int

@tool
def search(query: str) -> str:
    """Search for information."""
    print('=================COngrulation agent is calling the tool.================')
    return f"Results for: {query}"

print("agent is start working .....")
agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[search],
    system_prompt="You are a helpful assistant. Be concise and accurate.",
    response_format=Answer,
    state_schema=MyState
    )


result = agent.invoke({"messages": [{"role": "user",  "content": "Summarize AI trends"}]})

print('result is here------------->> ', result["structured_response"])