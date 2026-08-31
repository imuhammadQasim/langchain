import sys
from dotenv import load_dotenv

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's always sunny in {city}!"

model = ChatGroq(model="qwen/qwen3.6-27b")
agent = create_agent(
    model, 
    tools=[get_weather],
    checkpointer=InMemorySaver(),
)



thread_config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Muhammad Qasim."}]},
    thread_config,
)["messages"][-1].content


print(response)