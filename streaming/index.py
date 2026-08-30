import sys
from dotenv import load_dotenv

from langgraph.config import get_stream_writer  
from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    # stream any arbitrary data
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

model = ChatGroq(model="qwen/qwen3.6-27b")
agent = create_agent(
    model,
    tools=[get_weather]
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"],
    version="v2",
):
    print(f"stream_mode: {chunk['type']}")
    print(f"content: {chunk['data']}")
    print("*" *  50, "\n")
