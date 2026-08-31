import sys
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents.structured_output import ToolStrategy

from langchain.agents import create_agent
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
)

stream = agent.stream_events({
    "messages": [{"role": "user", "content": "What is the weather in SF?"}],
}, version="v3")

# for message in stream.messages:
#     for delta in message.text:
#         print(delta, end="", flush=True)

# final_state = stream.output


for event in stream:
    print(event , "*" * 40)