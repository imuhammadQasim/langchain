import sys
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents.structured_output import ToolStrategy

from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")


class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

model = ChatGroq(model="qwen/qwen3.6-27b")
agent = create_agent(
    model,
    response_format=ContactInfo 
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})


print(result["structured_response"])