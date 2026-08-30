import sys
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Literal , Union
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain.agents.structured_output import MultipleStructuredOutputsError


load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")

def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"



# class ProductRating(BaseModel):
#     rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
#     comment: str = Field(description="Review comment")



model = ChatGroq(model="qwen/qwen3.6-27b")

agent = create_agent(
    model,
    tools=[],
    response_format=ToolStrategy(Union[EventDetails , ContactInfo]),
)

result = result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})

for msg in result['messages']:
    # If message is actually a ToolMessage object (not a dict), check its class name
    if type(msg).__name__ == "ToolMessage":
        print(msg.content)
    # If message is a dictionary or you want a fallback
    elif isinstance(msg, dict) and msg.get('tool_call_id'):
        print(msg['content'])

# print(result["structured_response"])