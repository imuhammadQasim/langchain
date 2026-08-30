import sys
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
    key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")


model = ChatGroq(model="qwen/qwen3.6-27b")

agent = create_agent(
    model,
    response_format=ToolStrategy(ProductReview) 
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Not Good product: 1 out of 5 stars. Fast shipping, but expensive' and there are no space in the keypint use snake case or camel case for keypointss"}]
})

print(result["structured_response"])