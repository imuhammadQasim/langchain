from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
model = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    temperature=0.6,
    timeout=300,
    max_tokens=1024,
)

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather like in Boston?")
print("*" * 150)
print(response)
print("*" * 150)
print(response.tool_calls)
print("*" * 150)
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(tool_call)
    print("*" * 150)
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print("*" * 150)
    