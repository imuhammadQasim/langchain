
import sys
from dotenv import load_dotenv

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage , ToolMessage

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


model = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=200,
    )

system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")
human_msg = HumanMessage("Hello, how are you? Can you tell me about the KISS")
# ai_msg = AIMessage("Cherry blossoms bloom...")


# Dictionary Based Method
# messages = [
#     {"role": "system", "content": "You are a poetry expert"},
#     {"role": "user", "content": "Write a haiku about spring"},
#     {"role": "assistant", "content": "Cherry blossoms bloom..."}
# ]

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    ...

messages = [system_msg, human_msg]
# response = model.invoke(messages)  
# print(response.usage_metadata)


# CHunking the respne 

for chunk in model.stream(messages):
    print(chunk.text, end="", flush=True)