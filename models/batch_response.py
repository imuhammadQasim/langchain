import sys
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")


model = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    temperature=0.6,
    timeout=300,
    max_tokens=1024,
)

responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)
    print("*" * 150)
