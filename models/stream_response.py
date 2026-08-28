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

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Can You give me a detailed essay on the history of Computer in French Language upto 1000 Words?")
]

for chunk in model.stream(conversation):
    print(chunk.text, end="", flush=True)