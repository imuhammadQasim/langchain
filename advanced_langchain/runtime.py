import sys

from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent , AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model, after_model
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime  
from langgraph.runtime import Runtime


sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


SYSTEM_PROMPT = """You are a helpful assistant."""


@dataclass
class Context:
    user_name: str
    
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest)->str:
    user_name = request.runtime.context.user_name  
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt

@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"Processing request for user: {runtime.context.user_name}")
    return None
    
@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"Completed request for user: {runtime.context.user_name}")
    return None

@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:
    """Fetch the user's email preferences from the store."""
    
    info = runtime.execution_info
    print(f"Thread: {info.thread_id}, Run: {info.run_id}")

    user_id = runtime.context.user_id  

    preferences: str = "The user prefers you to write a brief and polite email."
    if runtime.store:
        if memory := runtime.store.get(("users",), user_id):
            preferences = memory.value["preferences"]

    return preferences


print("[main] initializing model...")

model = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    temperature=0,
    timeout=300,
    max_tokens=1024,
)


agent = create_agent(
    model=model,
    context_schema=Context,
    checkpointer=InMemorySaver(),
    middleware=[dynamic_system_prompt, log_before_model, log_after_model],

)


user_message = "What's my name?"

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": user_message}
        ]
    },
    context=Context(user_name="John Smith"),
    config={
        "configurable": {
            "thread_id": "user-123"
        }
    },
)

print("[main] final answer:\n")
print(result["messages"][-1].content)