import sys
from dataclasses import dataclass

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver


sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


# -----------------------------
# System prompt
# -----------------------------

SYSTEM_PROMPT = "You are a helpful assistant."


# -----------------------------
# Runtime context
# -----------------------------

@dataclass
class Context:
    user_name: str
    user_role: str
    deployment_env: str


# -----------------------------
# Dynamic system prompt
# -----------------------------

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:

    # Read values from Runtime Context
    user_name = request.runtime.context.user_name
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    message_count = len(request.messages)

    base = SYSTEM_PROMPT

    # User information
    base += f"\nThe user's name is {user_name}."

    # Role-based instructions
    if user_role == "admin":
        base += "\nYou have admin access. You can perform all operations."

    elif user_role == "viewer":
        base += "\nYou have read-only access. Guide users toward read-only operations."

    else:
        base += "\nThe user's role is unknown. Do not perform privileged operations."

    # Environment-based instructions
    if env == "production":
        base += "\nYou are operating in production. Be extra careful with data modifications."

    elif env == "staging":
        base += "\nYou are operating in staging."

    elif env == "development":
        base += "\nYou are operating in development."

    # Long conversation handling
    if message_count > 10:
        base += "\nThis is a long conversation. Be extra concise."

    return base


# -----------------------------
# Model
# -----------------------------

model = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    temperature=0,
    timeout=300,
    max_tokens=1024,
)


# -----------------------------
# Create agent
# -----------------------------

agent = create_agent(
    model=model,
    context_schema=Context,
    checkpointer=InMemorySaver(),
    middleware=[context_aware_prompt],
)


# -----------------------------
# User message
# -----------------------------

user_message = "What is my role and what environment am I in?"


# -----------------------------
# Invoke agent
# -----------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_message,
            }
        ]
    },
    context=Context(
        user_name="John Smith",
        user_role="admin",
        deployment_env="staging",
    ),
    config={
        "configurable": {
            "thread_id": "user-123",
        }
    },
)


# -----------------------------
# Print final answer
# -----------------------------

print("[main] final answer:\n")
print(result["messages"][-1].content)