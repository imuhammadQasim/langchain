import sys
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain.agents import create_agent, AgentState
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain.agents.middleware import SummarizationMiddleware
from langchain_groq import ChatGroq


sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


# ============================================================
# 1. STRUCTURED OUTPUT
# ============================================================

class Answer(BaseModel):
    summary: str
    confidence: float


# ============================================================
# 2. CUSTOM AGENT STATE
# ============================================================

class MyState(AgentState):
    user_id: str
    call_count: int


# ============================================================
# 3. TOOL
# ============================================================

@tool
def search(query: str) -> str:
    """Search for information."""

    print("\n")
    print("🔧 ================= TOOL START =================")
    print(f"🔎 Search query: {query}")
    print("🔧 ================= TOOL END =================")
    print("\n")

    return f"Search results for: {query}"


# ============================================================
# 4. CREATE AGENT
# ============================================================

print("\n🚀 Agent is starting...\n")


agent = create_agent(

    model=ChatGroq(model="qwen/qwen3.6-27b"),

    tools=[search],

    system_prompt="""
You are a helpful assistant.

IMPORTANT:
Whenever the user asks about AI trends, ALWAYS use the search tool
before answering.

After getting the search result, provide a concise answer.
""",

    # ToolStrategy => structured output via a tool call, so it can coexist with
    # real tools on providers (like Groq) that forbid JSON mode + tool calling.
    response_format=ToolStrategy(Answer),

    state_schema=MyState,

    # middleware must be a LIST (a sequence), even for a single middleware
    middleware=[
        SummarizationMiddleware(

            # Model used ONLY for summarization
            model=ChatGroq(model="qwen/qwen3.6-27b"),

            # Normally 4000.
            # We intentionally use a very small number for testing.
            trigger=("tokens", 500),

            # Keep only the latest 2 messages after summarization.
            keep=("messages", 2),
        )
    ],
)


# ============================================================
# 5. CREATE A HUGE CONVERSATION
# ============================================================

long_text = """
Artificial intelligence is changing software development.
Large language models are becoming more capable.
AI agents can use tools and interact with external systems.
RAG systems can retrieve information from private databases.
Vector databases are useful for semantic search.
Modern applications increasingly combine LLMs with APIs.
Developers need to understand context windows and token limits.
Agent middleware can be used for logging, security and guardrails.
AI systems can also use memory to maintain useful information.
Production AI applications require monitoring and observability.
""" * 30


# ============================================================
# 6. INVOKE AGENT
# ============================================================

print("📨 Sending request to agent...\n")


result = agent.invoke({

    "messages": [
        {
            "role": "user",
            "content": f"""
Here is some background information:

{long_text}

Now tell me the latest AI trends.

Remember:
You MUST use the search tool before answering.
"""
        }
    ],

    "user_id": "user_123",

    "call_count": 1,
})


# ============================================================
# 7. FINAL RESULT
# ============================================================

# print("\n")
# print("🎯 ================= FINAL RESULT =================")
# print(result["structured_response"])
# print("====================================================")


print("\n================ FINAL STATE ================\n")
print(result)
print("\n================ Result is here ================\n")

for i, message in enumerate(result["messages"]):

    print(f"\nMESSAGE {i + 1}")
    print("TYPE:", type(message).__name__)
    print("CONTENT:", message.content)

    if hasattr(message, "tool_calls") and message.tool_calls:
        print("TOOL CALLS:", message.tool_calls)