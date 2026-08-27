"""
Real-world subagent example: a research assistant.

The MAIN agent talks to the user. Whenever answering requires current/external
facts, it delegates to a `researcher` SUBAGENT (via the `task` tool that
SubAgentMiddleware adds automatically). The subagent runs its own private
loop -- it can call the web_search tool as many times as it needs -- and only
its final answer is returned to the main agent. The main agent never sees the
subagent's intermediate search calls; that's the whole point of a subagent
(it keeps the main agent's context window clean).

Run this file directly to see it live, with a full console trace of both
agents' model calls and tool calls:

    python agents/subagents.py
"""

import sys
from dotenv import load_dotenv

from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from ddgs import DDGS

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


@tool
def web_search(query: str) -> str:
    """Search the web and return the top results (title, snippet, URL) for a query."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    if not results:
        return f"No results found for: {query}"
    return "\n\n".join(
        f"{i}. {r['title']}\n   {r['body']}\n   {r['href']}"
        for i, r in enumerate(results, 1)
    )


class AgentTraceCallback(BaseCallbackHandler):
    """Prints every model call and tool call as they happen, tagged with which
    agent made them ("main" vs "researcher"), so the delegation is visible live
    instead of only seeing the final printed result.

    This works because SubAgentMiddleware compiles each subagent with
    `.with_config({"metadata": {"lc_agent_name": <name>}})`, and LangChain
    carries that metadata down into every nested model/tool call the subagent
    makes -- so we can read it straight off the callback events.
    """

    def __init__(self):
        self._labels: dict = {}
        self._tool_names: dict = {}

    def _label_for(self, run_id, parent_run_id, metadata):
        label = (metadata or {}).get("lc_agent_name") or self._labels.get(parent_run_id, "main")
        self._labels[run_id] = label
        return label

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        label = self._label_for(run_id, parent_run_id, metadata)
        print(f"\n[{label}] -> calling model  ({len(messages[0])} messages in context)")

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        label = self._labels.get(run_id, "main")
        message = response.generations[0][0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            for call in tool_calls:
                print(f"[{label}]    wants tool: {call['name']}({call['args']})")
        elif message.content:
            text = message.content if isinstance(message.content, str) else str(message.content)
            print(f"[{label}]    replied: {text[:300]}")

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, inputs=None, **kwargs):
        label = self._label_for(run_id, parent_run_id, metadata)
        name = serialized.get("name", "?")
        self._tool_names[run_id] = name
        print(f"[{label}] TOOL CALL   {name}({inputs or input_str})")

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        label = self._labels.get(run_id, "main")
        name = self._tool_names.get(run_id, "?")
        text = getattr(output, "content", output)
        print(f"[{label}] TOOL RESULT {name} -> {str(text)[:300]}")

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        label = self._labels.get(run_id, "main")
        print(f"[{label}] TOOL ERROR  {error}")


MODEL = "google_genai:gemini-3.6-flash"

backend = StateBackend()

agent = create_agent(
    model=MODEL,
    tools=[web_search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            system_prompt=(
                "For any question that needs current, external, or factual "
                "information, delegate it to the `researcher` subagent via the "
                "`task` tool instead of answering from memory."
            ),
            subagents=[
                {
                    "name": "researcher",
                    "description": (
                        "Searches the web and returns a source-cited summary. "
                        "Use for anything needing current or factual information."
                    ),
                    "system_prompt": (
                        "You are a research specialist. Call web_search at most twice "
                        "total -- once for the main query, and once more only if the "
                        "first result is clearly insufficient. Do not keep rephrasing "
                        "the same query. After that, answer with whatever you have and "
                        "cite the source URLs you used."
                    ),
                    "tools": [web_search],
                    "model": MODEL,
                    "middleware": [],
                }
            ],
        ),
    ],
)


if __name__ == "__main__":
    question = "What is LangChain's deepagents package and what are subagents used for?"
    print(f"USER: {question}\n{'=' * 80}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": [AgentTraceCallback()]},
    )

    print(f"\n{'=' * 80}\nFINAL ANSWER:\n")
    final_message = result["messages"][-1]
    print(final_message.content if isinstance(final_message, AIMessage) else final_message)
