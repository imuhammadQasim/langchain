"""
AI Research Digest Agent  —  LangChain agent + HumanInTheLoopMiddleware demo.

Flow:
  1. You ask it to research a topic and email you a digest.
  2. The agent does a REAL web search (DuckDuckGo, no API key).
  3. It drafts an HTML digest email with clickable sources.
  4. HumanInTheLoopMiddleware PAUSES before sending — you approve / edit / reject.
  5. On approval it sends a REAL email over Gmail SMTP.

Run:  python middleware/human_in_the_loop.py
.env: GROQ_API_KEY, GMAIL_EMAIL, GMAIL_APP_PASSWORD  (Gmail 16-char App Password)
"""

import os
import re
import sys

# Make the project root importable so `from services...` works when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()  # must run BEFORE importing services.smtp_email (it reads env at import)

from ddgs import DDGS

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_groq import ChatGroq

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from services.smtp_email import _send_email_sync

sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# 1. REAL TOOLS
# ============================================================

_AD_MARKERS = ("bing.com/aclick", "duckduckgo.com/y.js", "/aclk?", "googleadservices")


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the live web and return the top results as 'title | url | snippet' lines."""
    print(f"\n🔎 [web_search] {query!r}")
    hits = []
    for r in DDGS().text(query, max_results=max_results + 5, region="us-en"):
        url = r.get("href", "")
        if any(m in url for m in _AD_MARKERS):
            continue
        hits.append(f"- {r.get('title', '').strip()} | {url} | {r.get('body', '').strip()}")
        if len(hits) >= max_results:
            break
    print(f"   -> {len(hits)} results\n")
    return "\n".join(hits) if hits else "No results found."


@tool
def send_email_tool(to: str, subject: str, html_body: str) -> str:
    """Send an email. `to` is the recipient, `html_body` is the message as HTML."""
    print(f"\n📧 [send_email_tool] SENDING -> to={to!r} subject={subject!r}")
    if not os.environ.get("GMAIL_EMAIL") or not os.environ.get("GMAIL_APP_PASSWORD"):
        return "ERROR: set GMAIL_EMAIL and GMAIL_APP_PASSWORD in .env"
    _send_email_sync(to_email=to, subject=subject, html=html_body)
    return f"Email delivered to {to}"


# ============================================================
# 2. AGENT + HUMAN-IN-THE-LOOP MIDDLEWARE
# ============================================================
# interrupt_on maps a tool NAME to its approval policy:
#   True  -> pause; human may approve / edit / reject
#   False -> auto-approve
# HITL uses interrupt() internally, so the agent needs a checkpointer and every
# invoke() must pass a config with a thread_id so the paused run can be resumed.

SYSTEM_PROMPT = """\
You are a research assistant that emails digests.

When the user asks you to research a topic and email it:
1. Call web_search one or more times to gather current information.
2. Write a clean HTML email body: a one-line intro, then 4-6 <li> bullets, each
   summarising one finding in your own words and ending with
   <a href="URL">source</a>. Wrap it as <h2>..</h2><p>..</p><ul>..</ul>.
3. Call send_email_tool with the recipient, a short subject, and that HTML.
Never invent a recipient address or a source URL — use only URLs from search results.
"""

agent = create_agent(
    model=ChatGroq(model="qwen/qwen3.6-27b"),
    tools=[web_search, send_email_tool],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": True,   # irreversible -> require approval
                "web_search": False,       # read-only    -> auto-approve
            },
            description_prefix="⚠️  The agent wants to SEND this email",
        )
    ],
    checkpointer=InMemorySaver(),
)


# ============================================================
# 3. HUMAN REVIEW OF A PAUSED TOOL CALL
# ============================================================

def ask_human(action_request: dict) -> dict:
    """Turn one pending tool call into one HITL decision."""
    args = action_request["args"]
    print("\n" + "=" * 60)
    print("APPROVAL NEEDED:", action_request["name"])
    print("=" * 60)
    print("to     :", args.get("to"))
    print("subject:", args.get("subject"))
    print("body   :\n" + (args.get("html_body", "") or "")[:1200])
    print("-" * 60)

    try:
        choice = input("approve / edit / reject ? ").strip().lower()
    except EOFError:
        print("(no input -> auto-approving)")
        return {"type": "approve"}

    if choice == "reject":
        reason = input("reason (optional): ").strip()
        return {"type": "reject", "message": reason} if reason else {"type": "reject"}

    if choice == "edit":
        new_args = dict(args)
        for field in ("to", "subject"):
            v = input(f"{field} [{new_args.get(field)}]: ").strip()
            if v:
                new_args[field] = v
        return {
            "type": "edit",
            "edited_action": {"name": action_request["name"], "args": new_args},
        }

    return {"type": "approve"}


def strip_think(text: str) -> str:
    """Drop qwen's <think>...</think> reasoning from a printed reply."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def run_turn(user_text: str) -> None:
    """One chat message -> drain HITL interrupts -> print the reply."""
    result = agent.invoke({"messages": [{"role": "user", "content": user_text}]}, config)

    while "__interrupt__" in result:
        req = result["__interrupt__"][0].value
        decisions = [ask_human(a) for a in req["action_requests"]]
        print("\n▶️  resuming with:", decisions)
        result = agent.invoke(Command(resume={"decisions": decisions}), config)

    reply = result["messages"][-1]
    print("\n🤖", strip_think(getattr(reply, "text", None) or str(reply.content)), "\n")


# ============================================================
# 4. CHAT LOOP
# ============================================================

config = {"configurable": {"thread_id": "research-digest-1"}}

print("\n📰  AI Research Digest Agent")
print("    e.g.  research the latest on AI agent frameworks and email me a digest at you@gmail.com")
print("    type 'quit' to exit\n")

while True:
    try:
        user_text = input("You: ").strip()
    except EOFError:
        break
    if user_text.lower() in {"quit", "exit", "q", ""}:
        break
    run_turn(user_text)
