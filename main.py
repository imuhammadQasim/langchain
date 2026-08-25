import http.client
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""


MAX_FETCH_CHARS = 6000


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL.
    """
    print(f"[fetch_text_from_url] fetching: {url}")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        print(f"[fetch_text_from_url] fetch failed: {e}")
        return f"Fetch failed: {e}"
    except http.client.IncompleteRead as e:
        print(f"[fetch_text_from_url] connection dropped mid-download, using {len(e.partial)} bytes received so far")
        raw = e.partial
    except ConnectionError as e:
        print(f"[fetch_text_from_url] connection error: {e}")
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    print(f"[fetch_text_from_url] fetched {len(text)} characters")
    if len(text) > MAX_FETCH_CHARS:
        print(f"[fetch_text_from_url] truncating to {MAX_FETCH_CHARS} characters to stay within model token limits")
        text = text[:MAX_FETCH_CHARS] + "\n...[truncated]"
    return text


print("[main] initializing model...")
model = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    temperature=0,
    timeout=300,
    max_tokens=1024,
)

print("[main] creating agent...")
agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
)

user_message = (
    "Fetch the text from "
    "https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/master/1342.txt "
    "and summarize the opening paragraph."
)
print(f"[main] sending message: {user_message}")

result = agent.invoke(
    {"messages": [{"role": "user", "content": user_message}]}
)

print("[main] full result:")
print(result)

print("[main] final answer:")
print(result["messages"][-1].content)