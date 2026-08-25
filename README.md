# LangChain Learning Playground

A personal sandbox for learning [LangChain](https://python.langchain.com/) by following the official documentation and experimenting with its agent APIs.

## Setup

1. Create/activate a virtual environment (already present in `.venv/`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with the API keys you need:
   ```
   GROQ_API_KEY=your-groq-key
   GOOGLE_API_KEY=your-google-key
   ```

## Project structure

```
main.py                  # Tool-calling agent that fetches text from a URL and summarizes it
agents/
  agents.py               # Agent with structured output (response_format) and a custom AgentState
  mem_agent.py             # Work-in-progress: agent memory/context via checkpointer (InMemorySaver)
```

### `main.py`

Uses `create_agent` from `langchain.agents` with a Groq-hosted model (`groq:openai/gpt-oss-120b`) and a custom `fetch_text_from_url` tool. Demonstrates:
- Defining a tool with `@tool`
- Writing a system prompt that tells the model what tools it has
- Invoking the agent with a single user message and reading the final response

### `agents/agents.py`

Demonstrates:
- Structured output via a Pydantic `response_format` (`Answer`)
- Extending `AgentState` with custom fields (`user_id`, `call_count`)
- A simple mock `search` tool

### `agents/mem_agent.py`

Exploring agent memory and context, based on the LangChain docs for `context_schema` and `checkpointer` (using `langgraph.checkpoint.memory.InMemorySaver`). Currently a work in progress — the agent setup is commented out while experimenting with `stream_events`.

## Notes

This repo is for learning purposes — code favors clarity and experimentation over production quality, and some files are intentionally incomplete as I work through the LangChain documentation.

## LangChain concepts reference

General notes on core LangChain building blocks, for quick reference regardless of what's implemented in this repo.

### Chat models

- `init_chat_model(model, **kwargs)` creates a provider-agnostic chat model from a `provider:model-name` string (e.g. `groq:llama-3.1-70b`, `google_genai:gemini-3.6-flash`, `openai:gpt-4o`).
- Common kwargs: `temperature`, `max_tokens`, `timeout`, `max_retries`.
- Each provider needs its own API key set as an environment variable (e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`).

### Tools

- Defined with the `@tool` decorator from `langchain.tools` on a function with a docstring (the docstring becomes the tool description the model sees).
- Type-annotate arguments so the model knows the expected input schema.
- Tools can return strings, dicts, or structured objects; keep return values small enough to fit the model's context window.

### Agents

- `create_agent(model, tools, system_prompt, ...)` from `langchain.agents` builds a tool-calling agent loop (built on LangGraph).
- Key options:
  - `response_format`: a Pydantic model to force structured output (available as `result["structured_response"]`).
  - `state_schema`: extend `AgentState` with custom fields carried through the run.
  - `context_schema`: static, run-scoped data (e.g. `user_id`) passed via `context=` at invoke time, not part of message state.
  - `checkpointer`: persists state between invocations (e.g. `InMemorySaver` from `langgraph.checkpoint.memory`) for multi-turn memory; requires a `thread_id` in `config`.
- Invocation:
  - `agent.invoke({"messages": [...]})` — single call, returns final state.
  - `agent.stream(...)` / `agent.stream_events(...)` — stream intermediate steps/tokens.

### Messages

- `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage` from `langchain.messages` (or `langchain_core.messages`) represent conversation turns.
- Agent results are `{"messages": [...]}`; the last message is typically the final answer.

### Memory & state

- Short-term memory (within a single run): carried via `state_schema` / the message list.
- Long-term memory (across runs): a `checkpointer` keyed by `thread_id`, or an external store (vector DB, database) wired in as a tool.

### Typical gotchas

- Forgetting `load_dotenv()` before reading API keys from `.env`.
- Missing/expired API keys for the specific provider a model string points to.
- Tool docstrings that are too vague, causing the model to call tools incorrectly or not at all.
- Truncating tool output insufficiently, blowing past the model's context/token limit.

### Useful docs

- LangChain Python docs: https://python.langchain.com/
- LangGraph docs (underlies `create_agent`): https://langchain-ai.github.io/langgraph/
