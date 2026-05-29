# Customer Service Analytics Agent

> A LangGraph **ReAct agent** that answers natural-language questions about the
> [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
> (26,872 conversations) — routing each query, chaining tools to reason, and
> remembering you across sessions. Exposed via CLI, a FastMCP server, and a
> Streamlit web UI.

![Python](https://img.shields.io/badge/python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-orange)
![Nebius](https://img.shields.io/badge/LLM-Nebius%20Qwen3--30B-green)
![MCP](https://img.shields.io/badge/MCP-FastMCP%203.x-purple)
![Tests](https://img.shields.io/badge/tests-24%20passing-brightgreen)

---

## What it does

Ask it anything about the dataset and it figures out *how* to answer:

| You ask | It does |
|---------|---------|
| "How many refund requests are there?" | routes → `count_rows(intent='get_refund')` → **997** |
| "Which ACCOUNT intent is most common, and show 2 examples?" | chains `get_distribution` → `show_examples` (multi-step reasoning) |
| "Summarize how agents respond to complaints" | samples responses → LLM summary of tone & strategy |
| "What do you remember about me?" | reads your persistent profile |
| "Who is the president of France?" | politely **declines** — no general-knowledge answers |

It **analyzes** the support data; it does not act as a support agent.

---

## Why this project

- **Clean, layered architecture** — strict dependency rules (tools never import
  the agent), so each layer is testable in isolation. See [PLAN.md](PLAN.md).
- **The full agent toolkit** — query routing, Pydantic-typed tools, multi-step
  ReAct reasoning, a graceful iteration-limit fallback, two kinds of memory
  (episodic + semantic), and an MCP server — wired together, not hand-waved.
- **Grounded by design** — the agent answers only from tool results; an
  out-of-scope router and explicit grounding rules keep it from hallucinating.

---

## Architecture at a glance

```
START → router ─┬─ out_of_scope → decline → END
                └─ else → agent ─┬─ tool_calls → tools → agent  (ReAct loop)
                                 ├─ iteration cap → fallback → END
                                 └─ final answer → END
```

| Layer | Package | Responsibility |
|-------|---------|----------------|
| 4 — Entry points | `main.py`, `streamlit_app.py`, `mcp_server.py` | Thin UIs over the graph |
| 3 — Agent | `agent/` | StateGraph, nodes, prompts, memory |
| 2 — Tools | `tools/` | 11 LangChain tools + Pydantic schemas |
| 1 — Services/Data | `services/`, `data/` | LLM factory, profile store, dataset loader |
| 0 — Config | `config.py` | Env vars, model names, tuning knobs |

Full diagrams, dependency graph, and ReAct walkthroughs live in
**[PLAN.md](PLAN.md)**; requirements and acceptance criteria in **[SPEC.md](SPEC.md)**.

---

## Setup (≈5 minutes)

**Prerequisites:** Python 3.11 and a
[Nebius Token Factory](https://tokenfactory.nebius.com/settings/api-keys) API key.

```bash
# 1. Clone
git clone <your-repo-url>
cd customer-service-agent

# 2. Create an environment (conda or venv)
conda create -n cs-agent python=3.11 -y && conda activate cs-agent
# or: python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
#   then edit .env and set NEBIUS_API_KEY=...
```

The dataset downloads automatically from HuggingFace on first run (~20 MB,
cached afterwards).

---

## Usage

### CLI

```bash
python main.py --session my_session --user mohammad
```

- `--session` is the conversation thread — reuse it to **resume** a chat.
- `--user` selects the persistent profile.
- Reasoning is streamed live: each `-> tool:` call and `<- result:` is printed
  before the final answer. Type `quit` to exit.

### MCP server

Exposes four dataset tools (`count_rows`, `filter_by_category`,
`get_distribution`, `show_examples`) to any MCP client. It bypasses the agent
graph — MCP clients bring their own reasoning.

```bash
python mcp_server.py        # stdio transport
```

Connect from **Claude Desktop** by adding to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "customer-service": {
      "command": "python",
      "args": ["/absolute/path/to/customer-service-agent/mcp_server.py"]
    }
  }
}
```

Then ask Claude Desktop, e.g., *"Use the customer-service tools to get the
category distribution."*

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

Chat in the browser with reasoning shown in expandable panels. The sidebar
switches session/user — sessions are **shared with the CLI** (same SQLite
checkpointer), so you can start a conversation in one and resume it in the other.

---

## Memory

| Kind | What | Where | Keyed by |
|------|------|-------|----------|
| **Episodic** | Full message history (enables "show 3 more", follow-up math, resume-after-restart) | `conversations.db` via LangGraph `SqliteSaver` | `--session` |
| **Semantic** | Distilled facts (name, topics, preferences) | `user_profiles/{user}.json` | `--user` |

They are stored separately on purpose: conversation replay vs. durable facts are
different shapes of memory.

---

## Model choice

Both the router and the agent use **`Qwen/Qwen3-30B-A3B-Instruct-2507`** on
Nebius Token Factory:

- **Instruct, not Thinking** — it emits clean output with no `<think>` tags to
  strip. Our ReAct loop already provides the reasoning structure, so an internal
  chain-of-thought would be redundant and would pollute tool output.
- **MoE with ~3B active params** — fast and cheap to serve despite 30B total,
  and a reliable tool-caller.
- The router runs at **temperature 0.0** (deterministic classification); the
  agent at **0.2** (natural summaries, still reliable tool calls). Separate
  factory functions mean a cheaper router model can be swapped in later without
  touching the agent.

---

## Testing

```bash
pytest
```

24 tests cover the **deterministic layer** — the dataset loader, a vocabulary
drift guard (constants vs. live data), and every data/display tool (known input
→ expected output, plus edge cases). The LLM layer (router, agent) is
smoke-tested manually, since it is non-deterministic and costs API tokens.

---

## Project layout

```
customer-service-agent/
├── config.py              # Layer 0: env, model names, tuning knobs
├── data/loader.py         # Layer 1: cached DataFrame + vocabulary
├── services/              # Layer 1: llm.py (Nebius factory), profile_store.py
├── tools/                 # Layer 2: 11 tools + Pydantic schemas
├── agent/                 # Layer 3: state, prompts, nodes, graph, memory
├── main.py                # Layer 4: CLI
├── streamlit_app.py       # Layer 4: web UI
├── mcp_server.py          # FastMCP server
└── tests/                 # pytest (deterministic layer)
```

---

## License / attribution

Built for Nebius Academy Assignment 3. Dataset: Bitext Customer Support LLM
Chatbot Training Dataset (HuggingFace).
