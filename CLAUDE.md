# CLAUDE.md

## Project

Customer Service Analytics Agent — a LangGraph ReAct agent that answers questions about the Bitext Customer Service dataset. Built for Nebius Academy Assignment 3.

### Key Documents

- **[SPEC.md](SPEC.md)** — Requirements, acceptance criteria, grading breakdown, verification checklist
- **[PLAN.md](PLAN.md)** — Architecture, dependency rules, diagrams, project structure
- **[BREAKDOWN.md](BREAKDOWN.md)** — Block-by-block breakdown: effort (manual vs AI), grading mapping, AI/ML concepts, libraries

### Tech Stack

- **LLM**: Nebius Token Factory — `Qwen3-30B-A3B-Instruct-2507` (non-thinking MoE) for both primary + router roles, via `langchain-nebius`
- **Agent framework**: LangGraph (StateGraph, ReAct pattern)
- **Data**: Bitext dataset on HuggingFace → pandas DataFrame
- **Memory**: SqliteSaver (episodic) + JSON files (user profiles)
- **MCP**: FastMCP server
- **UI**: Streamlit (bonus)
- **Testing**: pytest — deterministic layer only (tools, loader, vocabulary drift guard). The LLM layer (router, agent) is smoke-tested manually, not unit-tested (non-deterministic + costs API tokens).

---

## My Learning Goals

- Understand the ReAct (Reasoning + Acting) pattern for AI agents
- Learn LangGraph: StateGraph, nodes, edges, conditional routing, checkpointers
- Learn how LLM tool-use / function calling works end-to-end
- Understand Pydantic schemas and why they matter for tool definitions
- Learn the MCP protocol and how to expose tools via FastMCP
- Practice clean architecture: layered dependencies, separation of concerns

---

## How to Work With Me

### Teaching Mode (default)

- **Do NOT write full implementations for me.** Guide me step by step.
- **Explain the concept/pattern BEFORE we write code.** For example, before building the router node, explain what structured output is and why we need it.
- **Show small code snippets** (10-20 lines max), then let me extend them.
- **When I get stuck, give hints before giving answers.** Ask me leading questions.
- **After each step, briefly explain WHY** we did it that way — connect it to the architecture in PLAN.md.
- **Ask me questions to check my understanding** when introducing new concepts.
- **Reference SPEC.md** to confirm we're meeting requirements as we go.

### When I say "just do it" or "implement this"

- Switch to implementation mode — write the full code.
- Still explain any non-obvious design decisions.
- Mark the relevant SPEC.md acceptance criteria as met.

### Code Standards

- Type hints on all function signatures
- Docstrings on public functions
- Meaningful variable names
- Follow the layered architecture in PLAN.md (tools never import from agent/)
- Tools return `str`, not raw DataFrames

---

## Implementation Steps

Work through these blocks in order. Each block is a self-contained conceptual unit that can be reviewed and tested before moving on.

---

### BLOCK A — Data Foundation (Layer 0 + 1)
**Goal:** Set up project skeleton + data access (no AI yet). **Done:** `from data.loader import get_dataframe` returns a clean pandas DataFrame.

- [ ] **A.1** Project skeleton:
  - Create `config.py` (env loading, model names, paths, ITERATION_LIMIT)
  - Create `.env.example` (with `NEBIUS_API_KEY=your-key-here`)
  - Create `.gitignore` (`.env`, `__pycache__/`, `*.db`, `user_profiles/`, the assignment PDF, `.venv/`)
  - Verify `requirements.txt` exists and versions are reasonable
  - Create `services/` directory (empty, will be populated in Block C/F)
  - Create empty `__init__.py` in `agent/`, `data/`, `services/`, `tools/`
- [ ] **A.2** Create `data/loader.py` — load Bitext dataset from HuggingFace, return singleton DataFrame
- [ ] **A.3** Explore the data: shape, columns, categories, intents, sample rows
  - **Verify**: 26,872 rows, 5 columns, 11 categories, 27 intents (numbers from SPEC.md)
- **Learn:** Project layout, dependency layers, `datasets` library, pandas basics on this dataset

---

### BLOCK B — Tool Layer: Deterministic Tools (Layer 2)
**Goal:** Build pure, deterministic tools on the DataFrame — no LLM, no I/O. **Done:** all 7 non-LLM tools work standalone.

- [ ] **B.1** Create `tools/schemas.py` — Pydantic input models for every tool
- [ ] **B.2** Create `tools/data_tools.py` — `count_rows`, `filter_by_category`, `filter_by_intent`, `get_distribution`, `list_unique_values`
- [ ] **B.3** Create `tools/display_tools.py` — `show_examples`, `find_instructions_by_keyword`
- [ ] **B.4** Set up pytest + test the deterministic layer (tools, loader, drift guard):
  - Add `pytest` to `requirements.txt`, create `tests/`
  - `tests/test_data.py` — loader shape/columns + `verify_vocabulary()` drift guard (constants match live data)
  - `tests/test_tools.py` — each tool: known input → expected output, plus edge cases (zero-match filters, invalid category → helpful error, `n` out of bounds)
  - Scope: **deterministic layer only**. The LLM layer (router, agent) is smoke-tested manually in later blocks, not unit-tested.
- **Learn:** `@tool` decorator, Pydantic `BaseModel` for args_schema, why tool descriptions matter to an LLM, pure-function design, testing pure functions with pytest

---

### BLOCK C — AI Layer: LLM Integration (Layer 1 + 2)
**Goal:** Integrate the LLM as a service, then use it inside a summarization tool. **Done:** you can call Nebius models from Python and `summarize_responses` works.

- [ ] **C.1** Create `services/llm.py` — `ChatNebius` factory for primary + router models
- [ ] **C.2** Smoke test: send a simple prompt, get a response
- [ ] **C.3** Test tool binding: `llm.bind_tools([count_rows])` — observe what tool calls the LLM generates
- [ ] **C.4** Create `tools/summary_tools.py` — `summarize_responses` (samples rows, calls LLM internally)
- [ ] **C.5** Create `tools/__init__.py` — assemble `all_tools` list
- **Learn:** ChatNebius config, `.bind_tools()`, how function calling works under the hood, prompt design for summarization

---

### BLOCK D — Agent Core: The ReAct Graph (Layer 3)
**Goal:** Build the multi-node LangGraph — router, agent ReAct loop, decline, fallback. **Done:** graph handles structured, unstructured, and out-of-scope queries end-to-end (no memory yet).

- [ ] **D.1** Create `agent/state.py` — `AgentState` TypedDict (messages, query_type, iteration_count, user_id)
- [ ] **D.2** Create `agent/prompts.py` — system prompts for router + agent
- [ ] **D.3** Create `agent/nodes.py` — `router_node` with structured output classifier
- [ ] **D.4** Test router in isolation on the 10 example queries from SPEC.md
- [ ] **D.5** Add `agent_node` (the ReAct reasoning step), `decline_node`, `fallback_node`
- [ ] **D.6** Create `agent/graph.py` — wire nodes with conditional edges, set `recursion_limit`
- [ ] **D.7** Test the full graph on all example queries — verify routing, tool chaining, fallback
- **Learn:** StateGraph composition, `.with_structured_output()`, ToolNode, conditional edges, the ReAct pattern, recursion limits

---

### BLOCK E — User Interface: CLI (Layer 4)
**Goal:** Make the agent usable from the terminal with visible reasoning. **Done:** `python main.py` opens an interactive chat showing each tool call and result.

- [ ] **E.1** Create `main.py` — argparse with `--session` and `--user` args (wired up in Block F)
- [ ] **E.2** Build the interactive loop using `graph.stream(stream_mode="updates")`
- [ ] **E.3** Print tool calls and observations in real time (not just final answers)
- [ ] **E.4** Handle exit gracefully (`quit`, `exit`, Ctrl+C)
- **Learn:** Streaming graph events, event types, how to extract reasoning steps from message objects

---

### BLOCK F — Memory & Persistence (Layer 1 + 3)
**Goal:** Make the agent stateful — episodic (per session) + semantic (per user) memory, stored separately. **Done:** agent remembers prior turns AND facts about the user across restarts.

#### Block F.1 — Episodic Memory (Task 2a, 20 pts)
- [ ] **F.1.1** Create `agent/memory.py` — `SqliteSaver` checkpointer factory
- [ ] **F.1.2** Wire checkpointer into `agent/graph.py` and `main.py`
- [ ] **F.1.3** Test: ask questions → quit → restart with same `--session` → conversation continues
- [ ] **F.1.4** Test: "Show 3 from REFUND" → "Show 3 more" → works
- [ ] **F.1.5** Test: "How many complaints?" → "What about refunds?" → "Total of last two?"

#### Block F.2 — Semantic Memory / User Profile (Task 2b, 10 pts)
- [ ] **F.2.1** Create `services/profile_store.py` — JSON CRUD (read/write/merge)
- [ ] **F.2.2** Create `tools/profile_tools.py` — `get_user_profile`, `update_user_profile`
- [ ] **F.2.3** Update `agent/prompts.py` — instruct agent when to read/update profile
- [ ] **F.2.4** Update `tools/__init__.py` — include profile tools
- [ ] **F.2.5** Test: "My name is X" → quit → restart → "What do you remember about me?" → recalls name

- **Learn:** Episodic vs semantic memory, why they need separate storage, LangGraph checkpointers, thread_id semantics

---

### BLOCK G — External Integration: MCP Server (Task 3, 20 pts)
**Goal:** Expose tools via MCP for external clients (bypasses the agent graph — MCP clients bring their own reasoning). **Done:** `python mcp_server.py` accepts client tool calls.

- [ ] **G.1** Create `mcp_server.py` — `FastMCP` instance with 4+ tools (filter_by_category, count_rows, show_examples, get_distribution)
- [ ] **G.2** Test: start the server, connect with a client, list tools, call a tool
- [ ] **G.3** Document client connection in README (e.g., Claude Desktop config JSON)
- **Learn:** MCP protocol basics, stdio transport, `@mcp.tool()` pattern, separation between server and agent

---

### BLOCK H — Advanced Features: Bonuses
**Goal:** Polish features — query recommender (H.1) and Streamlit UI (H.2). H.1 is most useful after F.2 since suggestions leverage the user profile. **Done:** agent suggests next queries; web chat works with reasoning visible.

#### Block H.1 — Query Recommender (Bonus B, +10 pts)
- [ ] **H.1.1** Create `tools/recommend_tools.py` — `suggest_query` tool
- [ ] **H.1.2** Update `agent/prompts.py` — instruct agent on the suggest → refine → confirm → execute flow
- [ ] **H.1.3** Test: "What should I query next?" → suggestion → user refines → user confirms → agent executes

#### Block H.2 — Streamlit UI (Bonus A, +10 pts)
- [ ] **H.2.1** Create `streamlit_app.py` — chat interface with `st.chat_message`
- [ ] **H.2.2** Sidebar with session ID + user ID inputs
- [ ] **H.2.3** Show reasoning steps in `st.expander` blocks (tool calls + results)
- [ ] **H.2.4** Wire to the shared checkpointer so it shares sessions with the CLI
- [ ] **H.2.5** Test: chat works, reasoning visible, session switching loads correct history

- **Learn:** Human-in-the-loop patterns, context-aware recommendation, Streamlit chat components, `@st.cache_resource`

---

### BLOCK I — Finalization
**Goal:** Submission-ready: README, verification, code quality pass. **Done:** a stranger can clone the repo and run the agent in 5 minutes.

- [ ] **I.1** Write `README.md` — setup, CLI usage, MCP connection guide, Streamlit usage, architecture overview, model choice justification
- [ ] **I.2** Run through the full verification checklist in [SPEC.md](SPEC.md)
- [ ] **I.3** Code quality pass: type hints, docstrings, naming, remove dead code
- [ ] **I.4** **Pin dependency versions** in `requirements.txt`:
  - Currently using `>=` for flexibility during development
  - Switch to `==` with exact versions captured from the working env (`pip freeze` as reference)
  - Reason: grader's machine runs `pip install -r requirements.txt` and must get the same versions we tested with — `>=` risks a future major release breaking things between submission and grading
- [ ] **I.5** **Reproduce grader's setup in a fresh env** (acid test):
  - `conda create -n nebius-test python=3.11 && conda activate nebius-test`
  - `pip install -r requirements.txt`
  - Run `python main.py`, `python mcp_server.py`, `streamlit run streamlit_app.py` — all must work
  - This catches conda→pip mismatches, missing system deps, and version pinning gaps before the grader sees them
- **Learn:** What makes a good portfolio README, how to test acceptance criteria systematically

---

## Architecture Rules

Lower layers never import from higher layers. **Critical:** `tools/` NEVER imports from `agent/`. `mcp_server.py` bypasses `agent/` (imports `tools/` + `data/` directly). Full layering and dependency graph in [PLAN.md](PLAN.md).

---

## API Version Notes (1.x / 3.x)

The installed env runs LangGraph 1.1.9, langchain-core 1.3.2, fastmcp 3.3.1. Several APIs differ from 0.x / 2.x patterns Claude may default to. **Use these forms** when writing code:

### LangGraph 1.x
- `recursion_limit` is passed at **invoke time**, not compile time:
  ```python
  graph.invoke(state, config={"recursion_limit": 25, "configurable": {"thread_id": session_id}})
  ```
  `workflow.compile(checkpointer=..., recursion_limit=25)` silently does nothing — the kwarg is ignored.
- `from langgraph.graph import StateGraph, END, START` — unchanged
- `from langgraph.prebuilt import ToolNode` — unchanged
- `from langgraph.graph.message import add_messages` — unchanged

### langgraph-checkpoint-sqlite 3.x
- `from langgraph.checkpoint.sqlite import SqliteSaver` — unchanged
- Two valid construction patterns:
  ```python
  # Pattern A: manual connection (we'll use this)
  conn = sqlite3.connect("conversations.db", check_same_thread=False)
  checkpointer = SqliteSaver(conn)

  # Pattern B: context manager (newer docs prefer this)
  with SqliteSaver.from_conn_string("conversations.db") as checkpointer:
      ...
  ```

### langchain-core 1.x
- `from langchain_core.tools import tool` — still works
- `from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage` — still works
- v1 also exposes `langchain.tools` and `langchain.messages` as top-level aliases — either path is fine
- `.bind_tools([...])` and `.with_structured_output(PydanticModel)` — unchanged
- The new `create_agent` (from `langchain.agents`) **rejects pre-bound models** — but we're hand-building our StateGraph so this doesn't affect us

### langchain-nebius 0.1.3
- `from langchain_nebius import ChatNebius` — unchanged
- **DO pass `base_url`** explicitly (pinned in `config.py` as `NEBIUS_BASE_URL`). The
  package's default points at the LEGACY `studio.nebius.ai` endpoint; we pin the
  Token Factory URL the official docs recommend, to hedge against the legacy URL
  being retired. (Both currently mirror the same models.)
  ```python
  from config import NEBIUS_API_KEY, NEBIUS_BASE_URL, PRIMARY_MODEL, PRIMARY_TEMPERATURE
  llm = ChatNebius(
      model=PRIMARY_MODEL,            # Qwen/Qwen3-30B-A3B-Instruct-2507
      api_key=NEBIUS_API_KEY,
      base_url=NEBIUS_BASE_URL,        # https://api.tokenfactory.nebius.com/v1/
      temperature=PRIMARY_TEMPERATURE,
  )
  ```
- **Use Instruct (non-thinking) models.** `Qwen3-32B` is a *thinking* model — it
  emits `<think>...</think>` in its content, which pollutes tool output and
  summaries. We use `Qwen3-30B-A3B-Instruct-2507` (no thinking tags, clean tool
  calls). There is no `Qwen3-14B` on Nebius — verify model names against the live
  `/models` endpoint, don't assume.

### fastmcp 3.x
- `from fastmcp import FastMCP` — unchanged
- Constructor takes ONLY the name; transport/host/port kwargs are gone:
  ```python
  mcp = FastMCP("Customer Service Dataset Tools")  # name only
  ```
- Transport config moved to `mcp.run()`:
  ```python
  mcp.run()                                  # stdio (default — for Claude Desktop)
  mcp.run(transport="stdio")                 # explicit stdio
  mcp.run(transport="http", host="0.0.0.0", port=8080)  # http
  ```
- `@mcp.tool` (no parens) is now the canonical form; `@mcp.tool()` still works
- Decorator now returns the original function (not a wrapper), so `tool_fn.name` no longer works — but direct calls do

---

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI
python main.py --session default --user default

# Run MCP server
python mcp_server.py

# Run Streamlit UI
streamlit run streamlit_app.py
```
