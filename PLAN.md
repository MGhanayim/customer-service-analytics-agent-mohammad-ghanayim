# PLAN.md — Implementation Plan

> See also: [SPEC.md](SPEC.md) (requirements), [BREAKDOWN.md](BREAKDOWN.md) (effort & knowledge by block)

---

## Context

Building a LangGraph ReAct agent for the Bitext Customer Service dataset (Nebius Academy Assignment 3). The agent answers structured data queries, summarizes data for open-ended questions, and declines out-of-scope questions. It uses Nebius Token Factory models, persists conversation via SQLite checkpointer, maintains per-user profiles, exposes tools via FastMCP, and includes a Streamlit UI.

**Dataset**: `bitext/Bitext-customer-support-llm-chatbot-training-dataset` on HuggingFace — 26,872 rows, columns: `flags`, `instruction`, `category` (11 values: ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION), `intent` (27 values), `response`.

**Model**: `Qwen/Qwen3-30B-A3B-Instruct-2507` (non-thinking MoE) for both primary and router roles, via `langchain-nebius`. Separate factory functions kept for future cost-tiering. See BREAKDOWN.md "Model Choice" for rationale.

---

## Clean Architecture — Dependency Rules

```
 ┌────────────────────────────────────────────────────────────────────┐
 │                        DEPENDENCY RULES                            │
 │                                                                    │
 │  Higher layers can import from lower layers.                       │
 │  Lower layers NEVER import from higher layers.                     │
 │  Same-layer modules can import from each other only if no cycles.  │
 │                                                                    │
 │  ┌──────────────────────────────────────────────────────────────┐  │
 │  │  LAYER 4 — Entry Points (main.py, streamlit_app.py)         │  │
 │  │            Thin wrappers. Import from agent/ only.           │  │
 │  └──────────────────────────┬───────────────────────────────────┘  │
 │                             │ imports                              │
 │  ┌──────────────────────────▼───────────────────────────────────┐  │
 │  │  LAYER 3 — Agent (agent/)                                    │  │
 │  │            Graph, nodes, state, prompts, memory.             │  │
 │  │            Imports from tools/ and services/.                │  │
 │  └──────────────────────────┬───────────────────────────────────┘  │
 │                             │ imports                              │
 │  ┌──────────────────────────▼───────────────────────────────────┐  │
 │  │  LAYER 2 — Tools (tools/)                                    │  │
 │  │            LangChain @tool definitions + Pydantic schemas.   │  │
 │  │            Imports from services/ and data/ only.            │  │
 │  │            NEVER imports from agent/.                        │  │
 │  └──────────────────────────┬───────────────────────────────────┘  │
 │                             │ imports                              │
 │  ┌──────────────────────────▼───────────────────────────────────┐  │
 │  │  LAYER 1 — Services & Data (services/, data/)                │  │
 │  │            Shared infrastructure. LLM factory, profile       │  │
 │  │            storage, dataset loader. No business logic.       │  │
 │  └──────────────────────────┬───────────────────────────────────┘  │
 │                             │ imports                              │
 │  ┌──────────────────────────▼───────────────────────────────────┐  │
 │  │  LAYER 0 — Config (config.py)                                │  │
 │  │            Env vars, model names, paths, constants.          │  │
 │  │            No imports from project modules.                  │  │
 │  └──────────────────────────────────────────────────────────────┘  │
 │                                                                    │
 │  SPECIAL: mcp_server.py imports from tools/ + data/ + services/   │
 │           (bypasses agent/ — MCP clients bring their own agent)    │
 └────────────────────────────────────────────────────────────────────┘
```

### Why this matters

In the original plan, `tools/summary_tools.py` imported the LLM directly from `agent/llm.py`. That creates a **circular dependency direction**: tools depend on agent, but agent also depends on tools. By extracting the LLM factory into `services/llm.py`, both layers import from a shared lower layer — no cycles, clean hierarchy.

---

## Project Structure

```
customer-service-agent/
│
├── config.py                     # LAYER 0: env vars, model names, paths, constants
│
├── data/                         # LAYER 1: data access
│   ├── __init__.py
│   └── loader.py                 #   Singleton DataFrame from HuggingFace
│
├── services/                     # LAYER 1: shared infrastructure
│   ├── __init__.py
│   ├── llm.py                    #   LLM factory (ChatNebius instances)
│   └── profile_store.py          #   User profile CRUD (JSON file I/O)
│
├── tools/                        # LAYER 2: LangChain tool definitions
│   ├── __init__.py               #   Re-exports all_tools list
│   ├── _validation.py            #   shared validate_category/validate_intent helpers
│   ├── schemas.py                #   Pydantic input models for all tools
│   ├── data_tools.py             #   count, filter_by_category/intent, distribution, list_unique_values
│   ├── display_tools.py          #   show_examples, find_instructions_by_keyword
│   ├── summary_tools.py          #   summarize_responses (calls services/llm)
│   ├── profile_tools.py          #   get/update profile (calls services/profile_store)
│   └── recommend_tools.py        #   suggest_query (Bonus B, calls services/llm)
│
├── agent/                        # LAYER 3: LangGraph agent
│   ├── __init__.py
│   ├── state.py                  #   AgentState TypedDict
│   ├── prompts.py                #   System prompts for router + agent
│   ├── nodes.py                  #   Node functions: router, agent, decline, fallback
│   ├── graph.py                  #   StateGraph build + compile
│   └── memory.py                 #   Checkpointer factory (SqliteSaver)
│
├── main.py                       # LAYER 4: CLI entry point
├── streamlit_app.py              # LAYER 4: Streamlit UI (Bonus A)
├── mcp_server.py                 # SPECIAL: FastMCP server (Task 3)
│
├── .env.example                  # NEBIUS_API_KEY=your-key-here
├── .gitignore
├── requirements.txt
├── README.md
├── PLAN.md
├── SPEC.md
├── BREAKDOWN.md
└── user_profiles/                # Runtime: per-user JSON files
```

### Import Graph (no cycles)

```
main.py ──────────────► agent/graph.py
streamlit_app.py ─────► agent/graph.py
                              │
                              ├──► agent/nodes.py ──► services/llm.py
                              │                   ──► tools/*
                              ├──► agent/state.py
                              ├──► agent/prompts.py
                              └──► agent/memory.py

tools/data_tools.py ──────► data/loader.py
tools/display_tools.py ───► data/loader.py
tools/summary_tools.py ───► data/loader.py + services/llm.py
tools/profile_tools.py ───► services/profile_store.py
tools/recommend_tools.py ─► services/llm.py

services/llm.py ──────────► config.py
services/profile_store.py ► config.py
data/loader.py ───────────► (HuggingFace datasets lib)

mcp_server.py ────────────► data/loader.py  (bypasses agent/)
```

---

## High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                                  │
│                                                                       │
│   ┌──────────┐     ┌────────────────┐     ┌────────────────────────┐ │
│   │  main.py │     │streamlit_app.py│     │   mcp_server.py        │ │
│   │  (CLI)   │     │   (Web UI)     │     │   (FastMCP)            │ │
│   └────┬─────┘     └──────┬─────────┘     └──────────┬─────────────┘ │
│        │                  │                           │               │
│        │  Shares graph    │                           │  Direct tool  │
│        │  & checkpointer  │                           │  access (no   │
│        ▼                  ▼                           │  agent/graph) │
│   ┌────────────────────────────────┐                  │               │
│   │         agent/                 │                  │               │
│   │  ┌────────┐ ┌───────┐         │                  │               │
│   │  │ graph  │ │ nodes │         │                  │               │
│   │  └───┬────┘ └───┬───┘         │                  │               │
│   │      │          │              │                  │               │
│   │  ┌───┴──┐  ┌────┴────┐        │                  │               │
│   │  │state │  │ prompts │        │                  │               │
│   │  └──────┘  └─────────┘        │                  │               │
│   │  ┌────────┐                    │                  │               │
│   │  │memory  │ (SqliteSaver)      │                  │               │
│   │  └────────┘                    │                  │               │
│   └────────────────┬───────────────┘                  │               │
│                    │                                  │               │
│                    ▼                                  ▼               │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │                       tools/                                  │   │
│   │                                                               │   │
│   │  data_tools ── display_tools ── summary_tools                 │   │
│   │  profile_tools ── recommend_tools                             │   │
│   └───────────┬──────────────────────────┬────────────────────────┘   │
│               │                          │                            │
│               ▼                          ▼                            │
│   ┌──────────────────┐     ┌─────────────────────────┐               │
│   │    data/          │     │      services/           │               │
│   │                   │     │                          │               │
│   │  loader.py        │     │  llm.py (LLM factory)   │               │
│   │  (DataFrame       │     │  profile_store.py       │               │
│   │   singleton)      │     │  (JSON CRUD)            │               │
│   └──────────────────┘     └──────────┬──────────────┘               │
│                                       │                               │
│                                       ▼                               │
│                            ┌──────────────────┐                       │
│                            │    config.py      │                       │
│                            │                   │                       │
│                            │  NEBIUS_API_KEY   │                       │
│                            │  MODEL_NAMES      │                       │
│                            │  DB_PATH          │                       │
│                            │  PROFILE_DIR      │                       │
│                            │  ITERATION_LIMIT  │                       │
│                            └──────────────────┘                       │
│                                                                       │
│   ┌──────────────────────────────────────────────────────────────┐    │
│   │              PERSISTENCE                                      │    │
│   │                                                               │    │
│   │  ┌──────────────────────┐   ┌─────────────────────────────┐  │    │
│   │  │  conversations.db    │   │  user_profiles/{id}.json    │  │    │
│   │  │  (SqliteSaver)       │   │  (profile_store.py)         │  │    │
│   │  │  episodic memory     │   │  semantic memory            │  │    │
│   │  └──────────────────────┘   └─────────────────────────────┘  │    │
│   └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│   ┌──────────────────────────────────────────────────────────────┐    │
│   │              EXTERNAL SERVICES                                │    │
│   │                                                               │    │
│   │  ┌───────────────────────────────────────────────────────┐   │    │
│   │  │  Nebius Token Factory (api.tokenfactory.nebius.com)   │   │    │
│   │  │                                                       │   │    │
│   │  │  ┌───────────────────────────────────────────────┐  │   │    │
│   │  │  │  Qwen3-30B-A3B-Instruct-2507 (non-thinking MoE)│  │   │    │
│   │  │  │  one model — router (temp 0) + agent (temp .2) │  │   │    │
│   │  │  └───────────────────────────────────────────────┘  │   │    │
│   │  └───────────────────────────────────────────────────────┘   │    │
│   └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Graph Architecture (LangGraph StateGraph)

### Node & Edge Diagram

```
                    ┌───────────┐
                    │   START   │
                    └─────┬─────┘
                          │
                          ▼
               ┌─────────────────────┐
               │      router         │  Qwen3-30B-A3B (temp 0)
               │                     │  classifies query as:
               │  "structured"       │  structured / unstructured / out_of_scope
               │  "unstructured"     │
               │  "out_of_scope"     │
               └──┬──────────────┬───┘
                  │              │
     ┌────────────┘              └────────────┐
     │ structured                             │ out_of_scope
     │ OR unstructured                        │
     ▼                                        ▼
┌──────────────────────┐           ┌─────────────────────┐
│       agent          │           │      decline        │
│                      │           │                     │
│  Qwen3-30B-A3B with  │           │  "I can only help   │
│  tool bindings       │           │   with questions     │
│                      │           │   about the Bitext   │
│  - Reasons about     │           │   dataset."          │
│    the question      │           │                     │
│  - Decides which     │           └──────────┬──────────┘
│    tool(s) to call   │                      │
│  - OR produces       │                      ▼
│    final answer      │                   ┌──────┐
│                      │                   │ END  │
└──┬───────┬───────┬───┘                   └──────┘
   │       │       │
   │       │       │ iteration_count >= 12
   │       │       │
   │       │       ▼
   │       │  ┌──────────────────────┐
   │       │  │     fallback         │
   │       │  │                      │
   │       │  │  "I've reached my    │
   │       │  │   reasoning limit.   │
   │       │  │   Here's what I      │
   │       │  │   found so far..."   │
   │       │  └──────────┬───────────┘
   │       │             │
   │       │             ▼
   │       │          ┌──────┐
   │       │          │ END  │
   │       │          └──────┘
   │       │
   │       │ no tool_calls (final answer)
   │       │
   │       ▼
   │    ┌──────┐
   │    │ END  │
   │    └──────┘
   │
   │ has tool_calls
   │
   ▼
┌──────────────────────┐
│   tool_executor      │
│                      │
│  Runs the tool(s)    │
│  the agent requested │
│  (ToolNode)          │
│                      │
│  Appends ToolMessage │
│  with results        │
└──────────┬───────────┘
           │
           │ always
           │
           └──────────────► back to [agent]
```

### Conditional Edge Logic (after agent node)

```
                ┌─────────────────────────┐
                │  Check after agent node  │
                └────────┬────────────────┘
                         │
                         ▼
              ┌──────────────────────┐     YES    ┌───────────┐
              │ iteration_count >= 12?├──────────►│  fallback  │
              └──────────┬───────────┘            └───────────┘
                         │ NO
                         ▼
              ┌──────────────────────┐     YES    ┌───────────────┐
              │ has tool_calls?      ├──────────►│ tool_executor  │
              └──────────┬───────────┘            └───────────────┘
                         │ NO
                         ▼
                      ┌──────┐
                      │ END  │  (agent produced final text answer)
                      └──────┘
```

### State Shape

```
┌─────────────────────────────────────────────┐
│              AgentState                      │
├─────────────────────────────────────────────┤
│                                             │
│  messages: list[BaseMessage]                │
│    ├── HumanMessage    (user input)         │
│    ├── AIMessage       (agent reasoning)    │
│    │     └── .tool_calls  (if any)          │
│    └── ToolMessage     (tool results)       │
│                                             │
│  query_type: str                            │
│    "structured" | "unstructured" |          │
│    "out_of_scope"                           │
│                                             │
│  iteration_count: int                       │
│    incremented each time agent node runs    │
│                                             │
│  user_id: str                               │
│    identifies which profile to load/update  │
│                                             │
└─────────────────────────────────────────────┘
```

**Hard limit**: `recursion_limit=25` passed at **invoke time** (LangGraph 1.x) via `graph.invoke(input, config={"recursion_limit": 25, "configurable": {"thread_id": ...}})`. It is NOT a `.compile()` kwarg — passing it there silently does nothing. Catches any missed infinite loops.

---

## ReAct Loop — Example Walkthroughs

### Example 1: Single-tool structured query

**User:** "How many refund requests did we get?"

```
┌─ Turn ────────────────────────────────────────────────────────────────┐
│                                                                       │
│  User: "How many refund requests did we get?"                         │
│    │                                                                  │
│    ▼                                                                  │
│  [router] → classifies as "structured"                                │
│    │                                                                  │
│    ▼                                                                  │
│  [agent] Thought: I need to count rows with intent "get_refund"       │
│    │     Action: count_rows(intent="get_refund")                      │
│    │                                                                  │
│    ▼                                                                  │
│  [tool_executor] → runs count_rows → "Found 996 rows"                │
│    │                                                                  │
│    ▼                                                                  │
│  [agent] Answer: "There were 996 refund requests in the dataset."     │
│    │                                                                  │
│    ▼                                                                  │
│  END                                                                  │
└───────────────────────────────────────────────────────────────────────┘
```

### Example 2: Multi-tool structured query (loop iterates twice)

**User:** "Which intent in the ACCOUNT category is most common, and show me 2 examples of it?"

This is the only walkthrough where the ReAct loop runs more than once: the agent
calls one tool, reads the result, *then decides on a second tool* based on what it
learned. It exercises the back-edge `tool_executor → agent` and bumps
`iteration_count` twice.

```
┌─ Turn ────────────────────────────────────────────────────────────────┐
│                                                                       │
│  User: "Which intent in ACCOUNT is most common, show me 2 examples?"  │
│    │                                                                  │
│    ▼                                                                  │
│  [router] → "structured"                                              │
│    │                                                                  │
│    ▼                                                                  │
│  [agent] Thought: First find the top intent in ACCOUNT.   (iter 1)    │
│    │     Action: get_distribution(group_by="intent",                  │
│    │              filter_category="ACCOUNT")                          │
│    ▼                                                                  │
│  [tool_executor] → {"edit_account": 1000, "switch_account": 1000,    │
│    │                 "registration_problems": 999, ...}               │
│    ▼                                                                  │
│  [agent] Thought: Top intent is edit_account. Now fetch 2  (iter 2)   │
│    │     examples of it.                                              │
│    │     Action: show_examples(intent="edit_account", n=2)            │
│    ▼                                                                  │
│  [tool_executor] → 2 sample rows (instruction + response previews)   │
│    ▼                                                                  │
│  [agent] Answer: "The most common ACCOUNT intent is edit_account     │
│    │              (~1000 rows). Two examples: ..."                    │
│    ▼                                                                  │
│  END                                                                  │
└───────────────────────────────────────────────────────────────────────┘
```

> The same one-tool-with-a-filter call from before
> (`get_distribution(group_by="intent", filter_category="ACCOUNT")`) is still a
> *single* tool call — multiple arguments are not multiple tools. True multi-tool
> behavior is the agent **chaining** calls across loop iterations, as above.

### Example 3: Unstructured query (LLM summarization)

**User:** "Summarize the FEEDBACK category"

```
┌─ Turn ────────────────────────────────────────────────────────────────┐
│                                                                       │
│  User: "Summarize the FEEDBACK category"                              │
│    │                                                                  │
│    ▼                                                                  │
│  [router] → "unstructured"                                            │
│    │                                                                  │
│    ▼                                                                  │
│  [agent] Action: summarize_responses(category="FEEDBACK")             │
│    │                                                                  │
│    ▼                                                                  │
│  [tool_executor] → summarize_responses internally:                    │
│    │   1. Filters DataFrame to FEEDBACK                               │
│    │   2. Samples 10 rows                                             │
│    │   3. Calls the model to summarize patterns                       │
│    │   4. Returns summary text                                        │
│    │                                                                  │
│    ▼                                                                  │
│  [agent] Answer: "The FEEDBACK category contains two intents:         │
│    │   complaint and review. Agents typically respond with..."         │
│    ▼                                                                  │
│  END                                                                  │
└───────────────────────────────────────────────────────────────────────┘
```

### Example 4: Out-of-scope query

**User:** "Who is the president of France?"

```
┌─ Turn ────────────────────────────────────────────────────────────────┐
│                                                                       │
│  User: "Who is the president of France?"                              │
│    │                                                                  │
│    ▼                                                                  │
│  [router] → "out_of_scope"                                            │
│    │                                                                  │
│    ▼                                                                  │
│  [decline] → "I can only help with questions about the Bitext         │
│    │          Customer Service dataset. That question is outside       │
│    │          my scope."                                               │
│    ▼                                                                  │
│  END                                                                  │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Memory Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       MEMORY SYSTEM                              │
│                                                                  │
│  ┌────────────────────────────┐  ┌────────────────────────────┐  │
│  │   EPISODIC MEMORY          │  │   SEMANTIC MEMORY           │  │
│  │   (Conversation History)   │  │   (User Profile)            │  │
│  │                            │  │                             │  │
│  │  What: Full message log    │  │  What: Distilled facts      │  │
│  │  per session               │  │  per user                   │  │
│  │                            │  │                             │  │
│  │  Stored in:                │  │  Stored in:                 │  │
│  │  conversations.db          │  │  user_profiles/{id}.json    │  │
│  │  (SqliteSaver)             │  │  (profile_store.py)         │  │
│  │                            │  │                             │  │
│  │  Keyed by:                 │  │  Schema:                    │  │
│  │  thread_id (--session)     │  │  {                          │  │
│  │                            │  │    "name": "...",            │  │
│  │  Enables:                  │  │    "frequent_topics": [...], │  │
│  │  - "Show me 3 more"        │  │    "preferences": {...},    │  │
│  │  - "What about refunds?"   │  │    "notes": "...",          │  │
│  │  - Restart & resume        │  │    "last_updated": "..."   │  │
│  │                            │  │  }                          │  │
│  │  Updated by:               │  │                             │  │
│  │  LangGraph checkpointer    │  │  Enables:                   │  │
│  │  (automatic after each     │  │  - "What do you remember    │  │
│  │   node execution)          │  │     about me?"              │  │
│  │                            │  │  - Context-aware recs       │  │
│  └────────────────────────────┘  │                             │  │
│                                  │  Updated by:                │  │
│                                  │  update_user_profile tool   │  │
│                                  │  (agent decides when)       │  │
│                                  └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Session Flow (with memory)

```
  Session 1 (--session abc)            Session 2 (--session abc)
  ┌──────────────────────┐             ┌───────────────────────────┐
  │ User: "My name is Mo"│             │ (restarts app with same   │
  │ Agent: "Nice to meet │             │  --session abc)            │
  │   you, Mo!"          │             │                           │
  │   [updates profile]  │             │ User: "What do you        │
  │                      │             │   remember about me?"     │
  │ User: "Show 3 from   │ ─────────► │ Agent: reads profile →    │
  │   REFUND"            │  SQLite +   │   "You're Mo, you were   │
  │ Agent: [shows 3]     │  JSON file  │    interested in refunds" │
  │                      │  persist    │                           │
  │ User: "Show 3 more"  │             │ User: "Show 3 more"      │
  │ Agent: [shows 3 more │             │ Agent: [knows REFUND ctx  │
  │   — knows context]   │             │   from checkpointer →    │
  │                      │             │   shows 3 more]           │
  └──────────────────────┘             └───────────────────────────┘
```

---

## Tool Organization

```
┌──────────────────────────────────────────────────────────────────────┐
│                            tools/                                     │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  data_tools.py — Pure data operations (no LLM, no I/O)        │   │
│  │  Depends on: data/loader.py                                    │   │
│  │                                                                │   │
│  │  filter_by_category(category)   ──► filtered summary           │   │
│  │  filter_by_intent(intent)       ──► filtered summary           │   │
│  │  count_rows(category?, intent?) ──► count                      │   │
│  │  get_distribution(group_by,     ──► {value: count, ...}        │   │
│  │       filter_category?)                                        │   │
│  │  list_unique_values(column)     ──► [val1, val2, ...]          │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  display_tools.py — Show/search raw data (no LLM, no I/O)     │   │
│  │  Depends on: data/loader.py                                    │   │
│  │                                                                │   │
│  │  show_examples(n, category?, intent?) ──► formatted rows       │   │
│  │  find_instructions_by_keyword(keyword, n) ──► matching rows   │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  summary_tools.py — LLM-powered analysis                      │   │
│  │  Depends on: data/loader.py + services/llm.py                  │   │
│  │                                                                │   │
│  │  summarize_responses(category?, intent?)                       │   │
│  │    └── samples rows (fixed size) → calls model → returns summary│  │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  profile_tools.py — User profile management                    │   │
│  │  Depends on: services/profile_store.py                         │   │
│  │                                                                │   │
│  │  get_user_profile(user_id)         ──► profile JSON            │   │
│  │  update_user_profile(user_id, ...) ──► confirmation            │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  recommend_tools.py — Bonus B                                  │   │
│  │  Depends on: services/llm.py                                   │   │
│  │                                                                │   │
│  │  suggest_query(conversation_summary, user_profile_summary)     │   │
│  │    └── calls the model → returns 2-3 suggestions               │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  All tools return str (not DataFrames) to keep LLM context tight.    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## MCP Server Architecture

```
┌──────────────────────────┐      stdio (JSON-RPC)      ┌───────────────────────┐
│     MCP Client           │◄──────────────────────────►│   mcp_server.py       │
│                          │                             │   (FastMCP)           │
│  - Claude Desktop        │  1. Client discovers tools  │                       │
│  - Claude Code           │  2. Client calls tool       │   Exposes 4+ tools:   │
│  - Any MCP-compatible    │  3. Server executes on      │   - filter_by_category│
│    client                │     DataFrame               │   - count_rows        │
│                          │  4. Server returns result    │   - show_examples     │
│                          │                             │   - get_distribution  │
└──────────────────────────┘                             │                       │
                                                         │   ┌───────────────┐   │
  Config (claude_desktop_config.json):                   │   │data/loader.py │   │
  {                                                      │   │  (DataFrame)  │   │
    "mcpServers": {                                      │   └───────────────┘   │
      "customer-service": {                              │                       │
        "command": "python",                             │   No agent/graph —    │
        "args": ["mcp_server.py"]                        │   MCP clients bring   │
      }                                                  │   their own reasoning │
    }                                                    └───────────────────────┘
  }
```

---

## Streamlit UI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Customer Service Data Analyst Agent                     [Streamlit]│
├───────────────┬─────────────────────────────────────────────────────┤
│               │                                                     │
│   SIDEBAR     │              CHAT AREA                              │
│               │                                                     │
│  Session ID:  │  ┌─────────────────────────────────────────────┐    │
│  [my_session] │  │  User: What categories exist?                │    │
│               │  └─────────────────────────────────────────────┘    │
│  User ID:     │                                                     │
│  [mohammad ]  │  ┌─────────────────────────────────────────────┐    │
│               │  │  Agent:                                      │    │
│  [New Session]│  │                                              │    │
│               │  │  > Tool: list_unique_values(column=category) │    │
│               │  │    Result: ACCOUNT, CANCEL, CONTACT, ...     │    │
│               │  │                                              │    │
│               │  │  The dataset contains 11 categories:         │    │
│               │  │  ACCOUNT, CANCEL, CONTACT, DELIVERY, ...     │    │
│               │  └─────────────────────────────────────────────┘    │
│               │                                                     │
│               │  ┌─────────────────────────────────────────────┐    │
│               │  │  Ask about the customer service dataset...   │    │
│               │  └─────────────────────────────────────────────┘    │
└───────────────┴─────────────────────────────────────────────────────┘
```

---

## Tools Summary (11 total)

### Data Tools (`tools/data_tools.py`)
| Tool | Pydantic Schema | Returns |
|------|----------------|---------|
| `filter_by_category` | `category: str` | Row count + unique intents + 3 preview rows |
| `filter_by_intent` | `intent: str` | Row count + parent category + 3 example instructions |
| `count_rows` | `category: Optional[str], intent: Optional[str]` | Count string |
| `get_distribution` | `group_by: str, filter_category: Optional[str], filter_intent: Optional[str]` | Value counts dict |
| `list_unique_values` | `column: str` | Sorted unique values |

### Display Tools (`tools/display_tools.py`)
| Tool | Pydantic Schema | Returns |
|------|----------------|---------|
| `show_examples` | `n: int=5, category: Optional[str], intent: Optional[str]` | Sampled rows formatted |
| `find_instructions_by_keyword` | `keyword: str, n: int=5` | Matching rows (literal substring) with category/intent |

### Summary Tools (`tools/summary_tools.py`)
| Tool | Pydantic Schema | Returns |
|------|----------------|---------|
| `summarize_responses` | `category: Optional[str], intent: Optional[str]` (sample size is an internal config constant, not an arg) | LLM-generated summary |

### Profile Tools (`tools/profile_tools.py`)
| Tool | Pydantic Schema | Returns |
|------|----------------|---------|
| `get_user_profile` | `user_id: str` | Profile JSON or "no profile" |
| `update_user_profile` | `user_id: str, name: Optional[str], frequent_topics: Optional[list[str]], preferences: Optional[dict], notes: Optional[str]` | Confirmation |

### Recommend Tools (`tools/recommend_tools.py`) — Bonus B
| Tool | Pydantic Schema | Returns |
|------|----------------|---------|
| `suggest_query` | `conversation_summary: str, user_profile_summary: str` | 2-3 suggested follow-up queries |

---

## Implementation Order

See **[CLAUDE.md](CLAUDE.md)** for the canonical, block-based implementation order (Blocks A–I). This document focuses on **what** to build (architecture); CLAUDE.md focuses on **how to learn and build it** step by step.

---

## Key Dependencies

```
langgraph>=0.2.0
langgraph-checkpoint-sqlite>=2.0.0
langchain-core>=0.3.0
langchain-nebius>=0.1.3
datasets>=3.0.0
pandas>=2.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
fastmcp>=2.0.0
streamlit>=1.40.0
```

---

## Verification

See [SPEC.md](SPEC.md) for the full acceptance criteria and verification checklist.

Quick smoke tests:
- "What categories exist?" → structured → `list_unique_values`
- "How many refund requests?" → structured → `count_rows(intent="get_refund")`
- "Summarize the FEEDBACK category" → unstructured → `summarize_responses`
- "Who is the president of France?" → out_of_scope → polite decline
- Session persistence: quit → restart with same `--session` → context preserved
- "What do you remember about me?" → reads user profile
- MCP: `python mcp_server.py` → client calls tools
- Streamlit: `streamlit run streamlit_app.py` → chat + reasoning visible
