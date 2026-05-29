# SPEC.md — Assignment Requirements & Acceptance Criteria

> Source: "From AI Model to AI Agent — Assignment 3" (Nebius Academy)
> Due: 2026-05-29

---

## Overview

Build a **data analyst agent** for the **Bitext Customer Service dataset** that answers structured, unstructured, and out-of-scope questions. Implemented as a **LangGraph ReAct graph** with persistent memory, exposed via **FastMCP**, with optional **Streamlit UI**.

---

## Dataset

**Bitext Customer Service Tagged Training Dataset**
- Source: HuggingFace (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`)
- 26,872 rows, single `train` split
- Columns: `flags`, `instruction`, `category`, `intent`, `response`
- 11 categories, 27 intents
- Categories: ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION

---

## Constraints (Apply to All Tasks)

| ID | Constraint |
|----|-----------|
| C1 | Only **Nebius Token Factory** models allowed for all LLM calls |
| C2 | Must include `requirements.txt` or `pyproject.toml` with **version numbers** |
| C3 | Must include `README.md` — a grader must be able to clone and run within 5 minutes |
| C4 | README must cover: setup steps, how to run CLI, how to connect MCP client, architecture overview, model choice justification |
| C5 | Code quality: meaningful names, type hints, docstrings on public functions |
| C6 | Repo name must include first and last names of both students |

---

## Task 1 — Build the Initial Agent (50 pts)

### 1.1 Query Router (15 pts)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.1.1 | Dedicated router node in the graph | Router is a **separate named node** in the LangGraph StateGraph, not inline logic |
| 1.1.2 | Classifies queries into 3 types | Router outputs exactly one of: `structured`, `unstructured`, `out_of_scope` |
| 1.1.3 | Out-of-scope queries declined politely | Agent returns a polite refusal — does **NOT** answer from LLM general knowledge |
| 1.1.4 | Router runs before tool selection | Router node executes **before** the agent/tool-calling loop begins |

**Test queries for router:**
| Query | Expected Classification |
|-------|----------------------|
| "What categories exist in the dataset?" | structured |
| "How many refund requests did we get?" | structured |
| "Show me 3 examples from the SHIPPING intent" | structured |
| "What is the distribution of intents in the ACCOUNT category?" | structured |
| "Summarize the FEEDBACK category" | unstructured |
| "How do customer service reps typically respond to cancellation requests?" | unstructured |
| "Who won the 2024 Champions League?" | out_of_scope |
| "Write me a poem about customer service" | out_of_scope |
| "What's the best CRM software for handling complaints?" | out_of_scope |
| "Who is the president of France?" | out_of_scope |

### 1.2 Tools with Pydantic Schemas (15 pts)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.2.1 | Each tool has a clear name | Tool names are descriptive verbs (e.g., `count_rows`, `filter_by_category`) |
| 1.2.2 | Each tool has a clear description | A human can tell **when** to use the tool from description alone |
| 1.2.3 | Each tool has a Pydantic input schema | Input params defined via `BaseModel` with field descriptions and types |
| 1.2.4 | Return values are typed | Functions have return type annotations |
| 1.2.5 | "A few well-designed tools beat many poorly described ones" | Quality over quantity — each tool has a distinct, non-overlapping purpose |

### 1.3 Multi-step Reasoning (10 pts)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.3.1 | Agent chains multiple tools for complex queries | "How many refund requests?" triggers filter → count (or equivalent chaining) |
| 1.3.2 | Agent reasons about which tools to use | Agent's reasoning is visible in output (not just results) |

### 1.4 CLI Interface (5 pts)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.4.1 | Runs from command line | `python main.py` starts the agent |
| 1.4.2 | Interactive conversation loop | User can ask multiple questions in sequence |
| 1.4.3 | Shows reasoning steps | Tool calls **and** tool observations are printed, not just the final answer |

### 1.5 Max Iterations Fallback (5 pts)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 1.5.1 | Maximum iteration limit set (10-15 range) | Graph has a configured iteration cap |
| 1.5.2 | Graceful fallback message | If limit reached, agent returns a helpful message instead of spinning/crashing |

---

## Task 2 — Memory (30 pts)

### 2a. Conversation Memory / Episodic (20 pts)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 2a.1 | Uses LangGraph checkpoints | State persisted via a LangGraph `Checkpointer` (not custom serialization) |
| 2a.2 | Supports session ID argument | `python main.py --session my_session` starts/resumes a named session |
| 2a.3 | Same session ID restores conversation | Restarting app with same `--session` loads previous messages |
| 2a.4 | Handles follow-up references | "Show me 3 more" after "Show me 3 from REFUND" works correctly |
| 2a.5 | Handles cross-turn arithmetic | "How many complaints?" → "What about refunds?" → "Total of last two?" works |
| 2a.6 | Persists across restarts | Conversation state survives process termination (not in-memory only) |

### 2b. User Profile (10 pts)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 2b.1 | Stores distilled facts | Profile contains name, frequent topics, preferences — NOT a message replay |
| 2b.2 | Answers "What do you remember about me?" | Agent retrieves and reports profile contents |
| 2b.3 | Naturally updates from conversation | When user shares info ("My name is X"), profile updates without explicit command |
| 2b.4 | Stored separately from conversation history | Profile is a distinct data structure, not part of the checkpoint |
| 2b.5 | Persists across restarts | Profile survives process termination |

---

## Task 3 — MCP Server (20 pts)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| 3.1 | Built with FastMCP | Server uses the `FastMCP` class from `fastmcp` package |
| 3.2 | Exposes at least 3 tools | Minimum 3 tools registered as MCP tools |
| 3.3 | README shows how to start server | Clear instructions for running the MCP server |
| 3.4 | README shows how to connect a client | Example config or code showing a client calling one of the tools |

---

## Bonus A — Streamlit UI (+10 pts)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| A.1 | Chat interface | User types in chat input, sees agent responses in chat bubbles |
| A.2 | Shows reasoning steps | Tool calls and results displayed (e.g., in expanders), not just final answer |
| A.3 | Session ID in sidebar | Sidebar input lets user switch between / resume conversations |

---

## Bonus B — Query Recommender (+10 pts)

| ID | Requirement | Acceptance Criteria |
|----|------------|---------------------|
| B.1 | Triggered by "What should I query next?" | Agent recognizes recommendation requests |
| B.2 | Uses conversation history + user profile | Suggestions are contextually relevant, not generic |
| B.3 | Suggests but does NOT execute | Agent proposes a query and asks for confirmation |
| B.4 | Supports refinement | User can modify the suggestion through conversation |
| B.5 | Executes only on confirmation | Agent waits for explicit "yes" / "do it" before running the query |

**Example flow:**
```
User:  "What should I query next?"
Agent: "Based on your interest in refunds, you might want to see the
        distribution of intents in the REFUND category."
User:  "I'd rather see examples instead."
Agent: "Then I'd suggest: show 5 examples from the REFUND category.
        Should I go ahead?"
User:  "Yes, do it."
Agent: [executes and displays results]
```

---

## Grading Summary

| Component | Points |
|-----------|--------|
| Query router | 15 |
| Tools with Pydantic schemas | 15 |
| Multi-step reasoning | 10 |
| CLI with reasoning output | 5 |
| Max iterations fallback | 5 |
| **Task 1 subtotal** | **50** |
| Conversation memory (episodic) | 20 |
| User profile (semantic) | 10 |
| **Task 2 subtotal** | **30** |
| MCP server | 20 |
| **Task 3 subtotal** | **20** |
| **Base total** | **100** |
| Bonus A: Streamlit UI | +10 |
| Bonus B: Query recommender | +10 |
| **Maximum total** | **120** |

---

## Verification Checklist

Before submission, verify each of these end-to-end:

- [ ] `git clone` + follow README → agent running within 5 minutes
- [ ] "What categories exist?" → correct list returned
- [ ] "How many refund requests?" → correct count via tool chaining
- [ ] "Show me 5 examples of the SHIPPING category" → 5 relevant rows
- [ ] "Summarize how agents respond to complaint intents" → meaningful summary
- [ ] "Show me examples of people wanting their money back" → finds refund-related rows
- [ ] "What is the distribution of intents in the ACCOUNT category?" → correct breakdown
- [ ] "Who is the president of France?" → polite decline, no factual answer
- [ ] "What's the best CRM software?" → polite decline
- [ ] Session persistence: quit → restart with same `--session` → context preserved
- [ ] "Show 3 from REFUND" → "Show 3 more" → works
- [ ] "My name is X" → quit → restart → "What do you remember?" → returns name
- [ ] Max iteration fallback triggers gracefully
- [ ] MCP server starts and responds to client tool calls
- [ ] Streamlit: chat works, reasoning visible, session switching works
- [ ] Query recommender: suggests → refine → confirm → execute flow works
