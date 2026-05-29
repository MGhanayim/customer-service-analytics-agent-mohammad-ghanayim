"""Graph node functions and the conditional-edge routers.

Each node receives the full ``AgentState`` and returns a PARTIAL update that
LangGraph merges back in (messages append via add_messages; scalars replace).
The router functions don't mutate state — they only inspect it and return the
name of the next node for ``add_conditional_edges``.

Nodes:
  - router_node   : classifies the query (structured / unstructured / out_of_scope)
  - agent_node    : the ReAct step — reason, then either call tools or answer
  - decline_node  : polite refusal for out-of-scope queries
  - fallback_node : graceful message when the iteration limit is hit

Layer 3 (agent). Imports from services/ and tools/ (lower layers) plus sibling
agent modules (state, prompts). Never imported by tools/.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from agent.prompts import (
    ROUTER_SYSTEM_PROMPT,
    RouteDecision,
    build_agent_system_prompt,
)
from agent.state import AgentState
from config import ITERATION_LIMIT
from langgraph.graph import END
from services.llm import get_primary_llm, get_router_llm
from tools import all_tools

# Static refusal text. We do NOT ask the LLM to answer out-of-scope questions
# (SPEC.md 1.1.3) — the decline is deterministic so no general knowledge leaks.
_DECLINE_MESSAGE = (
    "I can only help with questions about analyzing the Bitext customer-support "
    "dataset — things like counts, distributions, example rows, or summaries of "
    "how agents respond. Your question looks outside that scope, so I can't help "
    "with it. Try asking, for example, \"How many refund requests are there?\" or "
    "\"Summarize how agents handle complaints.\""
)

_FALLBACK_MESSAGE = (
    "I've reached my reasoning limit for this question without arriving at a "
    "confident answer. Could you rephrase it or break it into smaller parts? For "
    "example, ask about one category or intent at a time."
)


def router_node(state: AgentState) -> dict:
    """Classify the user's latest query into one of three buckets.

    Uses the router model with structured output so the result is a parsed
    ``RouteDecision`` (a guaranteed-valid label), not free text. Passes the
    conversation history so follow-ups ("show 3 more") are classified in context.
    """
    classifier = get_router_llm().with_structured_output(RouteDecision)
    messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT), *state["messages"]]
    decision: RouteDecision = classifier.invoke(messages)
    return {"query_type": decision.query_type}


def agent_node(state: AgentState) -> dict:
    """The ReAct reasoning step: think, then call tools or produce a final answer.

    Binds all tools to the primary model and invokes it on the system prompt
    (with the live user_id) plus the conversation. The returned AIMessage either
    carries ``tool_calls`` (the loop will run them and come back) or is the final
    answer. Increments iteration_count to drive the soft fallback check.
    """
    llm_with_tools = get_primary_llm().bind_tools(all_tools)
    system = build_agent_system_prompt(state["user_id"])
    messages = [SystemMessage(content=system), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def decline_node(state: AgentState) -> dict:
    """Emit the polite, deterministic out-of-scope refusal."""
    return {"messages": [AIMessage(content=_DECLINE_MESSAGE)]}


def fallback_node(state: AgentState) -> dict:
    """Emit the graceful 'reasoning limit reached' message (SPEC.md 1.5)."""
    return {"messages": [AIMessage(content=_FALLBACK_MESSAGE)]}


# ----------------------------------------------------------------------------
# Conditional-edge routers (inspect state, return next node name)
# ----------------------------------------------------------------------------


def route_after_router(state: AgentState) -> str:
    """out_of_scope → decline; structured/unstructured → agent."""
    return "decline" if state["query_type"] == "out_of_scope" else "agent"


def route_after_agent(state: AgentState) -> str:
    """Decide what happens after the agent runs.

    Order matters: check the iteration cap FIRST (so a runaway loop is caught
    even if it keeps requesting tools), then whether the agent asked for tools,
    else the agent produced a final answer and we end.
    """
    if state.get("iteration_count", 0) >= ITERATION_LIMIT:
        return "fallback"
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END
