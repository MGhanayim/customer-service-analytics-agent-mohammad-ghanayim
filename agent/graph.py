"""Assemble and compile the ReAct StateGraph.

Wires the nodes (agent/nodes.py) and the ToolNode together with the conditional
edges that implement the ReAct loop:

    START → router ─┬─ out_of_scope → decline → END
                    └─ else → agent ─┬─ tool_calls → tools → agent (loop)
                                     ├─ iteration cap → fallback → END
                                     └─ final answer → END

See PLAN.md "Graph Architecture" for the full diagram.

Layer 3 (agent). The single entry point the Layer-4 apps (main.py,
streamlit_app.py) import.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import (
    agent_node,
    decline_node,
    fallback_node,
    route_after_agent,
    route_after_router,
    router_node,
)
from agent.state import AgentState
from tools import all_tools


def build_graph(checkpointer=None) -> CompiledStateGraph:
    """Build and compile the agent graph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver) for
            episodic memory. When provided, conversation state persists per
            ``thread_id``. Omit for a stateless graph (tests, MCP).

    Returns:
        A compiled graph ready for ``.invoke`` / ``.stream``.
    """
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(all_tools))
    workflow.add_node("decline", decline_node)
    workflow.add_node("fallback", fallback_node)

    # Edges
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {"agent": "agent", "decline": "decline"},
    )
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "fallback": "fallback", END: END},
    )
    workflow.add_edge("tools", "agent")  # ReAct back-edge
    workflow.add_edge("decline", END)
    workflow.add_edge("fallback", END)

    return workflow.compile(checkpointer=checkpointer)
