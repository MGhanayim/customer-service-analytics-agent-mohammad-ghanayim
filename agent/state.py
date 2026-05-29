"""Shared state for the ReAct agent graph.

Every node receives an ``AgentState`` and returns a PARTIAL update (only the
keys it changes); LangGraph merges that update back in. Most fields use the
default "replace" reducer — the node computes the final value. ``messages`` is
the exception: it uses the ``add_messages`` reducer so each node's new messages
are APPENDED to the history instead of overwriting it.

Layer 3 (agent). May import from any lower layer; nothing lower imports from here.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

QueryType = Literal["structured", "unstructured", "out_of_scope"]


class AgentState(TypedDict):
    """State threaded through router → agent → tools → END."""

    # Full conversation. add_messages appends new messages (and can update an
    # existing one by id) rather than replacing the list.
    messages: Annotated[list[BaseMessage], add_messages]

    # Router's classification of the latest user query.
    query_type: QueryType

    # Incremented by agent_node each run; drives the soft ITERATION_LIMIT check.
    iteration_count: int

    # Identifies which JSON profile to load/update (semantic memory, Block F).
    user_id: str
