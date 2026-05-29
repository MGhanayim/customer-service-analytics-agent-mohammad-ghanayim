"""Streamlit chat UI for the agent (Bonus A).

A web chat over the same compiled graph + SQLite checkpointer the CLI uses, so
sessions are SHARED: a conversation started in the CLI under --session foo
resumes here under the same session id, and vice versa. Reasoning (tool calls
and results) is shown in collapsible expanders alongside each answer
(SPEC.md Bonus A.2). The sidebar switches session/user (A.3).

Run it:
    streamlit run streamlit_app.py

Layer 4 (entry point). Imports from agent/ only.
"""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph
from agent.memory import get_checkpointer
from config import RECURSION_LIMIT


@st.cache_resource
def get_graph():
    """Build the graph + checkpointer once per server process (shared across reruns)."""
    return build_graph(checkpointer=get_checkpointer())


def _format_tool_call(call: dict) -> str:
    args = {k: v for k, v in call.get("args", {}).items() if v is not None}
    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{call.get('name', '?')}({arg_str})"


def render_history(graph, session: str) -> None:
    """Replay the persisted conversation for this session as chat bubbles."""
    state = graph.get_state({"configurable": {"thread_id": session}})
    messages = state.values.get("messages", []) if state and state.values else []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            if isinstance(msg.content, str) and msg.content.strip():
                with st.chat_message("assistant"):
                    st.markdown(msg.content)


def stream_turn(graph, config, user_input: str, user: str) -> None:
    """Stream one turn, rendering reasoning in an expander then the final answer."""
    state_input = {
        "messages": [HumanMessage(content=user_input)],
        "user_id": user,
        "iteration_count": 0,
    }

    with st.chat_message("assistant"):
        reasoning_box = st.expander("Reasoning (tool calls & results)", expanded=False)
        answer_placeholder = st.empty()
        final_answer = ""

        for chunk in graph.stream(state_input, config, stream_mode="updates"):
            for _node, update in chunk.items():
                messages = update.get("messages", []) if isinstance(update, dict) else []
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        tool_calls = getattr(msg, "tool_calls", None)
                        if tool_calls:
                            for call in tool_calls:
                                reasoning_box.markdown(
                                    f"**Tool call:** `{_format_tool_call(call)}`"
                                )
                        else:
                            final_answer = (
                                msg.content
                                if isinstance(msg.content, str)
                                else str(msg.content)
                            )
                            answer_placeholder.markdown(final_answer)
                    elif isinstance(msg, ToolMessage):
                        result = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        reasoning_box.markdown(f"**Result ({msg.name}):**")
                        reasoning_box.code(result)


def main() -> None:
    st.set_page_config(page_title="Customer Service Analytics Agent", page_icon="📊")
    st.title("Customer Service Analytics Agent")
    st.caption("Ask about the Bitext customer-support dataset (26,872 conversations).")

    with st.sidebar:
        st.header("Session")
        session = st.text_input("Session ID", value="default")
        user = st.text_input("User ID", value="default")
        if st.button("Clear chat view"):
            st.rerun()
        st.markdown(
            "---\nSessions are shared with the CLI — reuse a session id to resume "
            "the same conversation."
        )

    graph = get_graph()
    config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": session},
    }

    render_history(graph, session)

    if user_input := st.chat_input("Ask about the dataset..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        stream_turn(graph, config, user_input, user)


if __name__ == "__main__":
    main()
