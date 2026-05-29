"""CLI entry point for the Customer Service Analytics Agent.

An interactive terminal chat that streams the agent's reasoning in real time:
every tool call and tool result is printed as it happens (SPEC.md 1.4.3), not
just the final answer. Conversation state persists per ``--session`` via the
SQLite checkpointer, so re-running with the same session resumes it.

Usage:
    python main.py --session my_session --user mohammad

Layer 4 (entry point). Imports from agent/ only.
"""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph
from agent.memory import get_checkpointer
from config import RECURSION_LIMIT

# ANSI colors for legible reasoning traces (degrade gracefully if unsupported).
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_EXIT_WORDS = {"quit", "exit", "q", ":q"}


def _format_tool_call(call: dict) -> str:
    """Render a single tool call as ``name(arg=value, ...)``."""
    args = call.get("args", {})
    shown = {k: v for k, v in args.items() if v is not None}
    arg_str = ", ".join(f"{k}={v!r}" for k, v in shown.items())
    return f"{call.get('name', '?')}({arg_str})"


def _render_update(node: str, update: dict) -> None:
    """Print the messages a node produced, styled by message type."""
    messages = update.get("messages", []) if isinstance(update, dict) else []
    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                # Optional reasoning text alongside the tool request.
                if isinstance(msg.content, str) and msg.content.strip():
                    print(f"{_DIM}  thinking: {msg.content.strip()}{_RESET}")
                for call in tool_calls:
                    print(f"{_CYAN}  -> tool: {_format_tool_call(call)}{_RESET}")
            else:
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                print(f"\n{_GREEN}{_BOLD}Agent:{_RESET} {text.strip()}")
        elif isinstance(msg, ToolMessage):
            result = msg.content if isinstance(msg.content, str) else str(msg.content)
            preview = result.strip().replace("\n", "\n           ")
            print(f"{_YELLOW}  <- result ({msg.name}):{_RESET} {preview}")


def run_chat(session: str, user: str) -> None:
    """Run the interactive REPL loop for one session/user."""
    graph = build_graph(checkpointer=get_checkpointer())
    config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": session},
    }

    print(f"{_BOLD}Customer Service Analytics Agent{_RESET}")
    print(f"{_DIM}session={session}  user={user}  (type 'quit' to exit){_RESET}")
    print(f"{_DIM}Ask about the Bitext customer-support dataset.{_RESET}\n")

    while True:
        try:
            user_input = input(f"{_BOLD}You:{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not user_input:
            continue
        if user_input.lower() in _EXIT_WORDS:
            print("Goodbye.")
            return

        # Reset iteration_count each turn (replace reducer); messages append.
        state_input = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": user,
            "iteration_count": 0,
        }

        try:
            for chunk in graph.stream(state_input, config, stream_mode="updates"):
                for node, update in chunk.items():
                    _render_update(node, update)
        except Exception as exc:  # noqa: BLE001 — surface any runtime error cleanly
            print(f"\n[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        print()


def main() -> None:
    """Parse CLI args and start the chat loop."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI for the Customer Service Analytics Agent."
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session id (thread_id). Reuse to resume a conversation.",
    )
    parser.add_argument(
        "--user",
        default="default",
        help="User id — selects which persistent profile to use.",
    )
    args = parser.parse_args()
    run_chat(session=args.session, user=args.user)


if __name__ == "__main__":
    main()
