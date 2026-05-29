"""FastMCP server exposing dataset tools to external MCP clients (Task 3).

This server BYPASSES the agent graph: MCP clients (Claude Desktop, Claude Code,
etc.) bring their own reasoning, so they only need direct access to the
deterministic data tools. We re-use the exact same tool logic the agent uses
(by delegating to the LangChain tools) so behavior and validation stay identical
across both surfaces — single source of truth.

Run it:
    python mcp_server.py                 # stdio transport (for Claude Desktop)

Layer: SPECIAL. Imports from tools/ + data/ (never from agent/) — see PLAN.md.

API note (fastmcp 3.x): FastMCP(name) takes the name only; transport is chosen
in mcp.run(). ``@mcp.tool`` registers a function with its type hints + docstring
as the MCP tool schema.
"""

from __future__ import annotations

from fastmcp import FastMCP

# Reuse the agent's tools so logic/validation is identical on both surfaces.
from tools.data_tools import count_rows as _count_rows
from tools.data_tools import get_distribution as _get_distribution
from tools.display_tools import show_examples as _show_examples
from tools.data_tools import filter_by_category as _filter_by_category

mcp = FastMCP("Customer Service Dataset Tools")


@mcp.tool
def count_rows(category: str | None = None, intent: str | None = None) -> str:
    """Count rows in the Bitext dataset, optionally filtered by category and/or
    intent (filters combine with AND). Omit both to count the whole dataset."""
    return _count_rows.invoke({"category": category, "intent": intent})


@mcp.tool
def filter_by_category(category: str) -> str:
    """Overview of one category: row count, the intents it contains (with
    counts), and a few example customer messages."""
    return _filter_by_category.invoke({"category": category})


@mcp.tool
def get_distribution(
    group_by: str,
    filter_category: str | None = None,
    filter_intent: str | None = None,
) -> str:
    """Value-count distribution of a column ('category', 'intent', or 'flags'),
    optionally restricted to a category and/or intent first."""
    return _get_distribution.invoke(
        {
            "group_by": group_by,
            "filter_category": filter_category,
            "filter_intent": filter_intent,
        }
    )


@mcp.tool
def show_examples(
    n: int = 5,
    category: str | None = None,
    intent: str | None = None,
) -> str:
    """Show up to n example rows (customer message + truncated agent response),
    optionally filtered by category and/or intent."""
    return _show_examples.invoke({"n": n, "category": category, "intent": intent})


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
