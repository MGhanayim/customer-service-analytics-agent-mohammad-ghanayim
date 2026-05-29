"""Display tools: surface raw example rows for inspection.

Unlike data_tools (which return aggregates), these return actual row content —
so they truncate the long `response` column (up to ~2,472 chars) to PREVIEW_CHARS
to keep tool output token-cheap. `instruction` (max 92 chars) is shown in full.

Layer 2 (tools). Imports from config (Layer 0), data/loader (Layer 1), and the
shared validation helper. Never imports from agent/.
"""

from __future__ import annotations

import pandas as pd

from langchain_core.tools import tool

from config import PREVIEW_CHARS
from data.loader import get_dataframe
from tools._validation import validate_category, validate_intent
from tools.schemas import FindInstructionsByKeywordInput, ShowExamplesInput


def _truncate(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Clip ``text`` to ``limit`` chars, appending an ellipsis if it was cut."""
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _format_rows(rows: pd.DataFrame) -> str:
    """Render rows as a readable block: instruction in full, response truncated."""
    blocks = []
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        blocks.append(
            f"[{i}] category={row['category']} | intent={row['intent']}\n"
            f"    customer: {row['instruction']}\n"
            f"    agent:    {_truncate(row['response'])}"
        )
    return "\n".join(blocks)


@tool(args_schema=ShowExamplesInput)
def show_examples(
    n: int = 5,
    category: str | None = None,
    intent: str | None = None,
) -> str:
    """Show a random sample of example rows, optionally filtered by category
    and/or intent. Each row shows the customer message and a truncated agent
    response.

    Use for "show me N examples of X". For semantic topics, filter by intent
    (e.g., intent='get_refund' for refund-related messages).
    """
    if category is not None and (err := validate_category(category)):
        return err
    if intent is not None and (err := validate_intent(intent)):
        return err

    subset = get_dataframe()
    if category is not None:
        subset = subset[subset["category"] == category]
    if intent is not None:
        subset = subset[subset["intent"] == intent]

    parts = []
    if category is not None:
        parts.append(f"category={category}")
    if intent is not None:
        parts.append(f"intent={intent}")
    scope = f" ({', '.join(parts)})" if parts else ""

    if len(subset) == 0:
        return f"No rows match{scope}."

    sample = subset.sample(min(n, len(subset)))
    header = f"Showing {len(sample)} of {len(subset):,} matching rows{scope}:"
    return f"{header}\n{_format_rows(sample)}"


@tool(args_schema=FindInstructionsByKeywordInput)
def find_instructions_by_keyword(keyword: str, n: int = 5) -> str:
    """Find customer messages containing a LITERAL keyword/substring
    (case-insensitive). Does NOT understand meaning — for semantic/topic queries
    (refunds, complaints, etc.), filter by intent instead.

    Use for literal text needs, e.g. messages mentioning 'order number'.
    """
    df = get_dataframe()
    matches = df[df["instruction"].str.contains(keyword, case=False, na=False, regex=False)]

    if len(matches) == 0:
        return (
            f"No customer messages contain the literal text '{keyword}'. "
            "If you meant a topic (e.g., refunds), try filtering by intent instead."
        )

    sample = matches.head(n)
    header = f"Found {len(matches):,} messages containing '{keyword}'; showing {len(sample)}:"
    return f"{header}\n{_format_rows(sample)}"
