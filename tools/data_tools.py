"""Deterministic data tools: counting, filtering, distributions, value listing.

Pure functions over the cached DataFrame — no LLM calls, no file I/O. Each is a
LangChain @tool bound to its Pydantic input schema, and each returns a formatted
str (never a raw DataFrame) to keep the LLM's context small.

Category/intent arguments are validated against LIVE data (via tools._validation);
an invalid value returns an error message listing the valid values, which the
agent reads and self-corrects on its next call.

Layer 2 (tools). Imports from config (Layer 0) and data/loader (Layer 1).
Never imports from agent/.
"""

from __future__ import annotations

from langchain_core.tools import tool

from config import MAX_LIST_VALUES
from data.loader import get_dataframe
from tools._validation import validate_category, validate_intent
from tools.schemas import (
    CountRowsInput,
    FilterByCategoryInput,
    FilterByIntentInput,
    GetDistributionInput,
    ListUniqueValuesInput,
)


@tool(args_schema=CountRowsInput)
def count_rows(category: str | None = None, intent: str | None = None) -> str:
    """Count rows in the dataset, optionally filtered by category and/or intent.

    Use for "how many" questions. The two filters combine with AND. Omit both to
    count the entire dataset.
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
    scope = f" matching {', '.join(parts)}" if parts else " (entire dataset)"
    return f"Count: {len(subset):,} rows{scope}."


@tool(args_schema=FilterByCategoryInput)
def filter_by_category(category: str) -> str:
    """Overview of one category: row count, the intents it contains (with counts),
    and a few example customer messages.

    Use for "tell me about the X category". For a precise number alone use
    count_rows; for just the intent breakdown use get_distribution.
    """
    if err := validate_category(category):
        return err

    subset = get_dataframe()
    subset = subset[subset["category"] == category]
    intent_counts = subset["intent"].value_counts().sort_index()

    lines = [f"Category '{category}': {len(subset):,} rows", f"Intents ({len(intent_counts)}):"]
    for intent, cnt in intent_counts.items():
        lines.append(f"  - {intent}: {cnt:,}")
    lines.append("Example messages:")
    for msg in subset["instruction"].head(3):
        lines.append(f"  - {msg}")
    return "\n".join(lines)


@tool(args_schema=FilterByIntentInput)
def filter_by_intent(intent: str) -> str:
    """Overview of one intent: row count, its parent category, and a few example
    customer messages.

    Use for "tell me about the X intent" or to find which category an intent
    belongs to.
    """
    if err := validate_intent(intent):
        return err

    subset = get_dataframe()
    subset = subset[subset["intent"] == intent]
    parent_category = subset["category"].iloc[0]  # intent -> category is 1:1

    lines = [
        f"Intent '{intent}': {len(subset):,} rows",
        f"Parent category: {parent_category}",
        "Example messages:",
    ]
    for msg in subset["instruction"].head(3):
        lines.append(f"  - {msg}")
    return "\n".join(lines)


@tool(args_schema=GetDistributionInput)
def get_distribution(
    group_by: str,
    filter_category: str | None = None,
    filter_intent: str | None = None,
) -> str:
    """Value-count distribution of a column (category, intent, or flags),
    optionally restricted to a category and/or intent first.

    Use for "what is the distribution of X", or to get a full breakdown in one
    call instead of many separate count_rows calls.
    """
    if filter_category is not None and (err := validate_category(filter_category)):
        return err
    if filter_intent is not None and (err := validate_intent(filter_intent)):
        return err

    subset = get_dataframe()
    if filter_category is not None:
        subset = subset[subset["category"] == filter_category]
    if filter_intent is not None:
        subset = subset[subset["intent"] == filter_intent]

    parts = []
    if filter_category is not None:
        parts.append(f"category={filter_category}")
    if filter_intent is not None:
        parts.append(f"intent={filter_intent}")
    scope = f" (filtered: {', '.join(parts)})" if parts else ""

    total = len(subset)
    if total == 0:
        return f"No rows match{scope}; distribution is empty."

    counts = subset[group_by].value_counts()  # sorted by count, descending
    lines = [
        f"Distribution of '{group_by}'{scope} — {total:,} rows across "
        f"{len(counts)} distinct values:"
    ]
    for val, cnt in counts.head(MAX_LIST_VALUES).items():
        lines.append(f"  - {val}: {cnt:,} ({100 * cnt / total:.1f}%)")
    if len(counts) > MAX_LIST_VALUES:
        lines.append(f"  ...and {len(counts) - MAX_LIST_VALUES} more values.")
    return "\n".join(lines)


@tool(args_schema=ListUniqueValuesInput)
def list_unique_values(column: str) -> str:
    """List the unique values present in a column (category, intent, or flags).

    Use to discover what categories or intents exist in the dataset.
    """
    values = sorted(get_dataframe()[column].unique().tolist())
    lines = [f"Column '{column}' has {len(values)} unique values:"]
    for val in values[:MAX_LIST_VALUES]:
        lines.append(f"  - {val}")
    if len(values) > MAX_LIST_VALUES:
        lines.append(f"  ...and {len(values) - MAX_LIST_VALUES} more.")
    return "\n".join(lines)
