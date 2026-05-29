"""Shared input validation for tools.

Five of the seven data/display tools accept a category and/or intent. Rather
than duplicate the "is this a real value?" check (and rather than couple
data_tools to display_tools), both import these helpers.

Each validator returns an error MESSAGE (str) if the value is invalid, or None
if it's valid. The error message lists the valid values — when the LLM sends a
bad value, it reads this and self-corrects on the next tool call.

Validation is against LIVE data (get_categories / get_intents), so it is always
accurate even if a hardcoded constant or description drifts.

Layer 2 (tools). Imports from data/loader (Layer 1).
"""

from __future__ import annotations

from data.loader import get_categories, get_intents


def validate_category(category: str) -> str | None:
    """Return an error message if ``category`` is not a real category, else None."""
    valid = get_categories()
    if category not in valid:
        return (
            f"Invalid category '{category}'. Valid categories: {', '.join(valid)}."
        )
    return None


def validate_intent(intent: str) -> str | None:
    """Return an error message if ``intent`` is not a real intent, else None."""
    valid = get_intents()
    if intent not in valid:
        return (
            f"Invalid intent '{intent}'. Valid intents: {', '.join(valid)}."
        )
    return None
