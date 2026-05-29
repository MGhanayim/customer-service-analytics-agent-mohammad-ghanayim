"""Singleton loader for the Bitext Customer Service dataset.

Loads the dataset from HuggingFace on first call and caches it for the
lifetime of the process. All tools that need the data import from here.

Thread-safety: `functools.lru_cache` in CPython is thread-safe, so
concurrent first-time calls from multiple threads (e.g., Streamlit
sessions) coordinate correctly — exactly one HuggingFace fetch happens,
all callers receive the same DataFrame.

Layer 1 (data). Imports from config (Layer 0) only.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from datasets import load_dataset

from config import BITEXT_DATASET_ID

# Known dataset vocabulary. Hardcoded as plain tuples so importing this module
# does NOT trigger a dataset load (keeps get_dataframe() lazy). These are the
# "source of truth" for Pydantic field descriptions and tool input validation.
#
# Guarded against drift: the assertions in tools' test step (B.4) verify these
# match the live data via get_categories() / get_intents(). If the dataset
# version changes and these go stale, that test fails loudly — the early warning
# we lacked when SHIPPING_ADDRESS turned out to be SHIPPING.
VALID_COLUMNS: tuple[str, ...] = ("flags", "instruction", "category", "intent", "response")

VALID_CATEGORIES: tuple[str, ...] = (
    "ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE",
    "ORDER", "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION",
)

VALID_INTENTS: tuple[str, ...] = (
    "cancel_order", "change_order", "change_shipping_address",
    "check_cancellation_fee", "check_invoice", "check_payment_methods",
    "check_refund_policy", "complaint", "contact_customer_service",
    "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice",
    "get_refund", "newsletter_subscription", "payment_issue", "place_order",
    "recover_password", "registration_problems", "review",
    "set_up_shipping_address", "switch_account", "track_order", "track_refund",
)


@lru_cache(maxsize=1)
def get_dataframe() -> pd.DataFrame:
    """Return the Bitext dataset as a pandas DataFrame.

    Loads from HuggingFace on first call (~few seconds, ~20 MB on a cold
    cache; instant on a warm one). Subsequent calls return the cached
    DataFrame. Thread-safe.

    Returns:
        DataFrame with columns: flags, instruction, category, intent, response.
        Expected shape: (26872, 5).
    """
    ds = load_dataset(BITEXT_DATASET_ID)["train"]
    return ds.to_pandas()


@lru_cache(maxsize=1)
def get_categories() -> tuple[str, ...]:
    """Return the sorted tuple of unique categories in the dataset."""
    return tuple(sorted(get_dataframe()["category"].unique().tolist()))


@lru_cache(maxsize=1)
def get_intents() -> tuple[str, ...]:
    """Return the sorted tuple of unique intents in the dataset."""
    return tuple(sorted(get_dataframe()["intent"].unique().tolist()))


def verify_vocabulary() -> None:
    """Assert the hardcoded vocabulary constants still match the live dataset.

    Triggers a dataset load (via get_categories/get_intents) on a cold cache.
    Raises AssertionError listing the drift if VALID_CATEGORIES or VALID_INTENTS
    have fallen out of sync with the actual data — the early warning we lacked
    when SHIPPING_ADDRESS turned out to be SHIPPING.

    On failure, a human updates the constants (and any dependent docs/tools);
    this function only alerts, it does not auto-fix.
    """
    live_categories = set(get_categories())
    if set(VALID_CATEGORIES) != live_categories:
        missing = live_categories - set(VALID_CATEGORIES)
        extra = set(VALID_CATEGORIES) - live_categories
        raise AssertionError(
            f"VALID_CATEGORIES drift — in data but not constant: {missing or '{}'}; "
            f"in constant but not data: {extra or '{}'}"
        )

    live_intents = set(get_intents())
    if set(VALID_INTENTS) != live_intents:
        missing = live_intents - set(VALID_INTENTS)
        extra = set(VALID_INTENTS) - live_intents
        raise AssertionError(
            f"VALID_INTENTS drift — in data but not constant: {missing or '{}'}; "
            f"in constant but not data: {extra or '{}'}"
        )
