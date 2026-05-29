"""LLM factory — constructs the Nebius chat models used across the project.

Two roles, currently backed by the SAME model (Qwen3-30B-A3B-Instruct-2507):
  - primary (temp 0.2): agent reasoning, tool calling, summarization
  - router  (temp 0.0): deterministic query classification

Separate factory functions are kept so a cheaper router model can be swapped in
later without touching the primary. Instances are cached (one per role) so
repeated calls reuse the same client.

Layer 1 (services). Imported by agent/ (Layer 3) and tools/summary_tools.py
(Layer 2) — both depend on this shared lower layer, avoiding any tools->agent
import. Imports from config (Layer 0) only.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_nebius import ChatNebius

from config import (
    NEBIUS_API_KEY,
    NEBIUS_BASE_URL,
    PRIMARY_MODEL,
    PRIMARY_TEMPERATURE,
    ROUTER_MODEL,
    ROUTER_TEMPERATURE,
)


def _require_key() -> str:
    """Return the API key, or raise a clear error if it's missing.

    config import succeeds without a key (so tool/unit tests that never touch
    the LLM still work); the failure is deferred to the moment we actually try
    to build a model.
    """
    if not NEBIUS_API_KEY:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Copy .env.example to .env and add your "
            "key (get one at https://tokenfactory.nebius.com/settings/api-keys)."
        )
    return NEBIUS_API_KEY


@lru_cache(maxsize=1)
def get_primary_llm() -> ChatNebius:
    """Primary model — agent reasoning, tool calling, and summarization."""
    return ChatNebius(
        model=PRIMARY_MODEL,
        api_key=_require_key(),
        base_url=NEBIUS_BASE_URL,
        temperature=PRIMARY_TEMPERATURE,
    )


@lru_cache(maxsize=1)
def get_router_llm() -> ChatNebius:
    """Router model — fast, cheap query classification."""
    return ChatNebius(
        model=ROUTER_MODEL,
        api_key=_require_key(),
        base_url=NEBIUS_BASE_URL,
        temperature=ROUTER_TEMPERATURE,
    )
