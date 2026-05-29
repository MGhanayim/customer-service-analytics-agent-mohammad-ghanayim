"""Centralized configuration for the Customer Service Analytics Agent.

Layer 0 in the architecture — has no project imports, everything else imports from here.

Reads secrets from `.env` (via python-dotenv) and exposes them alongside
non-secret settings (model names, paths, tuning knobs).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# Environment / secrets
# ----------------------------------------------------------------------------

# Load .env into os.environ. No-op if .env doesn't exist (production env vars
# might be set another way, e.g., systemd / docker / CI).
load_dotenv()

NEBIUS_API_KEY: str | None = os.getenv("NEBIUS_API_KEY")

# ----------------------------------------------------------------------------
# Nebius connection
# ----------------------------------------------------------------------------

# Token Factory endpoint (OpenAI-compatible). We pin this explicitly rather than
# rely on langchain-nebius's default, which points at the LEGACY studio.nebius.ai
# URL. Both currently mirror the same models, but the official docs recommend the
# Token Factory endpoint, so pinning it future-proofs against the legacy URL being
# retired.
NEBIUS_BASE_URL: str = "https://api.tokenfactory.nebius.com/v1/"

# ----------------------------------------------------------------------------
# Model selection (Nebius Token Factory)
# ----------------------------------------------------------------------------

# Both roles use the same model: Qwen3-30B-A3B-Instruct-2507.
#   - Instruct (NOT Thinking) variant: emits clean output, no <think> tags to
#     strip — important since our ReAct loop already provides the reasoning
#     structure, so the model's internal chain-of-thought would be redundant
#     and would pollute tool output / summaries.
#   - MoE with only ~3B active params: fast and cheap despite 30B total.
#   - Verified to do clean tool calling.
# We keep separate PRIMARY/ROUTER constants (and factory functions) so a cheaper
# router model can be swapped in later without touching the primary — the code
# already supports a two-tier split; we just point both at one capable model now.
PRIMARY_MODEL: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ROUTER_MODEL: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Temperature: 0.0 for the router (deterministic classification); slightly higher
# for the primary so summaries aren't robotic, but still low for reliable tool calls.
PRIMARY_TEMPERATURE: float = 0.2
ROUTER_TEMPERATURE: float = 0.0

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

BITEXT_DATASET_ID: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# ----------------------------------------------------------------------------
# Paths (resolved relative to this file so they work from any CWD)
# ----------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).parent.resolve()

# SQLite DB for episodic memory (LangGraph checkpointer)
CONVERSATIONS_DB_PATH: Path = PROJECT_ROOT / "conversations.db"

# Per-user profile directory (JSON files, semantic memory)
USER_PROFILES_DIR: Path = PROJECT_ROOT / "user_profiles"

# ----------------------------------------------------------------------------
# Agent tuning knobs
# ----------------------------------------------------------------------------

# Soft iteration limit: when iteration_count >= this, agent routes to fallback_node
# (graceful "I couldn't resolve this" message). See SPEC.md 1.5.
ITERATION_LIMIT: int = 12

# Hard limit: passed to graph.invoke(config={"recursion_limit": ...}) as a safety
# net. Catches infinite loops the soft limit might miss. LangGraph 1.x: this is
# an invoke-time config, NOT a .compile() kwarg.
RECURSION_LIMIT: int = 25

# ----------------------------------------------------------------------------
# Tool output bounds (keep tool results token-cheap and context-safe)
# ----------------------------------------------------------------------------

# Max rows a single show_examples / search_instructions call may return.
# Enforced at the Pydantic schema level (Field(le=MAX_EXAMPLES)) so the LLM
# cannot request more — the request is rejected before the tool body runs.
MAX_EXAMPLES: int = 20

# The `response` column reaches ~2,472 chars. When showing rows, truncate each
# response preview to this many characters (with an ellipsis) to keep tool
# output small. `instruction` (max 92 chars) is never truncated.
PREVIEW_CHARS: int = 250

# Max distinct values list_unique_values / get_distribution will print before
# truncating with a "...and N more" note. category (11) and intent (27) fit
# comfortably; `flags` has 394 distinct values, which this cap keeps in check.
MAX_LIST_VALUES: int = 30

# summarize_responses samples this many response texts to feed the LLM. This is
# an INTERNAL system choice, not an LLM-facing parameter — the model has no basis
# to pick a good sample size, so we fix it here. Enough to reveal patterns, few
# enough to keep the prompt cheap (responses average ~634 chars).
SUMMARY_SAMPLE_SIZE: int = 10
