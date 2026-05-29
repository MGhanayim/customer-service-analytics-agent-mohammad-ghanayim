"""Recommender tool (Bonus B): propose relevant next queries.

Given a short summary of the conversation so far and what we know about the
user, the LLM suggests 2-3 concrete follow-up queries the user could run next.
The tool only PROPOSES — the agent's prompt (agent/prompts.py) enforces the
suggest → refine → confirm → execute flow, so nothing runs without the user's
explicit go-ahead (SPEC.md Bonus B).

Layer 2 (tools). Imports the LLM from services/ (Layer 1) — never from agent/.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from services.llm import get_primary_llm
from tools.schemas import SuggestQueryInput

_SUGGEST_SYSTEM_PROMPT = (
    "You help a user explore the Bitext customer-support dataset (11 categories: "
    "ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, "
    "REFUND, SHIPPING, SUBSCRIPTION; 27 intents). Given what the user has done "
    "so far and what we know about them, propose 2-3 SPECIFIC, runnable next "
    "queries about analyzing this data (counts, distributions, examples, or "
    "response summaries). Make them relevant to their demonstrated interests, "
    "not generic. Return a short numbered list, one query each, no preamble."
)


@tool(args_schema=SuggestQueryInput)
def suggest_query(conversation_summary: str, user_profile_summary: str = "") -> str:
    """Suggest 2-3 relevant follow-up queries based on the conversation and user
    profile.

    Use when the user asks "what should I query next?" or wants ideas. This only
    PROPOSES queries — present them to the user and wait for them to pick or
    refine one before actually running anything.
    """
    context = f"Conversation so far: {conversation_summary}"
    if user_profile_summary.strip():
        context += f"\nWhat we know about the user: {user_profile_summary}"

    result = get_primary_llm().invoke(
        [
            SystemMessage(content=_SUGGEST_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
    )
    return f"Suggested next queries:\n{result.content.strip()}"
