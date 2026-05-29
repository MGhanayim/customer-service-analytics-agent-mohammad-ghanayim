"""LLM-powered tool: summarize agent responses for unstructured queries.

Unlike data_tools/display_tools (pure, deterministic), this tool calls the LLM
internally. It samples response texts for a category/intent and asks the model
to describe the patterns, tone, and strategies — answering open-ended questions
like "summarize how agents respond to complaints".

Layer 2 (tools). Imports the LLM from services/ (Layer 1) — NOT from agent/ —
which is exactly why the LLM factory lives in services: both tools and the
agent depend on it from below, avoiding a tools->agent import.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from config import SUMMARY_SAMPLE_SIZE
from data.loader import get_dataframe
from services.llm import get_primary_llm
from tools._validation import validate_category, validate_intent
from tools.schemas import SummarizeResponsesInput

_SUMMARY_SYSTEM_PROMPT = (
    "You are a data analyst summarizing customer-service agent responses. "
    "Given a sample of agent replies, describe the COMMON PATTERNS you observe: "
    "typical tone, recurring phrases, the strategies agents use, and how they "
    "structure their replies. Be concise (a short paragraph or a few bullets). "
    "Summarize only what the sample shows — do not invent details."
)


@tool(args_schema=SummarizeResponsesInput)
def summarize_responses(
    category: str | None = None,
    intent: str | None = None,
) -> str:
    """Summarize the patterns, tone, and strategies in agent RESPONSES for a
    given category and/or intent.

    Use for open-ended/unstructured questions like "summarize how agents respond
    to complaints" or "how do agents typically handle refund requests". Samples
    response texts and uses the LLM to describe what they have in common.
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
    scope = ", ".join(parts) if parts else "the entire dataset"

    if len(subset) == 0:
        return f"No rows match ({scope}); nothing to summarize."

    sample = subset["response"].sample(min(SUMMARY_SAMPLE_SIZE, len(subset))).tolist()
    numbered = "\n\n".join(f"[{i}] {r}" for i, r in enumerate(sample, start=1))

    user_msg = (
        f"Here are {len(sample)} sample agent responses for {scope}. "
        f"Summarize the patterns:\n\n{numbered}"
    )
    result = get_primary_llm().invoke(
        [SystemMessage(content=_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=user_msg)]
    )
    header = f"Summary of agent responses ({scope}, sampled {len(sample)} of {len(subset):,}):"
    return f"{header}\n{result.content.strip()}"
