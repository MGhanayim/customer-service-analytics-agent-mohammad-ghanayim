"""Pydantic input schemas for every tool.

Each model defines the arguments one tool accepts. LangChain converts these
into JSON Schema that the LLM reads to know what arguments to send. The
``Field(description=...)`` strings are part of the prompt the LLM sees — they
steer it toward valid values, so they're written carefully.

Layer 2 (tools). Imports the validation vocabulary from data/loader (Layer 1).
Importing those constants does NOT load the dataset (they're plain tuples).
"""

from typing import Literal

from pydantic import BaseModel, Field

from config import MAX_EXAMPLES
from data.loader import VALID_CATEGORIES

# Pre-rendered category list for field descriptions. There are only 11, so
# listing them inline steers the LLM well. (27 intents are too many to list in
# every field — those fields point the LLM at list_unique_values instead.)
_CATEGORIES = ", ".join(VALID_CATEGORIES)


class CountRowsInput(BaseModel):
    """Count rows in the dataset, optionally filtered by category and/or intent."""

    category: str | None = Field(
        default=None,
        description=(
            f"Filter to this category before counting. Valid categories: {_CATEGORIES}. "
            "Omit to count across all categories."
        ),
    )
    intent: str | None = Field(
        default=None,
        description=(
            "Filter to this intent before counting (e.g., 'get_refund'). "
            "Use list_unique_values(column='intent') to see all valid intents. "
            "Omit to not filter by intent."
        ),
    )


class FilterByCategoryInput(BaseModel):
    """Summarize the rows belonging to a single category."""

    category: str = Field(
        description=(
            f"The category to filter by. Valid categories: {_CATEGORIES}."
        ),
    )


class FilterByIntentInput(BaseModel):
    """Summarize the rows belonging to a single intent."""

    intent: str = Field(
        description=(
            "The intent to filter by (e.g., 'get_refund', 'complaint'). "
            "Use list_unique_values(column='intent') to see all valid intents."
        ),
    )


class GetDistributionInput(BaseModel):
    """Get the value-count distribution of a column, with optional filtering."""

    group_by: Literal["category", "intent", "flags"] = Field(
        description=(
            "Which column to compute the distribution over. One of: "
            "'category', 'intent', 'flags'."
        ),
    )
    filter_category: str | None = Field(
        default=None,
        description=(
            f"Restrict to this category before computing the distribution. "
            f"Valid categories: {_CATEGORIES}. Omit for no category filter."
        ),
    )
    filter_intent: str | None = Field(
        default=None,
        description=(
            "Restrict to this intent before computing the distribution. "
            "Omit for no intent filter."
        ),
    )


class ListUniqueValuesInput(BaseModel):
    """List the unique values present in a column."""

    column: Literal["category", "intent", "flags"] = Field(
        description=(
            "Which column's unique values to list. One of: "
            "'category' (11 values), 'intent' (27 values), 'flags'."
        ),
    )


class ShowExamplesInput(BaseModel):
    """Show a sample of example rows, optionally filtered by category and/or intent."""

    n: int = Field(
        default=5,
        ge=1,
        le=MAX_EXAMPLES,
        description=f"How many example rows to show (1-{MAX_EXAMPLES}).",
    )
    category: str | None = Field(
        default=None,
        description=(
            f"Restrict examples to this category. Valid categories: {_CATEGORIES}. "
            "Omit to sample from all categories."
        ),
    )
    intent: str | None = Field(
        default=None,
        description=(
            "Restrict examples to this intent (e.g., 'get_refund'). "
            "Omit to sample from all intents."
        ),
    )


class SummarizeResponsesInput(BaseModel):
    """Summarize the patterns in agent RESPONSES for a category and/or intent.

    The sample size is an internal system setting, not an argument — the LLM
    only chooses what to summarize (category/intent), not how many to sample.
    """

    category: str | None = Field(
        default=None,
        description=(
            f"Restrict to this category before summarizing. Valid categories: "
            f"{_CATEGORIES}. Omit to summarize across all categories."
        ),
    )
    intent: str | None = Field(
        default=None,
        description=(
            "Restrict to this intent before summarizing (e.g., 'complaint'). "
            "Omit to not filter by intent."
        ),
    )


class FindInstructionsByKeywordInput(BaseModel):
    """Find customer messages containing a LITERAL keyword/substring.

    Does NOT understand meaning — only exact (case-insensitive) text matching.
    For semantic/topic queries (refunds, complaints, etc.), filter by intent instead.
    """

    keyword: str = Field(
        description=(
            "Case-insensitive substring to match within the 'instruction' column "
            "(the customer's message), e.g. 'order number', 'cancel'. This is "
            "LITERAL text matching, NOT semantic search: 'money back' will not "
            "match messages that say 'refund'. For topic/meaning queries, filter "
            "by intent instead (e.g., show_examples(intent='get_refund'))."
        ),
    )
    n: int = Field(
        default=5,
        ge=1,
        le=MAX_EXAMPLES,
        description=f"Maximum number of matching rows to return (1-{MAX_EXAMPLES}).",
    )


# ----------------------------------------------------------------------------
# Profile tools (semantic memory — Task 2b)
# ----------------------------------------------------------------------------


class GetUserProfileInput(BaseModel):
    """Read the stored profile (distilled facts) for the current user."""

    user_id: str = Field(
        description=(
            "The id of the user whose profile to read. Use the exact user_id "
            "given to you in the system prompt — do not invent one."
        ),
    )


class UpdateUserProfileInput(BaseModel):
    """Save distilled facts about the user to their persistent profile.

    Call this when the user reveals something durable about themselves (their
    name, a recurring interest, a stated preference). Pass ONLY the fields you
    learned; omit the rest — they are merged into the existing profile, not
    overwritten wholesale.
    """

    user_id: str = Field(
        description=(
            "The id of the user to update. Use the exact user_id given to you in "
            "the system prompt — do not invent one."
        ),
    )
    name: str | None = Field(
        default=None,
        description="The user's name, if they shared it. Omit if unknown.",
    )
    frequent_topics: list[str] | None = Field(
        default=None,
        description=(
            "Topics/categories the user keeps asking about (e.g. ['refunds', "
            "'shipping']). Added to any existing topics. Omit if none new."
        ),
    )
    preferences: dict[str, str] | None = Field(
        default=None,
        description=(
            "Stated preferences as key->value strings (e.g. {'output': 'concise', "
            "'examples_per_query': '3'}). Merged into existing preferences. Omit "
            "if none."
        ),
    )
    notes: str | None = Field(
        default=None,
        description="Any other durable free-text note about the user. Omit if none.",
    )


# ----------------------------------------------------------------------------
# Recommender tool (Bonus B)
# ----------------------------------------------------------------------------


class SuggestQueryInput(BaseModel):
    """Propose relevant follow-up queries grounded in the conversation + profile."""

    conversation_summary: str = Field(
        description=(
            "A short summary of what the user has asked so far in this "
            "conversation, written by you from the message history."
        ),
    )
    user_profile_summary: str = Field(
        default="",
        description=(
            "A short summary of what you know about the user from their profile "
            "(name, frequent topics, preferences). Pass an empty string if no "
            "profile is available."
        ),
    )
