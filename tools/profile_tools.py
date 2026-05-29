"""Profile tools: let the agent read and update a user's semantic memory.

These wrap services/profile_store (JSON CRUD) as LangChain tools so the agent
can recall facts ("what do you remember about me?") and persist new ones when
the user reveals them ("my name is Mo"). The agent decides WHEN to call these;
the prompt (agent/prompts.py) tells it the policy.

Profiles are stored separately from conversation history (SqliteSaver) — see
SPEC.md 2b.4 / PLAN.md "Memory Architecture".

Layer 2 (tools). Imports from services/profile_store (Layer 1). Never imports
from agent/.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from services.profile_store import read_profile, update_profile
from tools.schemas import GetUserProfileInput, UpdateUserProfileInput


@tool(args_schema=GetUserProfileInput)
def get_user_profile(user_id: str) -> str:
    """Retrieve the stored profile (distilled facts) for the current user.

    Use to answer "what do you remember about me?" or to personalize an answer.
    Returns the profile as readable JSON, or a note if no profile exists yet.
    """
    profile = read_profile(user_id)
    if profile is None:
        return (
            f"No profile stored yet for user '{user_id}'. "
            "Nothing is remembered about this user."
        )
    return f"Profile for '{user_id}':\n{json.dumps(profile, indent=2, ensure_ascii=False)}"


@tool(args_schema=UpdateUserProfileInput)
def update_user_profile(
    user_id: str,
    name: str | None = None,
    frequent_topics: list[str] | None = None,
    preferences: dict[str, str] | None = None,
    notes: str | None = None,
) -> str:
    """Save durable facts about the user to their persistent profile.

    Call this proactively when the user reveals something lasting about
    themselves (name, recurring interests, preferences) — not for one-off query
    parameters. Pass only the fields you learned; they are MERGED into the
    existing profile, not overwritten.
    """
    if not any(v is not None for v in (name, frequent_topics, preferences, notes)):
        return "Nothing to update — provide at least one field (name, topics, etc.)."

    updated = update_profile(
        user_id,
        name=name,
        frequent_topics=frequent_topics,
        preferences=preferences,
        notes=notes,
    )
    return (
        f"Profile for '{user_id}' updated. Current profile:\n"
        f"{json.dumps(updated, indent=2, ensure_ascii=False)}"
    )
