"""User profile store — semantic memory persisted as one JSON file per user.

This is the SEMANTIC half of the agent's memory (distilled facts about a user:
name, recurring topics, preferences), kept deliberately SEPARATE from the
episodic conversation history (which LangGraph's SqliteSaver owns). Two different
shapes of memory, two different stores — see PLAN.md "Memory Architecture".

Profiles live in ``USER_PROFILES_DIR/{user_id}.json``. All access goes through
the read/write/update helpers here so the on-disk schema stays in one place.

Layer 1 (services). Imports from config (Layer 0) only.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from config import USER_PROFILES_DIR

# The distilled-facts schema. Stored per user; absent keys simply mean "unknown".
_EMPTY_PROFILE: dict[str, Any] = {
    "name": None,
    "frequent_topics": [],
    "preferences": {},
    "notes": None,
    "last_updated": None,
}

# user_id becomes a filename, so restrict it to a safe slug to prevent path
# traversal (e.g. "../../etc/passwd") and illegal filename characters.
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


def _slug(user_id: str) -> str:
    """Sanitize ``user_id`` into a filesystem-safe slug."""
    slug = _SAFE_ID.sub("_", user_id.strip()) or "default"
    return slug[:64]


def _profile_path(user_id: str):
    """Return the JSON path for a user (does not create anything)."""
    return USER_PROFILES_DIR / f"{_slug(user_id)}.json"


def read_profile(user_id: str) -> dict[str, Any] | None:
    """Return the stored profile dict for ``user_id``, or None if none exists yet."""
    path = _profile_path(user_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # A corrupt profile shouldn't crash the agent — treat it as absent.
        return None


def write_profile(user_id: str, profile: dict[str, Any]) -> None:
    """Persist ``profile`` for ``user_id`` (creates the directory if needed)."""
    USER_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    with _profile_path(user_id).open("w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, ensure_ascii=False)


def update_profile(
    user_id: str,
    name: str | None = None,
    frequent_topics: list[str] | None = None,
    preferences: dict[str, str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Merge new facts into a user's profile and persist it.

    Merge semantics (chosen so repeated small updates accumulate rather than
    overwrite):
      - ``name`` / ``notes``: replaced when provided.
      - ``frequent_topics``: union with existing, order-preserving, de-duplicated.
      - ``preferences``: shallow dict merge (new keys win).
    ``last_updated`` is stamped on every write.

    Returns the merged profile.
    """
    profile = read_profile(user_id) or dict(_EMPTY_PROFILE)

    if name is not None:
        profile["name"] = name
    if notes is not None:
        profile["notes"] = notes
    if frequent_topics:
        existing = profile.get("frequent_topics") or []
        merged = list(existing)
        for topic in frequent_topics:
            if topic not in merged:
                merged.append(topic)
        profile["frequent_topics"] = merged
    if preferences:
        current = dict(profile.get("preferences") or {})
        current.update(preferences)
        profile["preferences"] = current

    profile["last_updated"] = datetime.now(timezone.utc).isoformat()
    write_profile(user_id, profile)
    return profile
