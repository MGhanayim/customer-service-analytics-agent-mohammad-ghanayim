"""Tests for the profile store (semantic memory — deterministic JSON CRUD).

The store is part of the deterministic layer, so its non-trivial merge logic
(name/notes replace, topic union with order-preserving dedup, shallow
preferences merge) and its path-traversal-safe slug are unit-tested here.

Each test redirects USER_PROFILES_DIR to a pytest tmp_path so nothing touches
the real project directory.
"""

import json

import pytest

from services import profile_store


@pytest.fixture(autouse=True)
def _isolated_profiles_dir(tmp_path, monkeypatch):
    """Point the store at a throwaway directory for every test."""
    monkeypatch.setattr(profile_store, "USER_PROFILES_DIR", tmp_path)
    return tmp_path


# --- read / write roundtrip -------------------------------------------------

def test_read_missing_returns_none() -> None:
    assert profile_store.read_profile("nobody") is None


def test_write_then_read_roundtrip() -> None:
    profile = {"name": "Mo", "frequent_topics": ["refunds"], "preferences": {},
               "notes": None, "last_updated": None}
    profile_store.write_profile("u1", profile)
    assert profile_store.read_profile("u1") == profile


def test_corrupt_json_reads_as_absent(_isolated_profiles_dir) -> None:
    (_isolated_profiles_dir / "u1.json").write_text("{not valid json", encoding="utf-8")
    assert profile_store.read_profile("u1") is None


# --- update / merge semantics -----------------------------------------------

def test_update_creates_profile_with_name_and_timestamp() -> None:
    updated = profile_store.update_profile("u1", name="Mohammad")
    assert updated["name"] == "Mohammad"
    assert updated["last_updated"] is not None
    # Persisted, not just returned.
    assert profile_store.read_profile("u1")["name"] == "Mohammad"


def test_name_and_notes_are_replaced() -> None:
    profile_store.update_profile("u1", name="Mo", notes="first")
    updated = profile_store.update_profile("u1", name="Mohammad", notes="second")
    assert updated["name"] == "Mohammad"
    assert updated["notes"] == "second"


def test_frequent_topics_union_dedup_order_preserving() -> None:
    profile_store.update_profile("u1", frequent_topics=["refunds", "shipping"])
    updated = profile_store.update_profile("u1", frequent_topics=["shipping", "orders"])
    # shipping already present → not duplicated; order preserved; new one appended.
    assert updated["frequent_topics"] == ["refunds", "shipping", "orders"]


def test_preferences_shallow_merge() -> None:
    profile_store.update_profile("u1", preferences={"output": "concise", "lang": "en"})
    updated = profile_store.update_profile("u1", preferences={"output": "verbose"})
    # existing key kept, overlapping key overwritten.
    assert updated["preferences"] == {"output": "verbose", "lang": "en"}


def test_update_only_touches_provided_fields() -> None:
    profile_store.update_profile("u1", name="Mo", frequent_topics=["refunds"])
    updated = profile_store.update_profile("u1", notes="just a note")
    assert updated["name"] == "Mo"  # untouched
    assert updated["frequent_topics"] == ["refunds"]  # untouched
    assert updated["notes"] == "just a note"


# --- slug safety ------------------------------------------------------------

def test_slug_blocks_path_traversal(_isolated_profiles_dir) -> None:
    profile_store.update_profile("../../etc/passwd", name="evil")
    # Path separators are stripped, so the write stays INSIDE the profiles dir
    # (no traversal) and lands in a single flat file there.
    written = list(_isolated_profiles_dir.iterdir())
    assert len(written) == 1
    assert "/" not in written[0].name
    assert written[0].resolve().parent == _isolated_profiles_dir.resolve()


def test_slug_empty_falls_back_to_default(_isolated_profiles_dir) -> None:
    profile_store.update_profile("", name="anon")
    assert (_isolated_profiles_dir / "default.json").exists()


def test_written_file_is_valid_json() -> None:
    profile_store.update_profile("u1", name="Mo")
    with (profile_store.USER_PROFILES_DIR / "u1.json").open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["name"] == "Mo"
