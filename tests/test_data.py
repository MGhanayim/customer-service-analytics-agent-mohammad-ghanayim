"""Tests for the data layer: loader shape/columns and vocabulary drift guard.

Deterministic layer only. These run a real HuggingFace fetch on a cold cache
(cached locally afterward), then make pure assertions on the loaded data.
"""

from data.loader import (
    VALID_CATEGORIES,
    VALID_COLUMNS,
    VALID_INTENTS,
    get_categories,
    get_dataframe,
    get_intents,
    verify_vocabulary,
)


def test_dataframe_shape() -> None:
    """The Bitext dataset has the expected row count and column count."""
    df = get_dataframe()
    assert df.shape == (26872, 5)


def test_dataframe_columns() -> None:
    """Columns match the documented schema."""
    df = get_dataframe()
    assert tuple(df.columns) == VALID_COLUMNS


def test_dataframe_is_cached() -> None:
    """get_dataframe returns the same cached object on repeat calls (singleton)."""
    assert get_dataframe() is get_dataframe()


def test_category_count() -> None:
    """Exactly 11 categories, matching the hardcoded constant."""
    assert len(get_categories()) == 11
    assert set(get_categories()) == set(VALID_CATEGORIES)


def test_intent_count() -> None:
    """Exactly 27 intents, matching the hardcoded constant."""
    assert len(get_intents()) == 27
    assert set(get_intents()) == set(VALID_INTENTS)


def test_vocabulary_no_drift() -> None:
    """The drift guard passes: constants are in sync with live data.

    If the dataset version changes and the constants go stale, this fails
    loudly with a message naming the drift.
    """
    verify_vocabulary()  # raises AssertionError on drift
