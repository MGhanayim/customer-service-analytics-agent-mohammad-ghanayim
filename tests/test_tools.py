"""Tests for the deterministic tools (data_tools + display_tools).

Expected values are computed from the live DataFrame rather than hardcoded, so
the tests verify the tools' LOGIC (correct filtering/counting) and stay valid if
row counts shift slightly. Tools are invoked via .invoke() (the @tool wrapper).
"""

from data.loader import get_dataframe
from tools.data_tools import (
    count_rows,
    filter_by_category,
    filter_by_intent,
    get_distribution,
    list_unique_values,
)
from tools.display_tools import find_instructions_by_keyword, show_examples


# --- count_rows -------------------------------------------------------------

def test_count_rows_entire_dataset() -> None:
    total = len(get_dataframe())
    assert f"{total:,}" in count_rows.invoke({})


def test_count_rows_by_intent_matches_data() -> None:
    df = get_dataframe()
    expected = len(df[df["intent"] == "get_refund"])
    assert f"{expected:,}" in count_rows.invoke({"intent": "get_refund"})


def test_count_rows_combined_filters() -> None:
    df = get_dataframe()
    expected = len(df[(df["category"] == "ORDER") & (df["intent"] == "cancel_order")])
    out = count_rows.invoke({"category": "ORDER", "intent": "cancel_order"})
    assert f"{expected:,}" in out


def test_count_rows_invalid_category_returns_error() -> None:
    out = count_rows.invoke({"category": "refunds"})
    assert "Invalid category" in out
    assert "REFUND" in out  # error lists the valid values


def test_count_rows_invalid_intent_returns_error() -> None:
    out = count_rows.invoke({"intent": "get_money"})
    assert "Invalid intent" in out


# --- filter_by_category / filter_by_intent ----------------------------------

def test_filter_by_category_lists_its_intents() -> None:
    df = get_dataframe()
    out = filter_by_category.invoke({"category": "REFUND"})
    for intent in df[df["category"] == "REFUND"]["intent"].unique():
        assert intent in out


def test_filter_by_intent_reports_correct_parent_category() -> None:
    out = filter_by_intent.invoke({"intent": "get_refund"})
    assert "REFUND" in out  # get_refund's parent category


def test_filter_by_intent_invalid_returns_error() -> None:
    assert "Invalid intent" in filter_by_intent.invoke({"intent": "nope"})


# --- get_distribution -------------------------------------------------------

def test_get_distribution_account_intents() -> None:
    out = get_distribution.invoke({"group_by": "intent", "filter_category": "ACCOUNT"})
    assert "6 distinct values" in out
    assert "edit_account" in out


def test_get_distribution_empty_combo() -> None:
    out = get_distribution.invoke(
        {"group_by": "intent", "filter_category": "REFUND", "filter_intent": "complaint"}
    )
    assert "empty" in out.lower()


def test_get_distribution_flags_is_capped() -> None:
    out = get_distribution.invoke({"group_by": "flags"})
    assert "more values" in out  # 394 flags > MAX_LIST_VALUES, so truncated


# --- list_unique_values -----------------------------------------------------

def test_list_unique_values_categories() -> None:
    out = list_unique_values.invoke({"column": "category"})
    assert "11 unique values" in out
    assert "REFUND" in out


def test_list_unique_values_flags_capped() -> None:
    out = list_unique_values.invoke({"column": "flags"})
    assert "394 unique values" in out
    assert "more" in out


# --- show_examples ----------------------------------------------------------

def test_show_examples_respects_n() -> None:
    out = show_examples.invoke({"n": 3, "intent": "get_refund"})
    assert "Showing 3 of" in out


def test_show_examples_invalid_category() -> None:
    assert "Invalid category" in show_examples.invoke({"category": "bogus"})


def test_show_examples_no_match() -> None:
    # valid category + valid intent that don't co-occur -> zero rows
    out = show_examples.invoke({"category": "REFUND", "intent": "complaint"})
    assert "No rows match" in out


# --- find_instructions_by_keyword -------------------------------------------

def test_find_by_keyword_matches() -> None:
    out = find_instructions_by_keyword.invoke({"keyword": "order number", "n": 2})
    assert "containing 'order number'" in out


def test_find_by_keyword_no_match_guides_to_intent() -> None:
    out = find_instructions_by_keyword.invoke({"keyword": "zzzznomatch"})
    assert "No customer messages" in out
    assert "intent" in out.lower()  # guidance toward intent filtering
