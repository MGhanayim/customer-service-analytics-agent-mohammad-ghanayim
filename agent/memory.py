"""Episodic memory: the LangGraph checkpointer factory (Task 2a).

A ``SqliteSaver`` persists the full graph state (the message history) keyed by
``thread_id`` (our ``--session`` id). Because it writes to a SQLite file, the
conversation survives process restarts — re-running with the same session id
resumes exactly where it left off. This is the EPISODIC half of memory; the
SEMANTIC half (user profiles) lives in services/profile_store.py.

Layer 3 (agent). Imports from config (Layer 0) only.

API note (langgraph-checkpoint-sqlite 3.x): we build the SqliteSaver from a
manual sqlite3 connection with ``check_same_thread=False`` so the same
checkpointer can be used across threads (Streamlit, CLI).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from config import CONVERSATIONS_DB_PATH


def get_checkpointer(db_path: Path | str = CONVERSATIONS_DB_PATH) -> SqliteSaver:
    """Build a SqliteSaver checkpointer backed by a SQLite file.

    Args:
        db_path: Path to the SQLite database file. Defaults to the configured
            conversations.db in the project root. Created if it doesn't exist.

    Returns:
        A SqliteSaver ready to pass to ``StateGraph.compile(checkpointer=...)``.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)
