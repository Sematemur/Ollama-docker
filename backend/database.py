"""Database helpers for storing and fetching chat history in PostgreSQL."""

from __future__ import annotations

import logging
import time
from typing import Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor

from config import settings

LOGGER = logging.getLogger("chat.database")

_INIT_MAX_RETRIES = 10
_INIT_RETRY_DELAY = 3


def get_connection():
    """Create a PostgreSQL connection with dict-like rows."""
    return psycopg2.connect(settings.database_url, cursor_factory=RealDictCursor)


def init_db():
    """Create tables and indexes, retrying until PostgreSQL is reachable."""
    for attempt in range(1, _INIT_MAX_RETRIES + 1):
        try:
            conn = get_connection()
            try:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS conversations (
                            id SERIAL PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            message TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                    cursor.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)
                        """
                    )
                    conn.commit()
                finally:
                    cursor.close()
            finally:
                conn.close()
            LOGGER.info("PostgreSQL database is ready (attempt %d)", attempt)
            return
        except (psycopg2.OperationalError, psycopg2.DatabaseError) as exc:
            LOGGER.warning(
                "PostgreSQL not ready, retrying in %ds (attempt %d/%d): %s",
                _INIT_RETRY_DELAY, attempt, _INIT_MAX_RETRIES, exc,
            )
            time.sleep(_INIT_RETRY_DELAY)

    raise RuntimeError("Could not connect to PostgreSQL after %d attempts" % _INIT_MAX_RETRIES)


def save_message(session_id: str, role: str, message: str):
    """Persist a single chat message."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO conversations (session_id, role, message) VALUES (%s, %s, %s)",
                (session_id, role, message),
            )
            conn.commit()
        finally:
            cursor.close()
    finally:
        conn.close()


def _rows_to_messages(rows: List[Dict]) -> List[Dict]:
    return [
        {
            "role": row["role"],
            "content": row["message"],
            "created_at": str(row["created_at"]) if row["created_at"] else None,
        }
        for row in rows
    ]


def get_conversation(session_id: str) -> List[Dict]:
    """Fetch full history for a given session."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT role, message, created_at FROM conversations WHERE session_id = %s ORDER BY created_at",
                (session_id,),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        conn.close()

    return _rows_to_messages(rows)


def get_recent_conversation(session_id: str, limit: int = 12) -> List[Dict]:
    """Fetch only recent messages to keep /chat latency predictable."""
    safe_limit = max(1, min(int(limit), 100))
    conn = get_connection()
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT role, message, created_at FROM conversations WHERE session_id = %s ORDER BY created_at DESC LIMIT %s",
                (session_id, safe_limit),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        conn.close()

    rows.reverse()
    return _rows_to_messages(rows)


def get_all_sessions() -> List[str]:
    """List all unique session ids."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT session_id FROM conversations")
            rows = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        conn.close()
    return [row["session_id"] for row in rows]


if __name__ == "__main__":
    init_db()
