
import sqlite3
from datetime import datetime, timezone

DB_NAME = "chatbot.db"


def get_connection():
    return sqlite3.connect(DB_NAME,check_same_thread=False)


def init_db():

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEX NOT NULL
            )
        """)


def create_thread(thread_id):

    with get_connection() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO conversations
            (thread_id,title, created_at )
            VALUES (?, ?, ?)
        """, (
            thread_id,
            "NewChat",
            datetime.now(timezone.utc).isoformat()
        ))


def update_title(thread_id, title):

    with get_connection() as conn:
        conn.execute("""
            UPDATE conversations
            SET title = ?
            WHERE thread_id = ?
        """, (title, thread_id))


def get_all_threads():

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT thread_id, title
            FROM conversations
            ORDER BY created_at DESC
        """).fetchall()

    return rows

def get_thread_title(thread_id):

    with get_connection() as conn:

        row = conn.execute("""
            SELECT title
            FROM conversations
            WHERE thread_id = ?
        """, (thread_id,)).fetchone()

    return row[0] if row else None