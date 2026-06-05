import sqlite3
import json
import sqlite3
import argparse
import sys

from pathlib import Path
from datetime import datetime
from typing import Optional

DB_PATH = Path(__file__).parent / "progress.db"

def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init():
    """Erstellt Tabellen falls noch nicht vorhanden."""
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                filepath    TEXT NOT NULL,
                text        TEXT NOT NULL,
                spans_json  TEXT NOT NULL DEFAULT '[]',
                status      TEXT NOT NULL DEFAULT 'in_progress',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        conn.commit()


def save_session(filename: str, filepath: str, text: str,
                    spans: list, status: str = "in_progress") -> int:
    now = datetime.now().isoformat()
    with connect() as conn:
        # Bestehende Session für diese Datei aktualisieren oder neu anlegen
        row = conn.execute(
            "SELECT id FROM sessions WHERE filepath = ?", (filepath,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE sessions SET spans_json=?, status=?, updated_at=? WHERE id=?",
                (json.dumps(spans, ensure_ascii=False), status, now, row["id"])
            )
            conn.commit()
            return row["id"]
        else:
            cur = conn.execute(
                "INSERT INTO sessions (filename, filepath, text, spans_json, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (filename, filepath, text,
                 json.dumps(spans, ensure_ascii=False), status, now, now)
            )
            conn.commit()
            return cur.lastrowid


def load_session(filepath: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE filepath = ?", (filepath,)
        ).fetchone()
        if row:
            d = dict(row)
            d["spans"] = json.loads(d["spans_json"])
            return d
    return None


def list_sessions() -> list:
    with connect() as conn:
        rows = conn.execute(
            "SELECT filename, filepath, status, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_session(filepath: str):
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE filepath = ?", (filepath,))
        conn.commit()


def execute_sql(statement):
    with connect() as conn:   
        conn.execute(statement)
        conn.commit()
        
        print(f"Success! Rows affected: {conn.rowcount}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute SQL on your local DB.")
    parser.add_argument("sql", type=str, help="The SQL statement to execute (e.g., 'DELETE FROM table WHERE id=1')")
    
    args = parser.parse_args()
    execute_sql(args.sql)