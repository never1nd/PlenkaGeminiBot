from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

from bot.config import Settings
from bot.model_utils import DEFAULT_MODEL_KEY

logger = logging.getLogger("bot")

_DEFAULT_USER_ID = 0


class Database:
    """Thread-safe SQLite database for chat history and user data."""

    def __init__(self, settings: Settings) -> None:
        self._path = settings.users_db_file
        self._busy_ms = settings.sqlite_busy_timeout_ms
        self._timeout = max(1.0, settings.sqlite_busy_timeout_ms / 1000.0)
        self._local = threading.local()
        self._init_schema()

    # ── connection ──────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(self._path, timeout=self._timeout)
        conn.execute(f"PRAGMA busy_timeout = {self._busy_ms}")
        self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        c = self._conn()
        c.execute("PRAGMA journal_mode = WAL")
        c.execute("PRAGMA synchronous = NORMAL")
        with c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS allowed_users ("
                "user_id INTEGER PRIMARY KEY, added_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS chat_history ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL, "
                "user_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, "
                "created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS memory_settings ("
                "chat_id INTEGER NOT NULL, user_id INTEGER NOT NULL, "
                "enabled INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(chat_id, user_id))"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS chat_summary ("
                "chat_id INTEGER PRIMARY KEY, summary TEXT NOT NULL, "
                "updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_chat "
                "ON chat_history(chat_id, id DESC)"
            )
            self._migrate_history_columns(c)
            self._migrate_memory(c)

    def _migrate_history_columns(self, c: sqlite3.Connection) -> None:
        cols = {str(r[1]).lower() for r in c.execute("PRAGMA table_info(chat_history)")}
        if "content_enc" in cols and "content" not in cols:
            c.execute("ALTER TABLE chat_history RENAME COLUMN content_enc TO content")

    def _migrate_memory(self, c: sqlite3.Connection) -> None:
        cols = {str(r[1]).lower() for r in c.execute("PRAGMA table_info(memory_settings)")}
        if cols and "user_id" not in cols:
            c.execute("ALTER TABLE memory_settings RENAME TO _legacy_mem")
            c.execute(
                "CREATE TABLE memory_settings ("
                "chat_id INTEGER NOT NULL, user_id INTEGER NOT NULL, "
                "enabled INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(chat_id, user_id))"
            )
            c.execute(
                "INSERT INTO memory_settings(chat_id, user_id, enabled) "
                "SELECT chat_id, ?, enabled FROM _legacy_mem",
                (_DEFAULT_USER_ID,),
            )
            c.execute("DROP TABLE _legacy_mem")

    # ── allowlist ───────────────────────────────────────────────────

    def add_allowed(self, user_id: int) -> None:
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO allowed_users(user_id) VALUES (?)", (user_id,))

    def remove_allowed(self, user_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM allowed_users WHERE user_id = ?", (user_id,))

    def load_allowed_ids(self) -> set[int]:
        rows = self._conn().execute("SELECT user_id FROM allowed_users").fetchall()
        return {int(r[0]) for r in rows}

    def list_allowed(self) -> list[tuple[int, str]]:
        rows = self._conn().execute(
            "SELECT user_id, added_at FROM allowed_users ORDER BY added_at DESC"
        ).fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    # ── memory settings ────────────────────────────────────────────

    def is_memory_enabled(self, chat_id: int, user_id: int) -> bool:
        c = self._conn()
        row = c.execute(
            "SELECT enabled FROM memory_settings WHERE chat_id=? AND user_id=?",
            (chat_id, user_id),
        ).fetchone()
        if row is not None:
            return bool(int(row[0]))
        default = c.execute(
            "SELECT enabled FROM memory_settings WHERE chat_id=? AND user_id=?",
            (chat_id, _DEFAULT_USER_ID),
        ).fetchone()
        return bool(int(default[0])) if default else True

    def set_memory_enabled(self, chat_id: int, user_id: int, enabled: bool) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO memory_settings(chat_id, user_id, enabled) VALUES(?,?,?) "
                "ON CONFLICT(chat_id, user_id) DO UPDATE SET enabled=excluded.enabled",
                (chat_id, user_id, int(enabled)),
            )

    # ── chat history ────────────────────────────────────────────────

    def save_messages(self, chat_id: int, user_id: int, messages: list[tuple[str, str]]) -> None:
        rows = [
            (chat_id, user_id, role, text)
            for role, text in messages if role in ("user", "assistant")
        ]
        if not rows:
            return
        with self._conn() as c:
            c.executemany(
                "INSERT INTO chat_history(chat_id, user_id, role, content) VALUES (?,?,?,?)",
                rows,
            )

    def clear_history(self, chat_id: int, *, clear_summary: bool = True) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM chat_history WHERE chat_id=?", (chat_id,))
            if clear_summary:
                c.execute("DELETE FROM chat_summary WHERE chat_id=?", (chat_id,))

    def recent_history(self, chat_id: int, limit: int) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        rows = self._conn().execute(
            "SELECT role, content FROM chat_history WHERE chat_id=? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [
            {"role": str(r), "content": str(c)}
            for r, c in reversed(rows) if str(c).strip()
        ]

    def history_count(self, chat_id: int) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) FROM chat_history WHERE chat_id=?",
            (chat_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def all_history(self, chat_id: int) -> list[dict[str, str]]:
        rows = self._conn().execute(
            "SELECT role, content FROM chat_history WHERE chat_id=? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
        return [{"role": str(r), "content": str(c)} for r, c in rows if str(c).strip()]

    def get_summary(self, chat_id: int) -> str:
        row = self._conn().execute(
            "SELECT summary FROM chat_summary WHERE chat_id=?",
            (chat_id,),
        ).fetchone()
        return str(row[0]) if row and str(row[0]).strip() else ""

    def set_summary(self, chat_id: int, summary: str) -> None:
        text = str(summary or "").strip()
        if not text:
            return
        with self._conn() as c:
            c.execute(
                "INSERT INTO chat_summary(chat_id, summary) VALUES(?, ?) "
                "ON CONFLICT(chat_id) DO UPDATE SET summary=excluded.summary, "
                "updated_at=CURRENT_TIMESTAMP",
                (chat_id, text),
            )


class AllowList:
    """In-memory allow list backed by the database."""

    def __init__(self, db: Database, settings: Settings) -> None:
        self._db = db
        self._settings = settings
        self._allowed: set[int] = set()
        self._path = settings.allowlist_file
        self._migrate_legacy()
        self._reload()

    def _migrate_legacy(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for raw in data.get("user_ids", []):
                try:
                    self._db.add_allowed(int(raw))
                except (ValueError, TypeError):
                    pass
        except Exception as exc:
            logger.warning("Legacy allowlist migration failed: %s", exc)

    def _reload(self) -> None:
        ids = self._db.load_allowed_ids()
        ids.update(self._settings.owner_ids)
        self._allowed = ids
        self._save_backup()

    def _save_backup(self) -> None:
        data = {"user_ids": sorted(self._allowed)}
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def is_allowed(self, user_id: int) -> bool:
        return user_id in self._allowed

    def add(self, user_id: int) -> None:
        self._db.add_allowed(user_id)
        self._allowed.add(user_id)
        self._save_backup()

    def remove(self, user_id: int) -> None:
        self._db.remove_allowed(user_id)
        self._allowed.discard(user_id)
        self._save_backup()

    def list_users(self) -> list[tuple[int, str]]:
        return self._db.list_allowed()


class ModelPrefs:
    """Per-user model preference stored in a JSON file."""

    def __init__(self, path: Path, valid_keys: set[str]) -> None:
        self._path = path
        self._valid: set[str] = set(valid_keys)
        self._valid.add(DEFAULT_MODEL_KEY)
        self._prefs: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8")).get("prefs", {})
            if isinstance(raw, dict):
                self._prefs = {
                    str(k): str(v) for k, v in raw.items() if str(v) in self._valid
                }
        except Exception as exc:
            logger.warning("Model prefs load failed: %s", exc)

    def _save(self) -> None:
        self._path.write_text(
            json.dumps({"prefs": self._prefs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, user_id: int, default: str) -> str:
        key = self._prefs.get(str(user_id), default)
        return key if key in self._valid else default

    def has(self, user_id: int) -> bool:
        return self._prefs.get(str(user_id), "") in self._valid

    def set(self, user_id: int, key: str) -> None:
        if key not in self._valid:
            raise ValueError("Invalid model key")
        self._prefs[str(user_id)] = key
        self._save()

    def update_valid_keys(self, keys: set[str]) -> None:
        self._valid = set(keys)
        self._valid.add(DEFAULT_MODEL_KEY)
