"""Session Persistence Storage Layer

This module provides multi-backend session storage for the Gemini MCP server.

Layers:
- Foundation: SQLite-only session storage with basic CRUD
- Feature: Session lifecycle management with auto-save and migrations
- Polish: PostgreSQL and MongoDB backends with migration tools

Usage:
    # Foundation layer
    from demo.gemini.storage import SQLiteSessionStorage

    storage = SQLiteSessionStorage("sessions.db")
    await storage.save_session(session)

    # Feature layer
    from demo.gemini.storage import SessionManager

    manager = SessionManager(storage)
    await manager.auto_save(session)

    # Polish layer
    from demo.gemini.storage import PostgreSQLSessionStorage, MongoDBSessionStorage

    # Use PostgreSQL
    pg_storage = PostgreSQLSessionStorage("postgresql://localhost/gemini")

    # Use MongoDB
    mongo_storage = MongoDBSessionStorage("mongodb://localhost:27017/gemini")
"""

from .base import ChatMessage, ChatSession, SessionStorage, UploadedFileRecord

# Optional: SQLite storage (requires aiosqlite)
try:
    from .sqlite import SQLiteSessionStorage
    _HAS_SQLITE = True
except ImportError:
    _HAS_SQLITE = False
    SQLiteSessionStorage = None  # type: ignore

__all__ = [
    "SessionStorage",
    "ChatSession",
    "ChatMessage",
    "UploadedFileRecord",
]

if _HAS_SQLITE:
    __all__.append("SQLiteSessionStorage")
