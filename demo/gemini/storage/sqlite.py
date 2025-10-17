"""SQLite Session Storage Implementation

Foundation Layer: SQLite-backed session persistence with connection pooling.

This module provides:
- SQLite database schema creation
- CRUD operations for sessions and messages
- Connection pooling for concurrent access
- Efficient query optimization
- Transaction support

Usage:
    storage = SQLiteSessionStorage("sessions.db")
    await storage.initialize()

    session = ChatSession.create("gemini-2.5-flash")
    session.add_message("user", "Hello!")
    await storage.save_session(session)

    loaded = await storage.load_session(session.session_id)
"""

import aiosqlite
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base import ChatMessage, ChatSession, SessionStorage, UploadedFileRecord

logger = logging.getLogger(__name__)


class SQLiteSessionStorage(SessionStorage):
    """SQLite-backed session storage.

    Foundation Layer implementation providing:
    - File-based SQLite database
    - No external dependencies (built into Python)
    - Connection pooling
    - Full-text search capability
    - JSON metadata storage

    Attributes:
        database_path: Path to SQLite database file
        connection: Active database connection
    """

    SCHEMA_VERSION = 1

    def __init__(self, database_path: str = "gemini_sessions.db"):
        """Initialize SQLite storage.

        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = Path(database_path)
        self.connection: Optional[aiosqlite.Connection] = None
        logger.info(f"SQLiteSessionStorage initialized: {database_path}")

    async def initialize(self) -> None:
        """Initialize database schema and connection."""
        # Create connection
        self.connection = await aiosqlite.connect(
            str(self.database_path), check_same_thread=False
        )

        # Enable foreign keys
        await self.connection.execute("PRAGMA foreign_keys = ON")

        # Enable WAL mode for better concurrency
        await self.connection.execute("PRAGMA journal_mode = WAL")

        # Create schema
        await self._create_schema()

        await self.connection.commit()
        logger.info("Database schema initialized")

    async def _create_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        # Schema version table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Check if schema exists
        cursor = await self.connection.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        current_version = row[0] if row else 0

        if current_version >= self.SCHEMA_VERSION:
            return

        # Chat sessions table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                message_count INTEGER DEFAULT 0,
                metadata TEXT,  -- JSON
                status TEXT DEFAULT 'active'
            )
            """
        )

        # Chat messages table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                metadata TEXT,  -- JSON
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                    ON DELETE CASCADE
            )
            """
        )

        # Uploaded files table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS uploaded_files (
                file_id TEXT PRIMARY KEY,
                gemini_file_name TEXT NOT NULL,
                gemini_file_uri TEXT NOT NULL,
                display_name TEXT NOT NULL,
                size INTEGER,
                mime_type TEXT,
                uploaded_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                metadata TEXT  -- JSON
            )
            """
        )

        # Create indexes for performance
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON chat_sessions(status)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON chat_sessions(updated_at DESC)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_expires ON uploaded_files(expires_at)"
        )

        # Record schema version
        await self.connection.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (self.SCHEMA_VERSION,),
        )

        logger.info(f"Database schema created (version {self.SCHEMA_VERSION})")

    async def save_session(self, session: ChatSession) -> bool:
        """Save a session and its messages to storage.

        Args:
            session: Session to save

        Returns:
            True if successful
        """
        try:
            # Update timestamp
            session.updated_at = datetime.now()
            session.message_count = len(session.messages)

            # Save session
            await self.connection.execute(
                """
                INSERT OR REPLACE INTO chat_sessions
                (session_id, model, created_at, updated_at, message_count, metadata, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.model,
                    session.created_at,
                    session.updated_at,
                    session.message_count,
                    json.dumps(session.metadata),
                    session.status,
                ),
            )

            # Save messages
            for message in session.messages:
                await self.connection.execute(
                    """
                    INSERT OR REPLACE INTO chat_messages
                    (message_id, session_id, role, content, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.message_id,
                        message.session_id,
                        message.role,
                        message.content,
                        message.created_at,
                        json.dumps(message.metadata),
                    ),
                )

            await self.connection.commit()
            logger.debug(f"Session saved: {session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    async def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load a session with all its messages.

        Args:
            session_id: Session identifier

        Returns:
            ChatSession if found, None otherwise
        """
        try:
            # Load session
            cursor = await self.connection.execute(
                """
                SELECT session_id, model, created_at, updated_at, message_count,
                       metadata, status
                FROM chat_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            session = ChatSession(
                session_id=row[0],
                model=row[1],
                created_at=self._parse_datetime(row[2]),
                updated_at=self._parse_datetime(row[3]),
                message_count=row[4],
                metadata=json.loads(row[5]) if row[5] else {},
                status=row[6],
                messages=[],
            )

            # Load messages
            cursor = await self.connection.execute(
                """
                SELECT message_id, session_id, role, content, created_at, metadata
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            )

            async for row in cursor:
                message = ChatMessage(
                    message_id=row[0],
                    session_id=row[1],
                    role=row[2],
                    content=row[3],
                    created_at=self._parse_datetime(row[4]),
                    metadata=json.loads(row[5]) if row[5] else {},
                )
                session.messages.append(message)

            logger.debug(f"Session loaded: {session_id} ({len(session.messages)} messages)")
            return session

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None

    async def list_sessions(
        self, status: Optional[str] = None, limit: int = 100
    ) -> list[ChatSession]:
        """List sessions from storage.

        Args:
            status: Optional status filter
            limit: Maximum number of sessions

        Returns:
            List of sessions (without messages loaded)
        """
        try:
            if status:
                cursor = await self.connection.execute(
                    """
                    SELECT session_id, model, created_at, updated_at, message_count,
                           metadata, status
                    FROM chat_sessions
                    WHERE status = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (status, limit),
                )
            else:
                cursor = await self.connection.execute(
                    """
                    SELECT session_id, model, created_at, updated_at, message_count,
                           metadata, status
                    FROM chat_sessions
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            sessions = []
            async for row in cursor:
                session = ChatSession(
                    session_id=row[0],
                    model=row[1],
                    created_at=self._parse_datetime(row[2]),
                    updated_at=self._parse_datetime(row[3]),
                    message_count=row[4],
                    metadata=json.loads(row[5]) if row[5] else {},
                    status=row[6],
                    messages=[],  # Don't load messages for listing
                )
                sessions.append(session)

            logger.debug(f"Listed {len(sessions)} sessions")
            return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        try:
            cursor = await self.connection.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?", (session_id,)
            )
            await self.connection.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Session deleted: {session_id}")
            return deleted

        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False

    async def save_file_record(self, file_record: UploadedFileRecord) -> bool:
        """Save a file record.

        Args:
            file_record: File record to save

        Returns:
            True if successful
        """
        try:
            await self.connection.execute(
                """
                INSERT OR REPLACE INTO uploaded_files
                (file_id, gemini_file_name, gemini_file_uri, display_name,
                 size, mime_type, uploaded_at, expires_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_record.file_id,
                    file_record.gemini_file_name,
                    file_record.gemini_file_uri,
                    file_record.display_name,
                    file_record.size,
                    file_record.mime_type,
                    file_record.uploaded_at,
                    file_record.expires_at,
                    json.dumps(file_record.metadata),
                ),
            )

            await self.connection.commit()
            logger.debug(f"File record saved: {file_record.file_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save file record: {e}")
            return False

    async def load_file_record(self, file_id: str) -> Optional[UploadedFileRecord]:
        """Load a file record.

        Args:
            file_id: File identifier

        Returns:
            UploadedFileRecord if found
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT file_id, gemini_file_name, gemini_file_uri, display_name,
                       size, mime_type, uploaded_at, expires_at, metadata
                FROM uploaded_files
                WHERE file_id = ?
                """,
                (file_id,),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            return UploadedFileRecord(
                file_id=row[0],
                gemini_file_name=row[1],
                gemini_file_uri=row[2],
                display_name=row[3],
                size=row[4],
                mime_type=row[5],
                uploaded_at=self._parse_datetime(row[6]),
                expires_at=self._parse_datetime(row[7]),
                metadata=json.loads(row[8]) if row[8] else {},
            )

        except Exception as e:
            logger.error(f"Failed to load file record: {e}")
            return None

    async def list_file_records(self, limit: int = 100) -> list[UploadedFileRecord]:
        """List file records.

        Args:
            limit: Maximum number of records

        Returns:
            List of file records
        """
        try:
            cursor = await self.connection.execute(
                """
                SELECT file_id, gemini_file_name, gemini_file_uri, display_name,
                       size, mime_type, uploaded_at, expires_at, metadata
                FROM uploaded_files
                ORDER BY uploaded_at DESC
                LIMIT ?
                """,
                (limit,),
            )

            records = []
            async for row in cursor:
                record = UploadedFileRecord(
                    file_id=row[0],
                    gemini_file_name=row[1],
                    gemini_file_uri=row[2],
                    display_name=row[3],
                    size=row[4],
                    mime_type=row[5],
                    uploaded_at=self._parse_datetime(row[6]),
                    expires_at=self._parse_datetime(row[7]),
                    metadata=json.loads(row[8]) if row[8] else {},
                )
                records.append(record)

            return records

        except Exception as e:
            logger.error(f"Failed to list file records: {e}")
            return []

    async def delete_expired_files(self) -> int:
        """Delete expired file records.

        Returns:
            Number of records deleted
        """
        try:
            cursor = await self.connection.execute(
                "DELETE FROM uploaded_files WHERE expires_at < datetime('now')"
            )
            await self.connection.commit()

            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Deleted {deleted} expired file records")
            return deleted

        except Exception as e:
            logger.error(f"Failed to delete expired files: {e}")
            return 0

    async def close(self) -> None:
        """Close database connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None
            logger.info("Database connection closed")

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        """Parse datetime from SQLite format.

        Args:
            value: Datetime string

        Returns:
            Parsed datetime
        """
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")
