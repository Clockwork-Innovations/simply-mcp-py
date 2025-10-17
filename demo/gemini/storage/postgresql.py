"""PostgreSQL Session Storage Implementation - Polish Layer

Enterprise-grade session storage with PostgreSQL backend.

Features:
- Connection pooling with asyncpg
- Advanced query capabilities
- JSONB for efficient metadata storage
- Full-text search support
- Concurrent access handling
- Transaction support

Usage:
    storage = PostgreSQLSessionStorage(
        "postgresql://user:pass@localhost/gemini"
    )
    await storage.initialize()

    session = ChatSession.create("gemini-2.5-flash")
    await storage.save_session(session)

Requirements:
    pip install asyncpg
"""

import json
import logging
from datetime import datetime
from typing import Optional

from typing import TYPE_CHECKING

try:
    import asyncpg
    _HAS_ASYNCPG = True
except ImportError:
    asyncpg = None  # type: ignore
    _HAS_ASYNCPG = False

from .base import ChatMessage, ChatSession, SessionStorage, UploadedFileRecord

if TYPE_CHECKING and asyncpg:
    from asyncpg import Connection, Pool

logger = logging.getLogger(__name__)


class PostgreSQLSessionStorage(SessionStorage):
    """PostgreSQL-backed session storage.

    Polish Layer implementation providing:
    - Enterprise-grade relational database
    - Connection pooling for performance
    - JSONB for flexible metadata
    - Advanced indexing and query optimization
    - Full ACID transaction support

    Attributes:
        connection_string: PostgreSQL connection string
        pool: Connection pool
        min_pool_size: Minimum connections in pool
        max_pool_size: Maximum connections in pool
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        connection_string: str,
        min_pool_size: int = 5,
        max_pool_size: int = 20,
    ):
        """Initialize PostgreSQL storage.

        Args:
            connection_string: PostgreSQL connection string
            min_pool_size: Minimum connections in pool
            max_pool_size: Maximum connections in pool

        Raises:
            ImportError: If asyncpg is not installed
        """
        if asyncpg is None:
            raise ImportError(
                "asyncpg is required for PostgreSQL support. "
                "Install with: pip install asyncpg"
            )

        self.connection_string = connection_string
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.pool: Optional["asyncpg.Pool"] = None
        logger.info(f"PostgreSQLSessionStorage initialized: {connection_string}")

    async def initialize(self) -> None:
        """Initialize connection pool and schema."""
        # Create connection pool
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=self.min_pool_size,
            max_size=self.max_pool_size,
        )

        # Create schema
        async with self.pool.acquire() as conn:
            await self._create_schema(conn)

        logger.info("PostgreSQL connection pool and schema initialized")

    async def _create_schema(self, conn: "asyncpg.Connection") -> None:
        """Create database schema if it doesn't exist."""
        # Schema version table
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Check current version
        row = await conn.fetchrow(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        current_version = row["version"] if row else 0

        if current_version >= self.SCHEMA_VERSION:
            return

        # Chat sessions table with JSONB
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                message_count INTEGER DEFAULT 0,
                metadata JSONB,
                status TEXT DEFAULT 'active'
            )
            """
        )

        # Chat messages table
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id)
                    ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                metadata JSONB
            )
            """
        )

        # Uploaded files table
        await conn.execute(
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
                metadata JSONB
            )
            """
        )

        # Create indexes for performance
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON chat_sessions(status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON chat_sessions(updated_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_metadata ON chat_sessions USING gin(metadata)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_created ON chat_messages(created_at)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_expires ON uploaded_files(expires_at)"
        )

        # Record schema version
        await conn.execute(
            "INSERT INTO schema_version (version) VALUES ($1) ON CONFLICT (version) DO NOTHING",
            self.SCHEMA_VERSION,
        )

        logger.info(f"PostgreSQL schema created (version {self.SCHEMA_VERSION})")

    async def save_session(self, session: ChatSession) -> bool:
        """Save session and messages."""
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Update timestamp
                    session.updated_at = datetime.now()
                    session.message_count = len(session.messages)

                    # Upsert session
                    await conn.execute(
                        """
                        INSERT INTO chat_sessions
                        (session_id, model, created_at, updated_at, message_count, metadata, status)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (session_id) DO UPDATE SET
                            updated_at = EXCLUDED.updated_at,
                            message_count = EXCLUDED.message_count,
                            metadata = EXCLUDED.metadata,
                            status = EXCLUDED.status
                        """,
                        session.session_id,
                        session.model,
                        session.created_at,
                        session.updated_at,
                        session.message_count,
                        json.dumps(session.metadata),
                        session.status,
                    )

                    # Upsert messages
                    for message in session.messages:
                        await conn.execute(
                            """
                            INSERT INTO chat_messages
                            (message_id, session_id, role, content, created_at, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (message_id) DO UPDATE SET
                                content = EXCLUDED.content,
                                metadata = EXCLUDED.metadata
                            """,
                            message.message_id,
                            message.session_id,
                            message.role,
                            message.content,
                            message.created_at,
                            json.dumps(message.metadata),
                        )

            logger.debug(f"Session saved: {session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    async def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load session with all messages."""
        try:
            async with self.pool.acquire() as conn:
                # Load session
                row = await conn.fetchrow(
                    """
                    SELECT session_id, model, created_at, updated_at, message_count,
                           metadata, status
                    FROM chat_sessions
                    WHERE session_id = $1
                    """,
                    session_id,
                )

                if not row:
                    return None

                session = ChatSession(
                    session_id=row["session_id"],
                    model=row["model"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    message_count=row["message_count"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    status=row["status"],
                    messages=[],
                )

                # Load messages
                rows = await conn.fetch(
                    """
                    SELECT message_id, session_id, role, content, created_at, metadata
                    FROM chat_messages
                    WHERE session_id = $1
                    ORDER BY created_at ASC
                    """,
                    session_id,
                )

                for row in rows:
                    message = ChatMessage(
                        message_id=row["message_id"],
                        session_id=row["session_id"],
                        role=row["role"],
                        content=row["content"],
                        created_at=row["created_at"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
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
        """List sessions from storage."""
        try:
            async with self.pool.acquire() as conn:
                if status:
                    rows = await conn.fetch(
                        """
                        SELECT session_id, model, created_at, updated_at, message_count,
                               metadata, status
                        FROM chat_sessions
                        WHERE status = $1
                        ORDER BY updated_at DESC
                        LIMIT $2
                        """,
                        status,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT session_id, model, created_at, updated_at, message_count,
                               metadata, status
                        FROM chat_sessions
                        ORDER BY updated_at DESC
                        LIMIT $1
                        """,
                        limit,
                    )

                sessions = []
                for row in rows:
                    session = ChatSession(
                        session_id=row["session_id"],
                        model=row["model"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        message_count=row["message_count"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        status=row["status"],
                        messages=[],
                    )
                    sessions.append(session)

                logger.debug(f"Listed {len(sessions)} sessions")
                return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def delete_session(self, session_id: str) -> bool:
        """Delete session (CASCADE deletes messages)."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM chat_sessions WHERE session_id = $1", session_id
                )

                deleted = result.split()[1] == "1"  # "DELETE 1"
                if deleted:
                    logger.info(f"Session deleted: {session_id}")
                return deleted

        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False

    async def save_file_record(self, file_record: UploadedFileRecord) -> bool:
        """Save file record."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO uploaded_files
                    (file_id, gemini_file_name, gemini_file_uri, display_name,
                     size, mime_type, uploaded_at, expires_at, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (file_id) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        metadata = EXCLUDED.metadata
                    """,
                    file_record.file_id,
                    file_record.gemini_file_name,
                    file_record.gemini_file_uri,
                    file_record.display_name,
                    file_record.size,
                    file_record.mime_type,
                    file_record.uploaded_at,
                    file_record.expires_at,
                    json.dumps(file_record.metadata),
                )

            logger.debug(f"File record saved: {file_record.file_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save file record: {e}")
            return False

    async def load_file_record(self, file_id: str) -> Optional[UploadedFileRecord]:
        """Load file record."""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT file_id, gemini_file_name, gemini_file_uri, display_name,
                           size, mime_type, uploaded_at, expires_at, metadata
                    FROM uploaded_files
                    WHERE file_id = $1
                    """,
                    file_id,
                )

                if not row:
                    return None

                return UploadedFileRecord(
                    file_id=row["file_id"],
                    gemini_file_name=row["gemini_file_name"],
                    gemini_file_uri=row["gemini_file_uri"],
                    display_name=row["display_name"],
                    size=row["size"],
                    mime_type=row["mime_type"],
                    uploaded_at=row["uploaded_at"],
                    expires_at=row["expires_at"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )

        except Exception as e:
            logger.error(f"Failed to load file record: {e}")
            return None

    async def list_file_records(self, limit: int = 100) -> list[UploadedFileRecord]:
        """List file records."""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT file_id, gemini_file_name, gemini_file_uri, display_name,
                           size, mime_type, uploaded_at, expires_at, metadata
                    FROM uploaded_files
                    ORDER BY uploaded_at DESC
                    LIMIT $1
                    """,
                    limit,
                )

                records = []
                for row in rows:
                    record = UploadedFileRecord(
                        file_id=row["file_id"],
                        gemini_file_name=row["gemini_file_name"],
                        gemini_file_uri=row["gemini_file_uri"],
                        display_name=row["display_name"],
                        size=row["size"],
                        mime_type=row["mime_type"],
                        uploaded_at=row["uploaded_at"],
                        expires_at=row["expires_at"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    )
                    records.append(record)

                return records

        except Exception as e:
            logger.error(f"Failed to list file records: {e}")
            return []

    async def delete_expired_files(self) -> int:
        """Delete expired file records."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM uploaded_files WHERE expires_at < NOW()"
                )

                deleted = int(result.split()[1])  # "DELETE N"
                if deleted > 0:
                    logger.info(f"Deleted {deleted} expired file records")
                return deleted

        except Exception as e:
            logger.error(f"Failed to delete expired files: {e}")
            return 0

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("PostgreSQL connection pool closed")
