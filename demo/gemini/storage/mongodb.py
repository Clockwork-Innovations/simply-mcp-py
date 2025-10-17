"""MongoDB Session Storage Implementation - Polish Layer

Document-based session storage with MongoDB backend.

Features:
- NoSQL document database
- Flexible schema-less storage
- Embedded message documents
- Advanced query capabilities
- Horizontal scalability
- Built-in sharding support

Usage:
    storage = MongoDBSessionStorage(
        "mongodb://localhost:27017",
        database="gemini_mcp"
    )
    await storage.initialize()

    session = ChatSession.create("gemini-2.5-flash")
    await storage.save_session(session)

Requirements:
    pip install motor
"""

import logging
from datetime import datetime
from typing import Optional

from typing import TYPE_CHECKING

try:
    from motor import motor_asyncio
    from pymongo import IndexModel, ASCENDING, DESCENDING
    _HAS_MOTOR = True
except ImportError:
    motor_asyncio = None  # type: ignore
    IndexModel = None  # type: ignore
    ASCENDING = None  # type: ignore
    DESCENDING = None  # type: ignore
    _HAS_MOTOR = False

from .base import ChatMessage, ChatSession, SessionStorage, UploadedFileRecord

if TYPE_CHECKING and motor_asyncio:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class MongoDBSessionStorage(SessionStorage):
    """MongoDB-backed session storage.

    Polish Layer implementation providing:
    - Document-oriented NoSQL database
    - Embedded messages within sessions
    - Flexible schema evolution
    - Built-in replication and sharding
    - Geospatial and text search

    Attributes:
        connection_string: MongoDB connection string
        database_name: Database name
        client: MongoDB client
        db: Database instance
    """

    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017",
        database_name: str = "gemini_mcp",
    ):
        """Initialize MongoDB storage.

        Args:
            connection_string: MongoDB connection string
            database_name: Database name

        Raises:
            ImportError: If motor is not installed
        """
        if motor_asyncio is None:
            raise ImportError(
                "motor is required for MongoDB support. "
                "Install with: pip install motor"
            )

        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Optional["motor_asyncio.AsyncIOMotorClient"] = None
        self.db: Optional["motor_asyncio.AsyncIOMotorDatabase"] = None
        logger.info(f"MongoDBSessionStorage initialized: {database_name}")

    async def initialize(self) -> None:
        """Initialize MongoDB client and indexes."""
        # Create client
        self.client = motor_asyncio.AsyncIOMotorClient(self.connection_string)
        self.db = self.client[self.database_name]

        # Create indexes
        await self._create_indexes()

        logger.info("MongoDB client and indexes initialized")

    async def _create_indexes(self) -> None:
        """Create indexes for performance."""
        # Sessions collection indexes
        sessions_indexes = [
            IndexModel([("status", ASCENDING)]),
            IndexModel([("updated_at", DESCENDING)]),
            IndexModel([("model", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
        ]
        await self.db.sessions.create_indexes(sessions_indexes)

        # Files collection indexes
        files_indexes = [
            IndexModel([("expires_at", ASCENDING)]),
            IndexModel([("uploaded_at", DESCENDING)]),
            IndexModel([("gemini_file_name", ASCENDING)]),
        ]
        await self.db.uploaded_files.create_indexes(files_indexes)

        logger.info("MongoDB indexes created")

    async def save_session(self, session: ChatSession) -> bool:
        """Save session with embedded messages.

        In MongoDB, we store messages as embedded documents within the session.

        Args:
            session: Session to save

        Returns:
            True if successful
        """
        try:
            # Update timestamp
            session.updated_at = datetime.now()
            session.message_count = len(session.messages)

            # Convert to document
            doc = {
                "_id": session.session_id,
                "session_id": session.session_id,
                "model": session.model,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": session.message_count,
                "metadata": session.metadata,
                "status": session.status,
                "messages": [
                    {
                        "message_id": msg.message_id,
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at,
                        "metadata": msg.metadata,
                    }
                    for msg in session.messages
                ],
            }

            # Upsert document
            await self.db.sessions.replace_one(
                {"_id": session.session_id}, doc, upsert=True
            )

            logger.debug(f"Session saved: {session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    async def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load session with embedded messages.

        Args:
            session_id: Session identifier

        Returns:
            ChatSession if found
        """
        try:
            # Find document
            doc = await self.db.sessions.find_one({"_id": session_id})

            if not doc:
                return None

            # Convert from document
            session = ChatSession(
                session_id=doc["session_id"],
                model=doc["model"],
                created_at=doc["created_at"],
                updated_at=doc["updated_at"],
                message_count=doc["message_count"],
                metadata=doc.get("metadata", {}),
                status=doc.get("status", "active"),
                messages=[],
            )

            # Convert messages
            for msg_doc in doc.get("messages", []):
                message = ChatMessage(
                    message_id=msg_doc["message_id"],
                    session_id=session.session_id,
                    role=msg_doc["role"],
                    content=msg_doc["content"],
                    created_at=msg_doc["created_at"],
                    metadata=msg_doc.get("metadata", {}),
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
            List of sessions (with messages embedded)
        """
        try:
            # Build query
            query = {}
            if status:
                query["status"] = status

            # Execute query
            cursor = self.db.sessions.find(query).sort("updated_at", DESCENDING).limit(limit)

            sessions = []
            async for doc in cursor:
                session = ChatSession(
                    session_id=doc["session_id"],
                    model=doc["model"],
                    created_at=doc["created_at"],
                    updated_at=doc["updated_at"],
                    message_count=doc["message_count"],
                    metadata=doc.get("metadata", {}),
                    status=doc.get("status", "active"),
                    messages=[],
                )

                # Optionally load messages (for listing, we skip them for performance)
                # If you want messages, uncomment below:
                # for msg_doc in doc.get("messages", []):
                #     message = ChatMessage(...)
                #     session.messages.append(message)

                sessions.append(session)

            logger.debug(f"Listed {len(sessions)} sessions")
            return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def delete_session(self, session_id: str) -> bool:
        """Delete session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        try:
            result = await self.db.sessions.delete_one({"_id": session_id})

            deleted = result.deleted_count > 0
            if deleted:
                logger.info(f"Session deleted: {session_id}")
            return deleted

        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False

    async def save_file_record(self, file_record: UploadedFileRecord) -> bool:
        """Save file record.

        Args:
            file_record: File record to save

        Returns:
            True if successful
        """
        try:
            doc = {
                "_id": file_record.file_id,
                "file_id": file_record.file_id,
                "gemini_file_name": file_record.gemini_file_name,
                "gemini_file_uri": file_record.gemini_file_uri,
                "display_name": file_record.display_name,
                "size": file_record.size,
                "mime_type": file_record.mime_type,
                "uploaded_at": file_record.uploaded_at,
                "expires_at": file_record.expires_at,
                "metadata": file_record.metadata,
            }

            await self.db.uploaded_files.replace_one(
                {"_id": file_record.file_id}, doc, upsert=True
            )

            logger.debug(f"File record saved: {file_record.file_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save file record: {e}")
            return False

    async def load_file_record(self, file_id: str) -> Optional[UploadedFileRecord]:
        """Load file record.

        Args:
            file_id: File identifier

        Returns:
            UploadedFileRecord if found
        """
        try:
            doc = await self.db.uploaded_files.find_one({"_id": file_id})

            if not doc:
                return None

            return UploadedFileRecord(
                file_id=doc["file_id"],
                gemini_file_name=doc["gemini_file_name"],
                gemini_file_uri=doc["gemini_file_uri"],
                display_name=doc["display_name"],
                size=doc["size"],
                mime_type=doc["mime_type"],
                uploaded_at=doc["uploaded_at"],
                expires_at=doc["expires_at"],
                metadata=doc.get("metadata", {}),
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
            cursor = self.db.uploaded_files.find().sort("uploaded_at", DESCENDING).limit(limit)

            records = []
            async for doc in cursor:
                record = UploadedFileRecord(
                    file_id=doc["file_id"],
                    gemini_file_name=doc["gemini_file_name"],
                    gemini_file_uri=doc["gemini_file_uri"],
                    display_name=doc["display_name"],
                    size=doc["size"],
                    mime_type=doc["mime_type"],
                    uploaded_at=doc["uploaded_at"],
                    expires_at=doc["expires_at"],
                    metadata=doc.get("metadata", {}),
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
            result = await self.db.uploaded_files.delete_many(
                {"expires_at": {"$lt": datetime.now()}}
            )

            deleted = result.deleted_count
            if deleted > 0:
                logger.info(f"Deleted {deleted} expired file records")
            return deleted

        except Exception as e:
            logger.error(f"Failed to delete expired files: {e}")
            return 0

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("MongoDB connection closed")

    async def get_statistics(self) -> dict:
        """Get database statistics.

        Returns:
            Statistics dictionary
        """
        try:
            sessions_count = await self.db.sessions.count_documents({})
            active_count = await self.db.sessions.count_documents({"status": "active"})
            files_count = await self.db.uploaded_files.count_documents({})

            # Aggregate total messages
            pipeline = [
                {"$group": {"_id": None, "total_messages": {"$sum": "$message_count"}}}
            ]
            result = await self.db.sessions.aggregate(pipeline).to_list(1)
            total_messages = result[0]["total_messages"] if result else 0

            return {
                "total_sessions": sessions_count,
                "active_sessions": active_count,
                "total_files": files_count,
                "total_messages": total_messages,
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
