"""Abstract Base Classes for Session Storage

This module defines the abstract interfaces for session persistence.
All storage backends (SQLite, PostgreSQL, MongoDB) implement these interfaces.

Foundation Layer: Core abstractions for session storage.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class ChatMessage:
    """Represents a single chat message in a session.

    Attributes:
        message_id: Unique identifier for this message
        session_id: Parent session identifier
        role: Message role ('user' or 'assistant')
        content: Message content text
        created_at: When message was created
        metadata: Optional additional metadata
    """

    message_id: str
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, session_id: str, role: str, content: str) -> "ChatMessage":
        """Create a new message with auto-generated ID.

        Args:
            session_id: Parent session ID
            role: Message role ('user' or 'assistant')
            content: Message content

        Returns:
            New ChatMessage instance
        """
        return cls(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
        )


@dataclass
class UploadedFileRecord:
    """Record of an uploaded file.

    Attributes:
        file_id: Unique identifier for this file
        gemini_file_name: Gemini API file name
        gemini_file_uri: Gemini API file URI
        display_name: User-friendly display name
        size: File size in bytes
        mime_type: MIME type
        uploaded_at: When file was uploaded
        expires_at: When file expires in Gemini API
        metadata: Optional additional metadata
    """

    file_id: str
    gemini_file_name: str
    gemini_file_uri: str
    display_name: str
    size: int
    mime_type: str
    uploaded_at: datetime
    expires_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        gemini_file_name: str,
        gemini_file_uri: str,
        display_name: str,
        size: int,
        mime_type: str,
        expires_at: datetime,
    ) -> "UploadedFileRecord":
        """Create a new file record with auto-generated ID.

        Args:
            gemini_file_name: Gemini API file name
            gemini_file_uri: Gemini API file URI
            display_name: Display name
            size: File size in bytes
            mime_type: MIME type
            expires_at: Expiration time

        Returns:
            New UploadedFileRecord instance
        """
        return cls(
            file_id=str(uuid.uuid4()),
            gemini_file_name=gemini_file_name,
            gemini_file_uri=gemini_file_uri,
            display_name=display_name,
            size=size,
            mime_type=mime_type,
            uploaded_at=datetime.now(),
            expires_at=expires_at,
        )


@dataclass
class ChatSession:
    """Represents a chat session with message history.

    Attributes:
        session_id: Unique identifier for this session
        model: Model name (e.g., 'gemini-2.5-flash')
        created_at: When session was created
        updated_at: When session was last updated
        message_count: Number of messages in session
        messages: List of messages in session
        metadata: Optional additional metadata
        status: Session status ('active', 'archived', 'expired')
    """

    session_id: str
    model: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    messages: list[ChatMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active, archived, expired

    @classmethod
    def create(cls, model: str) -> "ChatSession":
        """Create a new session with auto-generated ID.

        Args:
            model: Model name

        Returns:
            New ChatSession instance
        """
        return cls(session_id=str(uuid.uuid4()), model=model)

    def add_message(self, role: str, content: str) -> ChatMessage:
        """Add a message to this session.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content

        Returns:
            The created ChatMessage
        """
        message = ChatMessage.create(self.session_id, role, content)
        self.messages.append(message)
        self.message_count = len(self.messages)
        self.updated_at = datetime.now()
        return message

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get conversation history in Gemini API format.

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]


class SessionStorage(ABC):
    """Abstract base class for session storage backends.

    All storage implementations (SQLite, PostgreSQL, MongoDB) must
    implement these methods.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (create tables, indexes, etc.)."""
        pass

    @abstractmethod
    async def save_session(self, session: ChatSession) -> bool:
        """Save a session to storage.

        Args:
            session: Session to save

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load a session from storage.

        Args:
            session_id: Session identifier

        Returns:
            ChatSession if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_sessions(
        self, status: Optional[str] = None, limit: int = 100
    ) -> list[ChatSession]:
        """List sessions from storage.

        Args:
            status: Optional status filter ('active', 'archived', 'expired')
            limit: Maximum number of sessions to return

        Returns:
            List of sessions
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from storage.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def save_file_record(self, file_record: UploadedFileRecord) -> bool:
        """Save an uploaded file record.

        Args:
            file_record: File record to save

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def load_file_record(self, file_id: str) -> Optional[UploadedFileRecord]:
        """Load a file record.

        Args:
            file_id: File identifier

        Returns:
            UploadedFileRecord if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_file_records(
        self, limit: int = 100
    ) -> list[UploadedFileRecord]:
        """List file records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of file records
        """
        pass

    @abstractmethod
    async def delete_expired_files(self) -> int:
        """Delete expired file records.

        Returns:
            Number of records deleted
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close storage connections and cleanup resources."""
        pass
