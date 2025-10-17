"""Session Manager - Feature Layer

Provides session lifecycle management with auto-save, startup loading,
session expiry, and export functionality.

This is the Feature Layer that wraps the Foundation storage backend
with higher-level session management capabilities.

Usage:
    storage = SQLiteSessionStorage("sessions.db")
    await storage.initialize()

    manager = SessionManager(storage)
    await manager.startup()  # Load sessions on startup

    # Auto-save after message
    await manager.auto_save(session)

    # Export conversation history
    exported = await manager.export_session(session_id, format="json")
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

from .base import ChatSession, SessionStorage

logger = logging.getLogger(__name__)


class SessionManager:
    """High-level session manager with auto-save and lifecycle management.

    Feature Layer providing:
    - Auto-save on message completion
    - Load all sessions on startup
    - Session expiry management
    - Conversation history export
    - Session archival

    Attributes:
        storage: Underlying storage backend
        sessions: In-memory cache of active sessions
        auto_save_enabled: Whether auto-save is enabled
        expiry_hours: Hours before session expires
    """

    def __init__(
        self,
        storage: SessionStorage,
        auto_save_enabled: bool = True,
        expiry_hours: int = 72,
    ):
        """Initialize session manager.

        Args:
            storage: Storage backend
            auto_save_enabled: Enable auto-save on updates
            expiry_hours: Hours before session expires
        """
        self.storage = storage
        self.auto_save_enabled = auto_save_enabled
        self.expiry_hours = expiry_hours
        self.sessions: dict[str, ChatSession] = {}
        logger.info(f"SessionManager initialized (auto_save={auto_save_enabled})")

    async def startup(self) -> dict[str, Any]:
        """Load all active sessions on startup.

        Returns:
            Startup statistics
        """
        logger.info("Loading sessions on startup...")

        # Load active sessions
        active_sessions = await self.storage.list_sessions(status="active", limit=1000)

        # Load into cache
        for session in active_sessions:
            # Load full session with messages
            full_session = await self.storage.load_session(session.session_id)
            if full_session:
                self.sessions[session.session_id] = full_session

        # Clean up expired sessions
        expired_count = await self.cleanup_expired_sessions()

        stats = {
            "active_sessions": len(self.sessions),
            "expired_cleaned": expired_count,
            "startup_time": datetime.now().isoformat(),
        }

        logger.info(
            f"Startup complete: {stats['active_sessions']} active, "
            f"{stats['expired_cleaned']} expired cleaned"
        )

        return stats

    async def auto_save(self, session: ChatSession) -> bool:
        """Auto-save a session after update.

        Args:
            session: Session to save

        Returns:
            True if saved successfully
        """
        if not self.auto_save_enabled:
            return False

        # Update in-memory cache
        self.sessions[session.session_id] = session

        # Save to storage
        saved = await self.storage.save_session(session)

        if saved:
            logger.debug(f"Auto-saved session: {session.session_id}")
        else:
            logger.warning(f"Failed to auto-save session: {session.session_id}")

        return saved

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID (from cache or storage).

        Args:
            session_id: Session identifier

        Returns:
            Session if found
        """
        # Check cache first
        if session_id in self.sessions:
            return self.sessions[session_id]

        # Load from storage
        session = await self.storage.load_session(session_id)
        if session:
            self.sessions[session_id] = session

        return session

    async def create_session(self, model: str) -> ChatSession:
        """Create a new session.

        Args:
            model: Model name

        Returns:
            New session
        """
        session = ChatSession.create(model)
        self.sessions[session.session_id] = session

        if self.auto_save_enabled:
            await self.storage.save_session(session)

        logger.info(f"Created session: {session.session_id}")
        return session

    async def archive_session(self, session_id: str) -> bool:
        """Archive a session (mark as archived but keep in storage).

        Args:
            session_id: Session identifier

        Returns:
            True if archived
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        session.status = "archived"
        session.updated_at = datetime.now()

        # Remove from active cache
        if session_id in self.sessions:
            del self.sessions[session_id]

        # Save to storage
        saved = await self.storage.save_session(session)

        if saved:
            logger.info(f"Archived session: {session_id}")

        return saved

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.

        Marks sessions as expired if they haven't been updated
        within the expiry period.

        Returns:
            Number of sessions expired
        """
        expiry_threshold = datetime.now() - timedelta(hours=self.expiry_hours)
        expired_count = 0

        # Check all active sessions
        active_sessions = await self.storage.list_sessions(status="active", limit=10000)

        for session in active_sessions:
            if session.updated_at < expiry_threshold:
                session.status = "expired"
                await self.storage.save_session(session)
                expired_count += 1

                # Remove from cache
                if session.session_id in self.sessions:
                    del self.sessions[session.session_id]

        if expired_count > 0:
            logger.info(f"Expired {expired_count} sessions")

        return expired_count

    async def export_session(
        self,
        session_id: str,
        format: Literal["json", "text", "markdown"] = "json",
    ) -> Optional[str]:
        """Export session conversation history.

        Args:
            session_id: Session identifier
            format: Export format (json, text, markdown)

        Returns:
            Exported content as string, or None if session not found
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        if format == "json":
            return self._export_json(session)
        elif format == "text":
            return self._export_text(session)
        elif format == "markdown":
            return self._export_markdown(session)
        else:
            raise ValueError(f"Unknown export format: {format}")

    def _export_json(self, session: ChatSession) -> str:
        """Export session as JSON.

        Args:
            session: Session to export

        Returns:
            JSON string
        """
        data = {
            "session_id": session.session_id,
            "model": session.model,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": session.message_count,
            "status": session.status,
            "metadata": session.metadata,
            "messages": [
                {
                    "message_id": msg.message_id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                    "metadata": msg.metadata,
                }
                for msg in session.messages
            ],
        }

        return json.dumps(data, indent=2)

    def _export_text(self, session: ChatSession) -> str:
        """Export session as plain text.

        Args:
            session: Session to export

        Returns:
            Plain text string
        """
        lines = [
            f"Chat Session: {session.session_id}",
            f"Model: {session.model}",
            f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Messages: {session.message_count}",
            "",
            "=" * 80,
            "",
        ]

        for msg in session.messages:
            timestamp = msg.created_at.strftime("%H:%M:%S")
            lines.append(f"[{timestamp}] {msg.role.upper()}:")
            lines.append(msg.content)
            lines.append("")

        return "\n".join(lines)

    def _export_markdown(self, session: ChatSession) -> str:
        """Export session as Markdown.

        Args:
            session: Session to export

        Returns:
            Markdown string
        """
        lines = [
            f"# Chat Session: {session.session_id}",
            "",
            f"**Model:** {session.model}  ",
            f"**Created:** {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Messages:** {session.message_count}  ",
            f"**Status:** {session.status}",
            "",
            "---",
            "",
        ]

        for msg in session.messages:
            timestamp = msg.created_at.strftime("%H:%M:%S")
            role_emoji = "👤" if msg.role == "user" else "🤖"

            lines.append(f"### {role_emoji} {msg.role.title()} ({timestamp})")
            lines.append("")
            lines.append(msg.content)
            lines.append("")

        return "\n".join(lines)

    async def export_to_file(
        self,
        session_id: str,
        output_path: Path,
        format: Literal["json", "text", "markdown"] = "json",
    ) -> bool:
        """Export session to file.

        Args:
            session_id: Session identifier
            output_path: Output file path
            format: Export format

        Returns:
            True if exported successfully
        """
        content = await self.export_session(session_id, format=format)
        if not content:
            return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)
            logger.info(f"Exported session to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export session: {e}")
            return False

    async def get_statistics(self) -> dict[str, Any]:
        """Get session statistics.

        Returns:
            Statistics dictionary
        """
        active_sessions = await self.storage.list_sessions(status="active", limit=10000)
        archived_sessions = await self.storage.list_sessions(status="archived", limit=10000)
        expired_sessions = await self.storage.list_sessions(status="expired", limit=10000)

        total_messages = sum(s.message_count for s in active_sessions)
        avg_messages = total_messages / len(active_sessions) if active_sessions else 0

        return {
            "active_count": len(active_sessions),
            "archived_count": len(archived_sessions),
            "expired_count": len(expired_sessions),
            "total_count": len(active_sessions) + len(archived_sessions) + len(expired_sessions),
            "total_messages": total_messages,
            "average_messages_per_session": round(avg_messages, 2),
            "cached_sessions": len(self.sessions),
        }

    async def shutdown(self) -> None:
        """Shutdown manager and save all cached sessions."""
        logger.info("Shutting down session manager...")

        # Save all cached sessions
        for session in self.sessions.values():
            await self.storage.save_session(session)

        # Close storage
        await self.storage.close()

        logger.info("Session manager shutdown complete")
