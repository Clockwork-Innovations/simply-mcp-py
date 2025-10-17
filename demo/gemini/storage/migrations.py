"""Database Migration System - Feature Layer

Provides schema version tracking and migration execution for storage backends.

This allows evolving the database schema over time while preserving data.

Usage:
    storage = SQLiteSessionStorage("sessions.db")
    await storage.initialize()

    migrator = MigrationSystem(storage)
    await migrator.apply_migrations()
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, TYPE_CHECKING

try:
    import aiosqlite
    _HAS_AIOSQLITE = True
except ImportError:
    aiosqlite = None  # type: ignore
    _HAS_AIOSQLITE = False

if TYPE_CHECKING and aiosqlite:
    from aiosqlite import Connection

logger = logging.getLogger(__name__)


class Migration(ABC):
    """Base class for database migrations.

    Each migration has a version number and implements up/down methods.
    """

    @property
    @abstractmethod
    def version(self) -> int:
        """Migration version number."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Migration description."""
        pass

    @abstractmethod
    async def up(self, connection: "aiosqlite.Connection") -> None:
        """Apply migration (upgrade schema).

        Args:
            connection: Database connection
        """
        pass

    @abstractmethod
    async def down(self, connection: "aiosqlite.Connection") -> None:
        """Revert migration (downgrade schema).

        Args:
            connection: Database connection
        """
        pass


class Migration001AddSessionMetadata(Migration):
    """Example migration: Add metadata column to sessions table."""

    @property
    def version(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Add metadata column to chat_sessions"

    async def up(self, connection: "aiosqlite.Connection") -> None:
        """Add metadata column (already in base schema, this is an example)."""
        # This would be run if we were migrating from an older schema
        logger.info(f"Migration {self.version}: {self.description} (UP)")
        # await connection.execute("ALTER TABLE chat_sessions ADD COLUMN metadata TEXT")

    async def down(self, connection: "aiosqlite.Connection") -> None:
        """Remove metadata column."""
        logger.info(f"Migration {self.version}: {self.description} (DOWN)")
        # SQLite doesn't support DROP COLUMN easily, would need to recreate table


class Migration002AddFileRecords(Migration):
    """Example migration: Add uploaded_files table."""

    @property
    def version(self) -> int:
        return 2

    @property
    def description(self) -> str:
        return "Add uploaded_files table"

    async def up(self, connection: "aiosqlite.Connection") -> None:
        """Add uploaded_files table (already in base schema, this is an example)."""
        logger.info(f"Migration {self.version}: {self.description} (UP)")
        # Would create table if not exists

    async def down(self, connection: "aiosqlite.Connection") -> None:
        """Remove uploaded_files table."""
        logger.info(f"Migration {self.version}: {self.description} (DOWN)")
        # await connection.execute("DROP TABLE IF EXISTS uploaded_files")


class MigrationSystem:
    """Manages database schema migrations.

    Tracks applied migrations and executes pending ones in order.

    Attributes:
        connection: Database connection
        migrations: List of available migrations
    """

    def __init__(self, connection: "aiosqlite.Connection"):
        """Initialize migration system.

        Args:
            connection: Database connection
        """
        self.connection = connection
        self.migrations: list[Migration] = [
            Migration001AddSessionMetadata(),
            Migration002AddFileRecords(),
        ]

        logger.info(f"MigrationSystem initialized with {len(self.migrations)} migrations")

    async def initialize_migrations_table(self) -> None:
        """Create migrations tracking table."""
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                execution_time_ms INTEGER
            )
            """
        )
        await self.connection.commit()

    async def get_current_version(self) -> int:
        """Get current schema version.

        Returns:
            Current version number
        """
        cursor = await self.connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        )
        row = await cursor.fetchone()
        return row[0] if row and row[0] is not None else 0

    async def get_applied_migrations(self) -> set[int]:
        """Get set of applied migration versions.

        Returns:
            Set of version numbers
        """
        cursor = await self.connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        )
        versions = set()
        async for row in cursor:
            versions.add(row[0])
        return versions

    async def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration.

        Args:
            migration: Migration to apply

        Returns:
            True if successful
        """
        try:
            logger.info(f"Applying migration {migration.version}: {migration.description}")
            start_time = datetime.now()

            # Apply migration
            await migration.up(self.connection)

            # Record migration
            end_time = datetime.now()
            execution_time = int((end_time - start_time).total_seconds() * 1000)

            await self.connection.execute(
                """
                INSERT INTO schema_migrations (version, description, execution_time_ms)
                VALUES (?, ?, ?)
                """,
                (migration.version, migration.description, execution_time),
            )

            await self.connection.commit()

            logger.info(
                f"Migration {migration.version} applied successfully "
                f"({execution_time}ms)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            await self.connection.rollback()
            return False

    async def revert_migration(self, migration: Migration) -> bool:
        """Revert a single migration.

        Args:
            migration: Migration to revert

        Returns:
            True if successful
        """
        try:
            logger.info(f"Reverting migration {migration.version}: {migration.description}")

            # Revert migration
            await migration.down(self.connection)

            # Remove from tracking
            await self.connection.execute(
                "DELETE FROM schema_migrations WHERE version = ?",
                (migration.version,),
            )

            await self.connection.commit()

            logger.info(f"Migration {migration.version} reverted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to revert migration {migration.version}: {e}")
            await self.connection.rollback()
            return False

    async def apply_migrations(self, target_version: int | None = None) -> dict[str, Any]:
        """Apply all pending migrations up to target version.

        Args:
            target_version: Target version (None = latest)

        Returns:
            Migration results
        """
        await self.initialize_migrations_table()

        applied_versions = await self.get_applied_migrations()
        current_version = max(applied_versions) if applied_versions else 0

        # Determine target
        if target_version is None:
            target_version = max(m.version for m in self.migrations)

        logger.info(
            f"Applying migrations from version {current_version} to {target_version}"
        )

        # Get pending migrations
        pending = [
            m
            for m in self.migrations
            if m.version not in applied_versions and m.version <= target_version
        ]

        # Sort by version
        pending.sort(key=lambda m: m.version)

        # Apply each migration
        applied = []
        failed = []

        for migration in pending:
            success = await self.apply_migration(migration)
            if success:
                applied.append(migration.version)
            else:
                failed.append(migration.version)
                break  # Stop on first failure

        result = {
            "initial_version": current_version,
            "target_version": target_version,
            "final_version": await self.get_current_version(),
            "applied": applied,
            "failed": failed,
            "success": len(failed) == 0,
        }

        logger.info(
            f"Migration complete: {len(applied)} applied, {len(failed)} failed"
        )

        return result

    async def revert_to_version(self, target_version: int) -> dict[str, Any]:
        """Revert migrations down to target version.

        Args:
            target_version: Target version to revert to

        Returns:
            Revert results
        """
        applied_versions = await self.get_applied_migrations()
        current_version = max(applied_versions) if applied_versions else 0

        logger.info(
            f"Reverting migrations from version {current_version} to {target_version}"
        )

        # Get migrations to revert (in reverse order)
        to_revert = [
            m
            for m in self.migrations
            if m.version in applied_versions and m.version > target_version
        ]

        # Sort by version (descending)
        to_revert.sort(key=lambda m: m.version, reverse=True)

        # Revert each migration
        reverted = []
        failed = []

        for migration in to_revert:
            success = await self.revert_migration(migration)
            if success:
                reverted.append(migration.version)
            else:
                failed.append(migration.version)
                break  # Stop on first failure

        result = {
            "initial_version": current_version,
            "target_version": target_version,
            "final_version": await self.get_current_version(),
            "reverted": reverted,
            "failed": failed,
            "success": len(failed) == 0,
        }

        logger.info(
            f"Revert complete: {len(reverted)} reverted, {len(failed)} failed"
        )

        return result

    async def get_migration_status(self) -> dict[str, Any]:
        """Get migration system status.

        Returns:
            Status dictionary
        """
        applied_versions = await self.get_applied_migrations()
        current_version = max(applied_versions) if applied_versions else 0
        latest_version = max(m.version for m in self.migrations)

        # Get migration history
        cursor = await self.connection.execute(
            """
            SELECT version, description, applied_at, execution_time_ms
            FROM schema_migrations
            ORDER BY version DESC
            LIMIT 10
            """
        )

        history = []
        async for row in cursor:
            history.append(
                {
                    "version": row[0],
                    "description": row[1],
                    "applied_at": row[2],
                    "execution_time_ms": row[3],
                }
            )

        return {
            "current_version": current_version,
            "latest_version": latest_version,
            "pending_count": len([m for m in self.migrations if m.version not in applied_versions]),
            "applied_count": len(applied_versions),
            "is_up_to_date": current_version == latest_version,
            "recent_migrations": history,
        }
