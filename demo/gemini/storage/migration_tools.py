"""Cross-Backend Migration Tools - Polish Layer

Tools for migrating data between storage backends and exporting/importing data.

Features:
- Migrate between SQLite, PostgreSQL, and MongoDB
- Bulk export to JSON
- Bulk import from JSON
- Data verification and validation
- Progress tracking

Usage:
    # Migrate from SQLite to PostgreSQL
    await migrate_storage(
        source=sqlite_storage,
        destination=postgresql_storage,
        verify=True
    )

    # Export to JSON
    await export_to_json(storage, "backup.json")

    # Import from JSON
    await import_from_json(storage, "backup.json")
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .base import ChatSession, SessionStorage, UploadedFileRecord

logger = logging.getLogger(__name__)


async def migrate_storage(
    source: SessionStorage,
    destination: SessionStorage,
    verify: bool = True,
    progress_callback: Optional[callable] = None,
) -> dict[str, Any]:
    """Migrate all data from source to destination storage.

    Args:
        source: Source storage backend
        destination: Destination storage backend
        verify: Verify data after migration
        progress_callback: Optional callback for progress updates

    Returns:
        Migration statistics
    """
    logger.info("Starting storage migration")
    start_time = datetime.now()

    # Initialize both storages
    await source.initialize()
    await destination.initialize()

    # Migrate sessions
    sessions = await source.list_sessions(limit=100000)
    sessions_migrated = 0
    sessions_failed = 0

    for i, session_summary in enumerate(sessions):
        try:
            # Load full session with messages
            full_session = await source.load_session(session_summary.session_id)
            if not full_session:
                logger.warning(f"Failed to load session: {session_summary.session_id}")
                sessions_failed += 1
                continue

            # Save to destination
            success = await destination.save_session(full_session)
            if success:
                sessions_migrated += 1
            else:
                sessions_failed += 1
                logger.warning(f"Failed to save session: {full_session.session_id}")

            # Progress callback
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(
                    {
                        "stage": "sessions",
                        "current": i + 1,
                        "total": len(sessions),
                        "percentage": ((i + 1) / len(sessions)) * 100,
                    }
                )

        except Exception as e:
            logger.error(f"Error migrating session: {e}")
            sessions_failed += 1

    # Migrate file records
    file_records = await source.list_file_records(limit=100000)
    files_migrated = 0
    files_failed = 0

    for i, file_record in enumerate(file_records):
        try:
            success = await destination.save_file_record(file_record)
            if success:
                files_migrated += 1
            else:
                files_failed += 1
                logger.warning(f"Failed to save file record: {file_record.file_id}")

            # Progress callback
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(
                    {
                        "stage": "files",
                        "current": i + 1,
                        "total": len(file_records),
                        "percentage": ((i + 1) / len(file_records)) * 100,
                    }
                )

        except Exception as e:
            logger.error(f"Error migrating file record: {e}")
            files_failed += 1

    # Verification
    verification_results = {}
    if verify:
        logger.info("Verifying migration...")
        verification_results = await verify_migration(source, destination)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    results = {
        "success": sessions_failed == 0 and files_failed == 0,
        "sessions_migrated": sessions_migrated,
        "sessions_failed": sessions_failed,
        "files_migrated": files_migrated,
        "files_failed": files_failed,
        "duration_seconds": duration,
        "verification": verification_results if verify else None,
    }

    logger.info(
        f"Migration complete: {sessions_migrated} sessions, {files_migrated} files "
        f"({duration:.2f}s)"
    )

    return results


async def verify_migration(
    source: SessionStorage, destination: SessionStorage
) -> dict[str, Any]:
    """Verify that migration was successful.

    Args:
        source: Source storage
        destination: Destination storage

    Returns:
        Verification results
    """
    logger.info("Starting verification")

    # Count sessions
    source_sessions = await source.list_sessions(limit=100000)
    dest_sessions = await destination.list_sessions(limit=100000)

    session_count_match = len(source_sessions) == len(dest_sessions)

    # Sample and verify a few sessions
    sample_size = min(10, len(source_sessions))
    samples_verified = 0
    samples_failed = 0

    for session_summary in source_sessions[:sample_size]:
        try:
            source_session = await source.load_session(session_summary.session_id)
            dest_session = await destination.load_session(session_summary.session_id)

            if not dest_session:
                samples_failed += 1
                logger.warning(f"Session not found in destination: {session_summary.session_id}")
                continue

            # Verify message count
            if source_session.message_count != dest_session.message_count:
                samples_failed += 1
                logger.warning(
                    f"Message count mismatch: {source_session.session_id} "
                    f"({source_session.message_count} vs {dest_session.message_count})"
                )
                continue

            samples_verified += 1

        except Exception as e:
            logger.error(f"Verification error: {e}")
            samples_failed += 1

    # Count files
    source_files = await source.list_file_records(limit=100000)
    dest_files = await destination.list_file_records(limit=100000)

    file_count_match = len(source_files) == len(dest_files)

    return {
        "success": session_count_match and file_count_match and samples_failed == 0,
        "session_count_match": session_count_match,
        "source_session_count": len(source_sessions),
        "dest_session_count": len(dest_sessions),
        "file_count_match": file_count_match,
        "source_file_count": len(source_files),
        "dest_file_count": len(dest_files),
        "samples_verified": samples_verified,
        "samples_failed": samples_failed,
    }


async def export_to_json(
    storage: SessionStorage, output_path: Path | str
) -> dict[str, Any]:
    """Export all data to JSON file.

    Args:
        storage: Storage backend
        output_path: Output file path

    Returns:
        Export statistics
    """
    logger.info(f"Exporting to JSON: {output_path}")
    output_path = Path(output_path)

    await storage.initialize()

    # Load all sessions
    sessions = await storage.list_sessions(limit=100000)
    sessions_data = []

    for session_summary in sessions:
        full_session = await storage.load_session(session_summary.session_id)
        if full_session:
            sessions_data.append(
                {
                    "session_id": full_session.session_id,
                    "model": full_session.model,
                    "created_at": full_session.created_at.isoformat(),
                    "updated_at": full_session.updated_at.isoformat(),
                    "message_count": full_session.message_count,
                    "metadata": full_session.metadata,
                    "status": full_session.status,
                    "messages": [
                        {
                            "message_id": msg.message_id,
                            "role": msg.role,
                            "content": msg.content,
                            "created_at": msg.created_at.isoformat(),
                            "metadata": msg.metadata,
                        }
                        for msg in full_session.messages
                    ],
                }
            )

    # Load all file records
    file_records = await storage.list_file_records(limit=100000)
    files_data = []

    for file_record in file_records:
        files_data.append(
            {
                "file_id": file_record.file_id,
                "gemini_file_name": file_record.gemini_file_name,
                "gemini_file_uri": file_record.gemini_file_uri,
                "display_name": file_record.display_name,
                "size": file_record.size,
                "mime_type": file_record.mime_type,
                "uploaded_at": file_record.uploaded_at.isoformat(),
                "expires_at": file_record.expires_at.isoformat(),
                "metadata": file_record.metadata,
            }
        )

    # Create export data
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "sessions": sessions_data,
        "file_records": files_data,
    }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    logger.info(
        f"Export complete: {len(sessions_data)} sessions, {len(files_data)} files"
    )

    return {
        "success": True,
        "sessions_exported": len(sessions_data),
        "files_exported": len(files_data),
        "output_path": str(output_path),
    }


async def import_from_json(
    storage: SessionStorage, input_path: Path | str
) -> dict[str, Any]:
    """Import data from JSON file.

    Args:
        storage: Storage backend
        input_path: Input file path

    Returns:
        Import statistics
    """
    logger.info(f"Importing from JSON: {input_path}")
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Import file not found: {input_path}")

    await storage.initialize()

    # Load JSON
    with open(input_path) as f:
        import_data = json.load(f)

    # Import sessions
    sessions_imported = 0
    sessions_failed = 0

    for session_data in import_data.get("sessions", []):
        try:
            # Reconstruct session
            session = ChatSession(
                session_id=session_data["session_id"],
                model=session_data["model"],
                created_at=datetime.fromisoformat(session_data["created_at"]),
                updated_at=datetime.fromisoformat(session_data["updated_at"]),
                message_count=session_data["message_count"],
                metadata=session_data.get("metadata", {}),
                status=session_data.get("status", "active"),
                messages=[],
            )

            # Reconstruct messages
            for msg_data in session_data.get("messages", []):
                from .base import ChatMessage

                message = ChatMessage(
                    message_id=msg_data["message_id"],
                    session_id=session.session_id,
                    role=msg_data["role"],
                    content=msg_data["content"],
                    created_at=datetime.fromisoformat(msg_data["created_at"]),
                    metadata=msg_data.get("metadata", {}),
                )
                session.messages.append(message)

            # Save session
            success = await storage.save_session(session)
            if success:
                sessions_imported += 1
            else:
                sessions_failed += 1

        except Exception as e:
            logger.error(f"Failed to import session: {e}")
            sessions_failed += 1

    # Import file records
    files_imported = 0
    files_failed = 0

    for file_data in import_data.get("file_records", []):
        try:
            file_record = UploadedFileRecord(
                file_id=file_data["file_id"],
                gemini_file_name=file_data["gemini_file_name"],
                gemini_file_uri=file_data["gemini_file_uri"],
                display_name=file_data["display_name"],
                size=file_data["size"],
                mime_type=file_data["mime_type"],
                uploaded_at=datetime.fromisoformat(file_data["uploaded_at"]),
                expires_at=datetime.fromisoformat(file_data["expires_at"]),
                metadata=file_data.get("metadata", {}),
            )

            success = await storage.save_file_record(file_record)
            if success:
                files_imported += 1
            else:
                files_failed += 1

        except Exception as e:
            logger.error(f"Failed to import file record: {e}")
            files_failed += 1

    logger.info(
        f"Import complete: {sessions_imported} sessions, {files_imported} files"
    )

    return {
        "success": sessions_failed == 0 and files_failed == 0,
        "sessions_imported": sessions_imported,
        "sessions_failed": sessions_failed,
        "files_imported": files_imported,
        "files_failed": files_failed,
    }


async def backup_storage(
    storage: SessionStorage, backup_dir: Path | str
) -> dict[str, Any]:
    """Create a timestamped backup of storage.

    Args:
        storage: Storage backend
        backup_dir: Backup directory

    Returns:
        Backup statistics
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"gemini_backup_{timestamp}.json"

    # Export to JSON
    result = await export_to_json(storage, backup_file)
    result["backup_file"] = str(backup_file)

    logger.info(f"Backup created: {backup_file}")

    return result
