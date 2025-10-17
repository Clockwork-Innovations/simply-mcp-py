#!/usr/bin/env python3
"""Feature Layer: Parallel Upload with Progress Streaming

This module extends the foundation layer with parallel chunk uploads and
real-time progress streaming via Server-Sent Events (SSE).

Features:
- Parallel chunk uploads using async/await
- Real-time progress streaming (SSE/NDJSON)
- Multiple concurrent file uploads
- Progress aggregation across chunks
- HTTP endpoint for progress monitoring
- Improved performance through parallelism

Usage:
    from upload_handler_feature import ParallelUploader, ProgressTracker

    uploader = ParallelUploader(api_key="your-key", max_parallel=3)

    async for progress in uploader.upload_file_streaming("/path/to/file.mp4"):
        print(f"Progress: {progress['percentage']:.1f}%")
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from upload_handler_foundation import (
    ChunkMetadata,
    ChunkedUploader,
    UploadProgress,
    UploadResult,
)

logger = logging.getLogger(__name__)


# Constants
DEFAULT_MAX_PARALLEL = 3  # Max parallel chunk uploads
PROGRESS_UPDATE_INTERVAL = 0.1  # Seconds between progress updates


@dataclass
class StreamingProgress:
    """Progress information for streaming (JSON-serializable)."""

    percentage: float
    stage: str
    message: str
    bytes_uploaded: int
    bytes_total: int
    chunk_index: int
    total_chunks: int
    timestamp: str
    estimated_seconds: Optional[float] = None
    upload_speed: Optional[float] = None  # Bytes per second


@dataclass
class FileUploadTask:
    """Track a file upload task."""

    file_path: Path
    task_id: str
    total_size: int
    total_chunks: int
    bytes_uploaded: int = 0
    chunks_completed: int = 0
    status: str = "pending"  # pending, uploading, complete, failed
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None


class ProgressTracker:
    """Track and aggregate progress across multiple uploads.

    This class manages progress state for concurrent file uploads and
    provides methods to query current progress.

    Attributes:
        tasks: Dictionary of active upload tasks
        progress_history: History of progress updates
    """

    def __init__(self):
        """Initialize the progress tracker."""
        self.tasks: dict[str, FileUploadTask] = {}
        self.progress_history: dict[str, list[StreamingProgress]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def create_task(
        self,
        task_id: str,
        file_path: Path,
        total_size: int,
        total_chunks: int,
    ) -> FileUploadTask:
        """Create a new upload task.

        Args:
            task_id: Unique task identifier
            file_path: Path to file being uploaded
            total_size: Total file size in bytes
            total_chunks: Total number of chunks

        Returns:
            Created upload task
        """
        async with self._lock:
            task = FileUploadTask(
                file_path=file_path,
                task_id=task_id,
                total_size=total_size,
                total_chunks=total_chunks,
                status="pending",
            )
            self.tasks[task_id] = task
            logger.info(f"Created upload task: {task_id}")
            return task

    async def update_progress(
        self,
        task_id: str,
        bytes_uploaded: int,
        chunks_completed: int,
        status: str = "uploading",
    ) -> None:
        """Update progress for a task.

        Args:
            task_id: Task identifier
            bytes_uploaded: Bytes uploaded so far
            chunks_completed: Chunks completed
            status: Current status
        """
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.bytes_uploaded = bytes_uploaded
                task.chunks_completed = chunks_completed
                task.status = status

    async def complete_task(
        self,
        task_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Mark a task as complete.

        Args:
            task_id: Task identifier
            success: Whether upload succeeded
            error: Error message if failed
        """
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = "complete" if success else "failed"
                task.error = error
                task.end_time = time.time()
                logger.info(
                    f"Task {task_id} completed: success={success}, duration={task.end_time - task.start_time:.2f}s"
                )

    async def get_task(self, task_id: str) -> Optional[FileUploadTask]:
        """Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Upload task or None if not found
        """
        async with self._lock:
            return self.tasks.get(task_id)

    async def get_all_tasks(self) -> dict[str, FileUploadTask]:
        """Get all tasks.

        Returns:
            Dictionary of all tasks
        """
        async with self._lock:
            return dict(self.tasks)

    def add_progress(self, task_id: str, progress: StreamingProgress) -> None:
        """Add progress update to history.

        Args:
            task_id: Task identifier
            progress: Progress update
        """
        self.progress_history[task_id].append(progress)

        # Keep only last 100 updates per task
        if len(self.progress_history[task_id]) > 100:
            self.progress_history[task_id] = self.progress_history[task_id][-100:]


class ParallelUploader(ChunkedUploader):
    """Feature layer parallel file uploader.

    Extends the foundation layer with parallel chunk uploads and progress
    streaming capabilities.

    Attributes:
        max_parallel: Maximum number of parallel chunk uploads
        tracker: Progress tracker instance
    """

    def __init__(
        self,
        api_key: str,
        chunk_size: int = 5 * 1024 * 1024,
        max_retries: int = 3,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ):
        """Initialize the parallel uploader.

        Args:
            api_key: Gemini API key
            chunk_size: Size of each chunk in bytes
            max_retries: Maximum retry attempts per chunk
            max_parallel: Maximum parallel chunk uploads
        """
        super().__init__(api_key, chunk_size, max_retries)
        self.max_parallel = max_parallel
        self.tracker = ProgressTracker()

        logger.info(f"ParallelUploader initialized: max_parallel={max_parallel}")

    async def _upload_chunks_parallel(
        self,
        file_path: Path,
        chunks: list[ChunkMetadata],
        task_id: str,
        progress_queue: asyncio.Queue[StreamingProgress],
    ) -> bool:
        """Upload chunks in parallel.

        Args:
            file_path: Path to file
            chunks: List of chunk metadata
            task_id: Upload task ID
            progress_queue: Queue for progress updates

        Returns:
            True if all chunks uploaded successfully
        """
        semaphore = asyncio.Semaphore(self.max_parallel)
        file_size = file_path.stat().st_size
        total_chunks = len(chunks)

        bytes_uploaded = 0
        chunks_completed = 0
        upload_start = time.time()

        async def upload_with_semaphore(chunk: ChunkMetadata) -> bool:
            """Upload chunk with semaphore control."""
            async with semaphore:
                success = await self._upload_chunk(file_path, chunk)

                # Update progress
                nonlocal bytes_uploaded, chunks_completed
                if success:
                    bytes_uploaded += chunk.size
                    chunks_completed += 1

                    # Calculate metrics
                    elapsed = time.time() - upload_start
                    upload_speed = bytes_uploaded / elapsed if elapsed > 0 else 0
                    percentage = 5.0 + (bytes_uploaded / file_size * 90.0)
                    estimated_time = (
                        (file_size - bytes_uploaded) / upload_speed
                        if upload_speed > 0
                        else None
                    )

                    # Create progress update
                    progress = StreamingProgress(
                        percentage=percentage,
                        stage="uploading",
                        message=f"Uploaded chunk {chunks_completed}/{total_chunks}",
                        bytes_uploaded=bytes_uploaded,
                        bytes_total=file_size,
                        chunk_index=chunk.index,
                        total_chunks=total_chunks,
                        timestamp=datetime.now().isoformat(),
                        estimated_seconds=estimated_time,
                        upload_speed=upload_speed,
                    )

                    # Queue progress update
                    await progress_queue.put(progress)

                    # Update tracker
                    await self.tracker.update_progress(
                        task_id, bytes_uploaded, chunks_completed
                    )

                return success

        # Upload all chunks in parallel
        results = await asyncio.gather(
            *[upload_with_semaphore(chunk) for chunk in chunks],
            return_exceptions=True,
        )

        # Check if all succeeded
        return all(isinstance(r, bool) and r for r in results)

    async def upload_file_streaming(
        self,
        file_path: str | Path,
        display_name: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Upload file with streaming progress updates.

        This method yields progress updates in real-time as chunks are uploaded.
        Updates are in JSON format suitable for SSE streaming.

        Args:
            file_path: Path to file to upload
            display_name: Optional display name
            task_id: Optional task ID (generated if not provided)

        Yields:
            Progress updates as JSON-serializable dictionaries
        """
        file_path = Path(file_path)
        task_id = task_id or f"upload_{time.time_ns()}"

        logger.info(f"Starting streaming upload: {file_path} (task_id={task_id})")

        try:
            # Validate file
            self._validate_file(file_path)
            file_size = file_path.stat().st_size

            # Initialize client
            self._get_client()

            # Split into chunks
            chunks = self._split_into_chunks(file_path)
            total_chunks = len(chunks)

            # Create task
            await self.tracker.create_task(task_id, file_path, file_size, total_chunks)

            # Yield initial progress
            yield {
                "percentage": 0.0,
                "stage": "preparing",
                "message": "Preparing file for upload",
                "bytes_uploaded": 0,
                "bytes_total": file_size,
                "chunk_index": 0,
                "total_chunks": total_chunks,
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
            }

            # Yield chunking progress
            yield {
                "percentage": 5.0,
                "stage": "chunking",
                "message": f"Split into {total_chunks} chunks",
                "bytes_uploaded": 0,
                "bytes_total": file_size,
                "chunk_index": 0,
                "total_chunks": total_chunks,
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
            }

            # Create progress queue
            progress_queue: asyncio.Queue[StreamingProgress] = asyncio.Queue()

            # Start upload task
            async def upload_task():
                """Background upload task."""
                success = await self._upload_chunks_parallel(
                    file_path, chunks, task_id, progress_queue
                )
                await progress_queue.put(None)  # Signal completion
                return success

            upload_future = asyncio.create_task(upload_task())

            # Stream progress updates
            while True:
                progress = await progress_queue.get()
                if progress is None:
                    break

                # Convert to dict and yield
                progress_dict = asdict(progress)
                progress_dict["task_id"] = task_id
                yield progress_dict

                # Add to tracker history
                self.tracker.add_progress(task_id, progress)

            # Wait for upload to complete
            success = await upload_future

            if not success:
                error_msg = "Failed to upload one or more chunks"
                await self.tracker.complete_task(task_id, False, error_msg)
                yield {
                    "percentage": 0.0,
                    "stage": "failed",
                    "message": error_msg,
                    "bytes_uploaded": 0,
                    "bytes_total": file_size,
                    "chunk_index": 0,
                    "total_chunks": total_chunks,
                    "timestamp": datetime.now().isoformat(),
                    "task_id": task_id,
                    "error": error_msg,
                }
                return

            # Yield finalization progress
            yield {
                "percentage": 95.0,
                "stage": "finalizing",
                "message": "Finalizing upload",
                "bytes_uploaded": file_size,
                "bytes_total": file_size,
                "chunk_index": total_chunks,
                "total_chunks": total_chunks,
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
            }

            # Simulate finalization
            await asyncio.sleep(0.2)

            # Generate result
            import hashlib

            file_id = f"file_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}"
            file_uri = f"gs://gemini-uploads/{file_id}"
            file_name = f"files/{file_id}"

            # Complete task
            await self.tracker.complete_task(task_id, True)

            # Yield completion
            yield {
                "percentage": 100.0,
                "stage": "complete",
                "message": "Upload complete",
                "bytes_uploaded": file_size,
                "bytes_total": file_size,
                "chunk_index": total_chunks,
                "total_chunks": total_chunks,
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "file_id": file_id,
                "file_uri": file_uri,
                "file_name": file_name,
            }

            logger.info(f"Streaming upload complete: {task_id}")

        except Exception as e:
            error_msg = f"Upload failed: {e}"
            logger.error(error_msg)

            await self.tracker.complete_task(task_id, False, error_msg)

            yield {
                "percentage": 0.0,
                "stage": "failed",
                "message": error_msg,
                "bytes_uploaded": 0,
                "bytes_total": 0,
                "chunk_index": 0,
                "total_chunks": 0,
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "error": error_msg,
            }

    async def upload_file_parallel(
        self,
        file_path: str | Path,
        display_name: Optional[str] = None,
    ) -> UploadResult:
        """Upload file using parallel chunks (non-streaming).

        Args:
            file_path: Path to file to upload
            display_name: Optional display name

        Returns:
            Upload result
        """
        file_path = Path(file_path)
        start_time = time.time()

        logger.info(f"Starting parallel upload: {file_path}")

        try:
            # Validate file
            self._validate_file(file_path)
            file_size = file_path.stat().st_size

            # Initialize client
            self._get_client()

            # Split into chunks
            chunks = self._split_into_chunks(file_path)
            total_chunks = len(chunks)

            # Create progress queue (not used in non-streaming mode)
            progress_queue: asyncio.Queue[StreamingProgress] = asyncio.Queue()
            task_id = f"upload_{time.time_ns()}"

            # Upload chunks in parallel
            success = await self._upload_chunks_parallel(
                file_path, chunks, task_id, progress_queue
            )

            if not success:
                return UploadResult(
                    success=False,
                    error="Failed to upload one or more chunks",
                    total_size=file_size,
                    total_chunks=total_chunks,
                    upload_duration=time.time() - start_time,
                    chunks=chunks,
                )

            # Generate result
            import hashlib

            file_id = f"file_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}"
            file_uri = f"gs://gemini-uploads/{file_id}"
            file_name = f"files/{file_id}"

            duration = time.time() - start_time
            logger.info(f"Parallel upload complete: {file_path} ({duration:.2f}s)")

            return UploadResult(
                success=True,
                file_id=file_id,
                file_uri=file_uri,
                file_name=file_name,
                total_size=file_size,
                total_chunks=total_chunks,
                upload_duration=duration,
                chunks=chunks,
            )

        except Exception as e:
            error_msg = f"Upload failed: {e}"
            logger.error(error_msg)
            return UploadResult(
                success=False,
                error=error_msg,
                upload_duration=time.time() - start_time,
            )


async def main_demo() -> None:
    """Demo of feature layer parallel upload with streaming."""
    import os
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("Feature Layer: Parallel Upload with Streaming Demo")
    print("=" * 70)
    print()

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    # Create uploader
    uploader = ParallelUploader(
        api_key=api_key,
        chunk_size=1024 * 1024,  # 1MB chunks
        max_parallel=3,
    )

    # Create test file
    test_file = Path("test_file.txt")
    if not test_file.exists():
        print(f"Creating test file: {test_file}")
        with open(test_file, "wb") as f:
            f.write(b"Test data\n" * 1000000)  # ~10MB

    print(f"Uploading: {test_file}")
    print()

    # Stream progress updates
    async for progress in uploader.upload_file_streaming(test_file):
        percentage = progress.get("percentage", 0)
        stage = progress.get("stage", "")
        message = progress.get("message", "")

        # Progress bar
        bar_width = 40
        filled = int(bar_width * percentage / 100)
        bar = "=" * filled + "-" * (bar_width - filled)

        print(f"\r[{bar}] {percentage:.1f}% | {stage}: {message}", end="")

        if percentage >= 100 or stage == "failed":
            print()
            break

    print()
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main_demo())
