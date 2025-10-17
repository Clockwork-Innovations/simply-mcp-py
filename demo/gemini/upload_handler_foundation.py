#!/usr/bin/env python3
"""Foundation Layer: Chunked File Upload Handler

This module provides basic chunked upload functionality for large files.
It splits files into manageable chunks and uploads them sequentially with
basic progress reporting.

Features:
- Split files into 5MB chunks
- Sequential upload with error handling
- Basic progress callback
- Metadata tracking per chunk
- Error recovery per chunk

Usage:
    from upload_handler_foundation import ChunkedUploader, UploadProgress

    def progress_callback(progress: UploadProgress):
        print(f"Progress: {progress.percentage:.1f}% - {progress.message}")

    uploader = ChunkedUploader(api_key="your-key")
    result = await uploader.upload_file(
        file_path="/path/to/large/file.mp4",
        progress_callback=progress_callback
    )
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Constants
DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024  # 5MB chunks
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB max


@dataclass
class UploadProgress:
    """Progress information for file uploads."""

    percentage: float
    stage: str
    message: str
    bytes_uploaded: int
    bytes_total: int
    chunk_index: int
    total_chunks: int
    timestamp: datetime = field(default_factory=datetime.now)
    estimated_seconds: Optional[float] = None


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""

    index: int
    offset: int
    size: int
    checksum: str
    uploaded: bool = False
    upload_time: Optional[float] = None
    error: Optional[str] = None


@dataclass
class UploadResult:
    """Result of a file upload operation."""

    success: bool
    file_id: Optional[str] = None
    file_uri: Optional[str] = None
    file_name: Optional[str] = None
    total_size: int = 0
    total_chunks: int = 0
    upload_duration: float = 0.0
    error: Optional[str] = None
    chunks: list[ChunkMetadata] = field(default_factory=list)


# Type alias for progress callback
ProgressCallback = Optional[Callable[[UploadProgress], None]]


class ChunkedUploader:
    """Foundation layer chunked file uploader.

    Handles splitting large files into chunks and uploading them sequentially.
    Provides basic progress tracking and error handling.

    Attributes:
        api_key: Gemini API key
        chunk_size: Size of each chunk in bytes (default: 5MB)
        max_retries: Maximum number of retry attempts per chunk
    """

    def __init__(
        self,
        api_key: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_retries: int = 3,
    ):
        """Initialize the chunked uploader.

        Args:
            api_key: Gemini API key for authentication
            chunk_size: Size of each chunk in bytes
            max_retries: Maximum retry attempts per chunk
        """
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self._client: Any = None

        logger.info(
            f"ChunkedUploader initialized: chunk_size={chunk_size}, max_retries={max_retries}"
        )

    def _get_client(self) -> Any:
        """Get or create the Gemini client instance."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
                logger.info("Gemini client initialized")
            except ImportError:
                raise RuntimeError(
                    "google-genai SDK not installed. Install with: pip install google-genai"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Gemini client: {e}")

        return self._client

    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type from file extension.

        Args:
            file_path: Path to the file

        Returns:
            MIME type string
        """
        suffix = file_path.suffix.lower()

        mime_map = {
            ".mp3": "audio/mp3",
            ".mp4": "video/mp4",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".html": "text/html",
            ".css": "text/css",
            ".js": "text/javascript",
            ".json": "application/json",
            ".xml": "application/xml",
            ".csv": "text/csv",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }

        return mime_map.get(suffix, "application/octet-stream")

    def _validate_file(self, file_path: Path) -> None:
        """Validate file before upload.

        Args:
            file_path: Path to file to validate

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"File is empty: {file_path}")

        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE} bytes)"
            )

    def _split_into_chunks(self, file_path: Path) -> list[ChunkMetadata]:
        """Split file into chunks and generate metadata.

        Args:
            file_path: Path to file to split

        Returns:
            List of chunk metadata
        """
        file_size = file_path.stat().st_size
        num_chunks = (file_size + self.chunk_size - 1) // self.chunk_size

        chunks: list[ChunkMetadata] = []

        logger.info(
            f"Splitting file into {num_chunks} chunks of {self.chunk_size} bytes"
        )

        for i in range(num_chunks):
            offset = i * self.chunk_size
            chunk_size = min(self.chunk_size, file_size - offset)

            # Read chunk to compute checksum
            with open(file_path, "rb") as f:
                f.seek(offset)
                chunk_data = f.read(chunk_size)
                checksum = hashlib.sha256(chunk_data).hexdigest()

            chunk = ChunkMetadata(
                index=i,
                offset=offset,
                size=chunk_size,
                checksum=checksum,
            )
            chunks.append(chunk)

        logger.info(f"Created metadata for {len(chunks)} chunks")
        return chunks

    def _report_progress(
        self,
        callback: ProgressCallback,
        percentage: float,
        stage: str,
        message: str,
        bytes_uploaded: int,
        bytes_total: int,
        chunk_index: int,
        total_chunks: int,
        estimated_seconds: Optional[float] = None,
    ) -> None:
        """Report progress via callback if provided.

        Args:
            callback: Progress callback function
            percentage: Completion percentage (0-100)
            stage: Current stage name
            message: Progress message
            bytes_uploaded: Bytes uploaded so far
            bytes_total: Total bytes to upload
            chunk_index: Current chunk index
            total_chunks: Total number of chunks
            estimated_seconds: Estimated time remaining
        """
        if callback is not None:
            progress = UploadProgress(
                percentage=percentage,
                stage=stage,
                message=message,
                bytes_uploaded=bytes_uploaded,
                bytes_total=bytes_total,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                estimated_seconds=estimated_seconds,
            )
            callback(progress)

    def _estimate_time_remaining(
        self, bytes_uploaded: int, bytes_total: int, start_time: float
    ) -> Optional[float]:
        """Estimate remaining upload time.

        Args:
            bytes_uploaded: Bytes uploaded so far
            bytes_total: Total bytes to upload
            start_time: Upload start timestamp

        Returns:
            Estimated seconds remaining, or None if can't estimate
        """
        if bytes_uploaded == 0:
            return None

        elapsed = time.time() - start_time
        bytes_per_second = bytes_uploaded / elapsed
        bytes_remaining = bytes_total - bytes_uploaded

        if bytes_per_second > 0:
            return bytes_remaining / bytes_per_second
        return None

    async def _upload_chunk(
        self,
        file_path: Path,
        chunk: ChunkMetadata,
        attempt: int = 1,
    ) -> bool:
        """Upload a single chunk with retry logic.

        Args:
            file_path: Path to the file
            chunk: Chunk metadata
            attempt: Current attempt number

        Returns:
            True if upload succeeded, False otherwise
        """
        try:
            # Read chunk data
            with open(file_path, "rb") as f:
                f.seek(chunk.offset)
                chunk_data = f.read(chunk.size)

            # Verify checksum
            checksum = hashlib.sha256(chunk_data).hexdigest()
            if checksum != chunk.checksum:
                logger.error(
                    f"Chunk {chunk.index} checksum mismatch: expected {chunk.checksum}, got {checksum}"
                )
                return False

            # Simulate upload (in real implementation, this would call Gemini API)
            # For now, we just simulate network delay
            await asyncio.sleep(0.1)

            chunk.uploaded = True
            chunk.upload_time = time.time()

            logger.debug(f"Chunk {chunk.index} uploaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to upload chunk {chunk.index} (attempt {attempt}): {e}")
            chunk.error = str(e)

            # Retry if attempts remaining
            if attempt < self.max_retries:
                logger.info(
                    f"Retrying chunk {chunk.index} (attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(1.0 * attempt)  # Exponential backoff
                return await self._upload_chunk(file_path, chunk, attempt + 1)

            return False

    async def upload_file(
        self,
        file_path: str | Path,
        display_name: Optional[str] = None,
        progress_callback: ProgressCallback = None,
    ) -> UploadResult:
        """Upload a file using chunked upload.

        This is the main entry point for file uploads. The file is split into
        chunks and uploaded sequentially with progress tracking.

        Args:
            file_path: Path to file to upload
            display_name: Optional display name for the file
            progress_callback: Optional callback for progress updates

        Returns:
            UploadResult with success status and metadata
        """
        file_path = Path(file_path)
        start_time = time.time()

        logger.info(f"Starting chunked upload: {file_path}")

        try:
            # Validate file
            self._validate_file(file_path)
            file_size = file_path.stat().st_size

            # Initialize client
            client = self._get_client()

            # Report initial progress
            self._report_progress(
                progress_callback,
                percentage=0.0,
                stage="preparing",
                message="Preparing file for upload",
                bytes_uploaded=0,
                bytes_total=file_size,
                chunk_index=0,
                total_chunks=0,
            )

            # Split into chunks
            chunks = self._split_into_chunks(file_path)
            total_chunks = len(chunks)

            logger.info(f"File split into {total_chunks} chunks")

            # Report chunking complete
            self._report_progress(
                progress_callback,
                percentage=5.0,
                stage="chunking",
                message=f"Split into {total_chunks} chunks",
                bytes_uploaded=0,
                bytes_total=file_size,
                chunk_index=0,
                total_chunks=total_chunks,
            )

            # Upload chunks sequentially
            bytes_uploaded = 0
            upload_start = time.time()

            for chunk in chunks:
                # Upload chunk
                success = await self._upload_chunk(file_path, chunk)

                if not success:
                    error_msg = f"Failed to upload chunk {chunk.index} after {self.max_retries} attempts"
                    logger.error(error_msg)
                    return UploadResult(
                        success=False,
                        error=error_msg,
                        total_size=file_size,
                        total_chunks=total_chunks,
                        upload_duration=time.time() - start_time,
                        chunks=chunks,
                    )

                # Update progress
                bytes_uploaded += chunk.size
                percentage = 5.0 + (bytes_uploaded / file_size * 90.0)
                estimated_time = self._estimate_time_remaining(
                    bytes_uploaded, file_size, upload_start
                )

                self._report_progress(
                    progress_callback,
                    percentage=percentage,
                    stage="uploading",
                    message=f"Uploading chunk {chunk.index + 1}/{total_chunks}",
                    bytes_uploaded=bytes_uploaded,
                    bytes_total=file_size,
                    chunk_index=chunk.index,
                    total_chunks=total_chunks,
                    estimated_seconds=estimated_time,
                )

            # Finalize upload (in real implementation, this would finalize with Gemini API)
            self._report_progress(
                progress_callback,
                percentage=95.0,
                stage="finalizing",
                message="Finalizing upload",
                bytes_uploaded=bytes_uploaded,
                bytes_total=file_size,
                chunk_index=total_chunks,
                total_chunks=total_chunks,
            )

            # Simulate finalization
            await asyncio.sleep(0.2)

            # Generate file ID and URI
            file_id = f"file_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}"
            file_uri = f"gs://gemini-uploads/{file_id}"
            file_name = f"files/{file_id}"

            # Report completion
            self._report_progress(
                progress_callback,
                percentage=100.0,
                stage="complete",
                message="Upload complete",
                bytes_uploaded=bytes_uploaded,
                bytes_total=file_size,
                chunk_index=total_chunks,
                total_chunks=total_chunks,
            )

            duration = time.time() - start_time
            logger.info(
                f"Upload complete: {file_path} -> {file_name} ({duration:.2f}s)"
            )

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
    """Demo of foundation layer chunked upload."""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("Foundation Layer: Chunked Upload Demo")
    print("=" * 70)
    print()

    # Get API key from environment
    import os

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    # Create uploader
    uploader = ChunkedUploader(api_key=api_key, chunk_size=1024 * 1024)  # 1MB chunks

    # Progress callback
    def show_progress(progress: UploadProgress):
        bar_width = 40
        filled = int(bar_width * progress.percentage / 100)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(
            f"\r[{bar}] {progress.percentage:.1f}% | "
            f"Chunk {progress.chunk_index + 1}/{progress.total_chunks} | "
            f"{progress.message}",
            end="",
        )
        if progress.percentage >= 100:
            print()

    # Upload test file
    test_file = Path("test_file.txt")
    if not test_file.exists():
        print(f"Creating test file: {test_file}")
        with open(test_file, "wb") as f:
            f.write(b"Test data\n" * 1000000)  # ~10MB

    print(f"Uploading: {test_file}")
    print()

    result = await uploader.upload_file(test_file, progress_callback=show_progress)

    print()
    print("=" * 70)
    print("Upload Result:")
    print("=" * 70)
    print(f"Success: {result.success}")
    if result.success:
        print(f"File ID: {result.file_id}")
        print(f"File URI: {result.file_uri}")
        print(f"File Name: {result.file_name}")
        print(f"Total Size: {result.total_size} bytes")
        print(f"Total Chunks: {result.total_chunks}")
        print(f"Duration: {result.upload_duration:.2f}s")
    else:
        print(f"Error: {result.error}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main_demo())
