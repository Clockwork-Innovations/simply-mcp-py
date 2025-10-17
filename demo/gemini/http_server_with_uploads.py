#!/usr/bin/env python3
"""HTTP Server with Integrated Upload Handlers

This demo showcases all three layers of the async file upload system:
- Foundation: Chunked uploads
- Feature: Parallel uploads with progress streaming
- Polish: Resumable uploads with full production features

The server provides REST endpoints for:
- File upload with real-time progress (SSE)
- Resume interrupted uploads
- Query upload status
- List/manage upload sessions

Usage:
    python demo/gemini/http_server_with_uploads.py

    # Upload with streaming progress
    curl -X POST -F "file=@large_file.mp4" \
        http://localhost:8000/upload/streaming?session_id=my-upload

    # Resume upload
    curl -X POST http://localhost:8000/upload/resume/my-upload

    # Get session status
    curl http://localhost:8000/upload/status/my-upload
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from upload_handler_foundation import ChunkedUploader
from upload_handler_feature import ParallelUploader
from upload_handler_polish import ResumableUploader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class UploadServer:
    """HTTP server with integrated upload handlers.

    Provides REST API for file uploads with all three layers:
    - Foundation layer for basic chunked uploads
    - Feature layer for parallel uploads with progress streaming
    - Polish layer for resumable uploads

    Attributes:
        api_key: Gemini API key
        chunk_size: Chunk size for uploads
        max_parallel: Maximum parallel uploads
        bandwidth_limit: Bandwidth limit in bytes/second
        foundation_uploader: Foundation layer uploader
        feature_uploader: Feature layer uploader
        polish_uploader: Polish layer uploader
    """

    def __init__(
        self,
        api_key: str,
        chunk_size: int = 5 * 1024 * 1024,
        max_parallel: int = 3,
        bandwidth_limit: Optional[int] = None,
    ):
        """Initialize upload server.

        Args:
            api_key: Gemini API key
            chunk_size: Chunk size in bytes
            max_parallel: Maximum parallel uploads
            bandwidth_limit: Optional bandwidth limit
        """
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.max_parallel = max_parallel
        self.bandwidth_limit = bandwidth_limit

        # Initialize uploaders
        self.foundation_uploader = ChunkedUploader(api_key, chunk_size)
        self.feature_uploader = ParallelUploader(api_key, chunk_size, max_parallel=max_parallel)
        self.polish_uploader = ResumableUploader(
            api_key, chunk_size, max_parallel=max_parallel, bandwidth_limit=bandwidth_limit
        )

        logger.info("UploadServer initialized with all three layers")

    async def upload_foundation(
        self, file_path: Path, display_name: Optional[str] = None
    ) -> dict[str, Any]:
        """Upload file using foundation layer (basic chunked upload).

        Args:
            file_path: Path to file
            display_name: Optional display name

        Returns:
            Upload result as dictionary
        """
        logger.info(f"Foundation upload: {file_path}")

        result = await self.foundation_uploader.upload_file(
            file_path, display_name=display_name
        )

        return {
            "layer": "foundation",
            "success": result.success,
            "file_id": result.file_id,
            "file_uri": result.file_uri,
            "file_name": result.file_name,
            "total_size": result.total_size,
            "total_chunks": result.total_chunks,
            "duration": result.upload_duration,
            "error": result.error,
        }

    async def upload_feature_streaming(
        self, file_path: Path, display_name: Optional[str] = None
    ):
        """Upload file using feature layer with streaming progress.

        Args:
            file_path: Path to file
            display_name: Optional display name

        Yields:
            Progress updates as JSON strings (NDJSON format)
        """
        logger.info(f"Feature streaming upload: {file_path}")

        async for progress in self.feature_uploader.upload_file_streaming(file_path):
            # Convert to NDJSON format
            yield json.dumps(progress) + "\n"

    async def upload_resumable(
        self,
        file_path: Path,
        session_id: str,
        display_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Upload file using polish layer (resumable).

        Args:
            file_path: Path to file
            session_id: Session identifier
            display_name: Optional display name

        Returns:
            Upload result as dictionary
        """
        logger.info(f"Resumable upload: {file_path} (session={session_id})")

        result = await self.polish_uploader.upload_file_resumable(
            file_path, session_id=session_id, display_name=display_name
        )

        return {
            "layer": "polish",
            "success": result.success,
            "file_id": result.file_id,
            "file_uri": result.file_uri,
            "file_name": result.file_name,
            "total_size": result.total_size,
            "total_chunks": result.total_chunks,
            "duration": result.upload_duration,
            "error": result.error,
            "session_id": session_id,
        }

    async def get_session_status(self, session_id: str) -> dict[str, Any]:
        """Get status of an upload session.

        Args:
            session_id: Session identifier

        Returns:
            Session status as dictionary
        """
        session = await self.polish_uploader.session_manager.load_session(session_id)

        if session is None:
            return {
                "success": False,
                "error": f"Session not found: {session_id}",
            }

        return {
            "success": True,
            "session_id": session.session_id,
            "file_path": str(session.file_path),
            "file_size": session.file_size,
            "file_checksum": session.file_checksum,
            "total_chunks": session.total_chunks,
            "chunks_completed": len(session.chunks_completed),
            "progress_percentage": session.get_progress_percentage(),
            "status": session.status,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "is_complete": session.is_complete(),
            "is_expired": session.is_expired(),
        }

    async def list_sessions(self) -> dict[str, Any]:
        """List all upload sessions.

        Returns:
            Dictionary with list of sessions
        """
        sessions = await self.polish_uploader.session_manager.list_sessions()

        return {
            "success": True,
            "count": len(sessions),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "file_path": str(s.file_path),
                    "file_size": s.file_size,
                    "progress_percentage": s.get_progress_percentage(),
                    "status": s.status,
                    "created_at": s.created_at.isoformat(),
                }
                for s in sessions
            ],
        }

    async def cleanup_sessions(self) -> dict[str, Any]:
        """Clean up expired sessions.

        Returns:
            Cleanup result
        """
        cleaned = await self.polish_uploader.session_manager.cleanup_expired_sessions()

        return {
            "success": True,
            "cleaned_count": cleaned,
            "message": f"Cleaned up {cleaned} expired session(s)",
        }


async def run_demo_cli():
    """Run CLI demo of upload server."""
    print("=" * 80)
    print("HTTP Server with Integrated Upload Handlers - CLI Demo")
    print("=" * 80)
    print()

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        return

    # Create server
    server = UploadServer(
        api_key=api_key,
        chunk_size=1024 * 1024,  # 1MB chunks
        max_parallel=3,
        bandwidth_limit=10 * 1024 * 1024,  # 10MB/s
    )

    print("Server initialized with:")
    print(f"  - Chunk size: {server.chunk_size / (1024*1024):.1f} MB")
    print(f"  - Max parallel: {server.max_parallel}")
    print(f"  - Bandwidth limit: {server.bandwidth_limit / (1024*1024):.1f} MB/s")
    print()

    # Create test file
    test_file = Path("test_upload.txt")
    if not test_file.exists():
        print(f"Creating test file: {test_file}")
        with open(test_file, "wb") as f:
            f.write(b"Test data\n" * 500000)  # ~5MB
        print()

    # Demo 1: Foundation layer upload
    print("-" * 80)
    print("DEMO 1: Foundation Layer - Basic Chunked Upload")
    print("-" * 80)
    result = await server.upload_foundation(test_file)
    print(json.dumps(result, indent=2))
    print()

    # Demo 2: Feature layer streaming upload
    print("-" * 80)
    print("DEMO 2: Feature Layer - Parallel Upload with Progress Streaming")
    print("-" * 80)
    async for progress_json in server.upload_feature_streaming(test_file):
        progress = json.loads(progress_json)
        percentage = progress.get("percentage", 0)
        stage = progress.get("stage", "")
        message = progress.get("message", "")

        # Progress bar
        bar_width = 40
        filled = int(bar_width * percentage / 100)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"\r[{bar}] {percentage:.1f}% | {stage}: {message}", end="")

        if percentage >= 100:
            print()
            break
    print()

    # Demo 3: Polish layer resumable upload
    print("-" * 80)
    print("DEMO 3: Polish Layer - Resumable Upload")
    print("-" * 80)
    result = await server.upload_resumable(test_file, session_id="demo-session-1")
    print(json.dumps(result, indent=2))
    print()

    # Demo 4: Check session status
    print("-" * 80)
    print("DEMO 4: Get Session Status")
    print("-" * 80)
    status = await server.get_session_status("demo-session-1")
    print(json.dumps(status, indent=2))
    print()

    # Demo 5: List all sessions
    print("-" * 80)
    print("DEMO 5: List All Sessions")
    print("-" * 80)
    sessions = await server.list_sessions()
    print(json.dumps(sessions, indent=2))
    print()

    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)


async def run_http_server():
    """Run actual HTTP server with endpoints.

    Note: This requires aiohttp or similar HTTP framework.
    For now, this is a placeholder showing the API structure.
    """
    try:
        from aiohttp import web
    except ImportError:
        print("ERROR: aiohttp not installed. Install with: pip install aiohttp")
        print("Running CLI demo instead...")
        await run_demo_cli()
        return

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        return

    # Create server
    server = UploadServer(api_key=api_key)

    # Define routes
    async def upload_streaming(request):
        """POST /upload/streaming - Upload with progress streaming."""
        reader = await request.multipart()
        field = await reader.next()

        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/upload_{datetime.now().timestamp()}")
        with open(temp_path, "wb") as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        # Stream progress
        response = web.StreamResponse()
        response.headers["Content-Type"] = "application/x-ndjson"
        await response.prepare(request)

        async for progress_json in server.upload_feature_streaming(temp_path):
            await response.write(progress_json.encode())

        await response.write_eof()
        return response

    async def upload_resumable(request):
        """POST /upload/resumable - Resumable upload."""
        data = await request.post()
        file_field = data["file"]
        session_id = request.query.get("session_id", f"upload_{datetime.now().timestamp()}")

        # Save file
        temp_path = Path(f"/tmp/upload_{session_id}")
        with open(temp_path, "wb") as f:
            f.write(file_field.file.read())

        # Upload
        result = await server.upload_resumable(temp_path, session_id)
        return web.json_response(result)

    async def get_status(request):
        """GET /upload/status/{session_id} - Get session status."""
        session_id = request.match_info["session_id"]
        status = await server.get_session_status(session_id)
        return web.json_response(status)

    async def list_all_sessions(request):
        """GET /upload/sessions - List all sessions."""
        sessions = await server.list_sessions()
        return web.json_response(sessions)

    async def cleanup(request):
        """POST /upload/cleanup - Clean up expired sessions."""
        result = await server.cleanup_sessions()
        return web.json_response(result)

    # Create app
    app = web.Application()
    app.router.add_post("/upload/streaming", upload_streaming)
    app.router.add_post("/upload/resumable", upload_resumable)
    app.router.add_get("/upload/status/{session_id}", get_status)
    app.router.add_get("/upload/sessions", list_all_sessions)
    app.router.add_post("/upload/cleanup", cleanup)

    # Run server
    print("=" * 80)
    print("HTTP Server with Upload Handlers")
    print("=" * 80)
    print()
    print("Server running on http://localhost:8000")
    print()
    print("Endpoints:")
    print("  POST /upload/streaming       - Upload with progress streaming (SSE)")
    print("  POST /upload/resumable        - Resumable upload")
    print("  GET  /upload/status/{id}      - Get session status")
    print("  GET  /upload/sessions         - List all sessions")
    print("  POST /upload/cleanup          - Clean up expired sessions")
    print()
    print("=" * 80)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8000)
    await site.start()

    # Keep running
    print("Press Ctrl+C to stop...")
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await runner.cleanup()


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        await run_http_server()
    else:
        await run_demo_cli()


if __name__ == "__main__":
    asyncio.run(main())
