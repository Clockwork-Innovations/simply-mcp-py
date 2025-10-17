#!/usr/bin/env python3
"""Gemini MCP Server - Feature Layer

This server provides core integration with Google's Gemini API through the google-genai SDK.
It enables file uploads, content generation, and interactive chat sessions.

Feature layer capabilities:
- Upload files to Gemini Files API
- List all uploaded files with expiration filtering
- Delete files from Gemini Files API
- Generate content with optional file context
- Start chat sessions with initial messages
- Send messages to continue chat sessions

Installation:
    pip install simply-mcp google-genai

Usage:
    # Set API key
    export GEMINI_API_KEY="your-api-key"

    # Run server
    simply-mcp dev demo/gemini/server.py

    # Or run directly
    python demo/gemini/server.py

Features:
    - File uploads with MIME type detection
    - File management (list, delete)
    - Content generation with Gemini models
    - Chat session management
    - Proper error handling
    - File and session registry
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from simply_mcp import BuildMCPServer

# Lazy import for google-genai SDK
# We defer the import until it's actually needed to handle subprocess environments better
GENAI_AVAILABLE = True  # Assume available until proven otherwise
genai = None  # type: ignore
types = None  # type: ignore
_SDK_IMPORT_ERROR: str | None = None


def _ensure_genai_available() -> bool:
    """Ensure google-genai SDK is imported and available.

    Returns:
        True if SDK is available, False otherwise
    """
    global GENAI_AVAILABLE, genai, types, _SDK_IMPORT_ERROR

    # If already imported, return status
    if genai is not None:
        return GENAI_AVAILABLE

    # Try to import
    try:
        from google import genai as _genai_module
        from google.genai import types as _types_module

        genai = _genai_module
        types = _types_module
        GENAI_AVAILABLE = True
        logger.debug("google-genai SDK imported successfully")
        return True

    except ImportError as e:
        GENAI_AVAILABLE = False
        _SDK_IMPORT_ERROR = str(e)
        logger.error(f"Failed to import google-genai SDK: {e}")
        logger.error("Please install: pip install google-genai")
        return False


# Try initial import at module load (best case)
try:
    from google import genai as _genai_module
    from google.genai import types as _types_module
    genai = _genai_module
    types = _types_module
    GENAI_AVAILABLE = True
except ImportError:
    # Will be imported lazily when needed
    pass

# Try to import tomllib (Python 3.11+) or tomli for TOML support
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

# Load dotenv
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
)

logger = logging.getLogger(__name__)

# Global state
GEMINI_CLIENT: Any = None
FILE_REGISTRY: dict[str, dict[str, Any]] = {}
CHAT_SESSIONS: dict[str, "ChatSession"] = {}


@dataclass
class ChatSession:
    """Represents an active chat session with Gemini."""

    session_id: str
    model: str
    chat: Any  # genai.chats.Chat object
    created_at: datetime
    message_count: int


@dataclass
class GeminiConfig:
    """Configuration for Gemini MCP Server."""

    api_key: str | None
    default_model: str
    file_warning_hours: int
    max_file_size: int


@dataclass
class Progress:
    """Progress information for long-running operations."""

    percentage: float
    stage: str
    message: str
    estimated_seconds: Optional[float] = None
    timestamp: Optional[datetime] = None


ProgressCallback = Optional[Callable[[Progress], None]]


def _load_env_files() -> None:
    """Load .env files from current directory or demo/gemini/ directory."""
    if load_dotenv is None:
        return

    # Try current directory first
    current_dir = Path.cwd()
    env_path = current_dir / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded .env from: {env_path}")
        return

    # Try demo/gemini/ directory
    script_dir = Path(__file__).parent
    env_path = script_dir / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded .env from: {env_path}")
        return

    logger.debug("No .env file found, using environment variables only")


def _load_toml_config() -> dict[str, Any]:
    """Load optional TOML configuration file.

    Returns:
        Configuration dictionary from TOML file, or empty dict if not found
    """
    if tomllib is None:
        logger.debug("TOML support not available (tomllib/tomli not installed)")
        return {}

    # Try current directory first
    current_dir = Path.cwd()
    toml_path = current_dir / "config.toml"

    if not toml_path.exists():
        # Try demo/gemini/ directory
        script_dir = Path(__file__).parent
        toml_path = script_dir / "config.toml"

    if toml_path.exists():
        try:
            with open(toml_path, "rb") as f:
                config = tomllib.load(f)
                logger.info(f"Loaded config.toml from: {toml_path}")
                return config.get("gemini", {})
        except Exception as e:
            logger.warning(f"Failed to load config.toml: {e}")
            return {}

    logger.debug("No config.toml file found")
    return {}


def _validate_config(config: GeminiConfig) -> None:
    """Validate configuration and log warnings for issues.

    Args:
        config: Configuration to validate
    """
    if not config.api_key:
        logger.error("API key is required but not provided")

    # Known valid Gemini models (as of 2025)
    valid_models = [
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.0-pro",
    ]

    if config.default_model not in valid_models:
        logger.warning(
            f"Model '{config.default_model}' may not be valid. "
            f"Known models: {', '.join(valid_models)}"
        )

    if config.file_warning_hours <= 0:
        logger.warning(
            f"file_warning_hours should be positive, got: {config.file_warning_hours}"
        )

    if config.max_file_size <= 0:
        logger.warning(
            f"max_file_size should be positive, got: {config.max_file_size}"
        )


def _load_configuration() -> GeminiConfig:
    """Load configuration from environment variables and config files.

    Precedence order (highest to lowest):
    1. Environment variables (GEMINI_*)
    2. TOML configuration file (config.toml)
    3. .env file
    4. Built-in defaults

    Returns:
        Loaded configuration
    """
    # Load .env files first (lowest precedence)
    _load_env_files()

    # Load TOML config (medium precedence)
    toml_config = _load_toml_config()

    # Build configuration with precedence: ENV > TOML > defaults
    config = GeminiConfig(
        api_key=os.getenv("GEMINI_API_KEY") or toml_config.get("api_key"),
        default_model=(
            os.getenv("GEMINI_DEFAULT_MODEL")
            or toml_config.get("default_model")
            or "gemini-2.5-flash"
        ),
        file_warning_hours=int(
            os.getenv("GEMINI_FILE_WARNING_HOURS")
            or toml_config.get("file_warning_hours")
            or 24
        ),
        max_file_size=int(
            os.getenv("GEMINI_MAX_FILE_SIZE")
            or toml_config.get("max_file_size")
            or 2147483648  # 2GB
        ),
    )

    # Validate configuration
    _validate_config(config)

    logger.info(
        "Configuration loaded",
        extra={
            "default_model": config.default_model,
            "file_warning_hours": config.file_warning_hours,
            "max_file_size": config.max_file_size,
            "api_key_set": bool(config.api_key),
        },
    )

    return config


def _report_progress(
    callback: ProgressCallback,
    percentage: float,
    stage: str,
    message: str,
    estimated_seconds: Optional[float] = None,
) -> None:
    """Report progress if callback is provided."""
    if callback is not None:
        progress = Progress(
            percentage=percentage,
            stage=stage,
            message=message,
            estimated_seconds=estimated_seconds,
            timestamp=datetime.now(),
        )
        logger.debug(f"Progress: {percentage:.1f}% - {stage} - {message}")
        callback(progress)


def _estimate_time(
    bytes_processed: int, bytes_total: int, start_time: float
) -> Optional[float]:
    """Estimate remaining time based on current progress."""
    if bytes_processed == 0:
        return None
    elapsed = time.time() - start_time
    bytes_per_second = bytes_processed / elapsed
    bytes_remaining = bytes_total - bytes_processed
    return bytes_remaining / bytes_per_second if bytes_per_second > 0 else None


def _format_progress_bar(percentage: float, width: int = 40) -> str:
    """Format a simple text progress bar."""
    filled = int(width * percentage / 100)
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {percentage:.1f}%"


# Load configuration at module level
CONFIG = _load_configuration()


def _get_gemini_client() -> Any:
    """Get or create the Gemini client instance.

    Returns:
        Initialized Gemini client or None if API key is not set
    """
    global GEMINI_CLIENT

    if GEMINI_CLIENT is not None:
        return GEMINI_CLIENT

    if not GENAI_AVAILABLE:
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None

    try:
        GEMINI_CLIENT = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized successfully")
        return GEMINI_CLIENT
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None


def _detect_mime_type(file_path: str) -> str:
    """Detect MIME type from file extension.

    Args:
        file_path: Path to the file

    Returns:
        MIME type string
    """
    suffix = Path(file_path).suffix.lower()

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


def _get_mime_type_for_uri(file_uri: str) -> str:
    """Get MIME type for a file URI from registry or detect from URI.

    Args:
        file_uri: Gemini file URI

    Returns:
        MIME type string
    """
    # Check if URI is in our registry
    for file_data in FILE_REGISTRY.values():
        if file_data["uri"] == file_uri:
            return file_data["mime_type"]

    # Try to detect from URI
    return _detect_mime_type(file_uri)


# ===================================================================
# Tool 1: Upload File
# ===================================================================
def upload_file(
    file_uri: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Upload a file to Gemini Files API.

    Args:
        file_uri: Local file path to upload
        display_name: Optional display name for the file

    Returns:
        Upload confirmation with file details
    """
    # Ensure SDK is available (lazy import if needed)
    if not _ensure_genai_available():
        return {
            "success": False,
            "error": "google-genai SDK not available",
            "install_hint": "pip install google-genai",
        }

    client = _get_gemini_client()
    if not client:
        return {
            "success": False,
            "error": "GEMINI_API_KEY environment variable not set",
            "hint": "Set GEMINI_API_KEY before using this tool",
        }

    # Validate file exists
    file_path = Path(file_uri)
    if not file_path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_uri}",
        }

    if not file_path.is_file():
        return {
            "success": False,
            "error": f"Path is not a file: {file_uri}",
        }

    try:
        # Detect MIME type
        mime_type = _detect_mime_type(file_uri)

        # Upload file
        logger.info(f"Uploading file: {file_uri} (mime_type: {mime_type})")

        uploaded_file = client.files.upload(file=file_path)

        # Generate file ID for registry
        file_id = f"file_{len(FILE_REGISTRY):03d}"

        # Calculate expiration (Gemini files typically expire after 48 hours)
        expires_at = datetime.now() + timedelta(hours=48)

        # Store in registry
        FILE_REGISTRY[file_id] = {
            "name": uploaded_file.name,
            "uri": uploaded_file.uri,
            "display_name": display_name or file_path.name,
            "local_path": str(file_path),
            "size": file_path.stat().st_size,
            "mime_type": mime_type,
            "uploaded_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
        }

        logger.info(
            f"File uploaded successfully: {file_id}",
            extra={"file_id": file_id, "gemini_name": uploaded_file.name},
        )

        return {
            "success": True,
            "file_id": file_id,
            "file_uri": uploaded_file.uri,
            "file_name": uploaded_file.name,
            "display_name": display_name or file_path.name,
            "size": file_path.stat().st_size,
            "mime_type": mime_type,
            "expires_at": expires_at.isoformat(),
            "message": f"File uploaded successfully: {file_id}",
        }

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return {
            "success": False,
            "error": f"Upload failed: {str(e)}",
        }


# ===================================================================
# Tool 2: Generate Content
# ===================================================================
def generate_content(
    prompt: str,
    file_uris: list[str] | None = None,
    model: str = "gemini-2.5-flash",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Generate content using Gemini API.

    Args:
        prompt: Text prompt for generation
        file_uris: Optional list of Gemini file URIs to include as context
        model: Gemini model to use (default: gemini-2.5-flash)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated content and metadata
    """
    # Ensure SDK is available (lazy import if needed)
    if not _ensure_genai_available():
        return {
            "success": False,
            "error": "google-genai SDK not available",
            "install_hint": "pip install google-genai",
        }

    client = _get_gemini_client()
    if not client:
        return {
            "success": False,
            "error": "GEMINI_API_KEY environment variable not set",
            "hint": "Set GEMINI_API_KEY before using this tool",
        }

    try:
        # Build contents list
        contents: list[Any] = []

        # Add file URIs if provided
        if file_uris:
            for file_uri in file_uris:
                # Create file part
                file_part = types.Part.from_uri(
                    file_uri=file_uri,
                    mime_type=_get_mime_type_for_uri(file_uri),
                )
                contents.append(file_part)

        # Add text prompt
        contents.append(prompt)

        # Build generation config
        config_kwargs: dict[str, Any] = {}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        generation_config = (
            types.GenerateContentConfig(**config_kwargs)
            if config_kwargs
            else None
        )

        # Generate content
        logger.info(
            f"Generating content with model: {model}",
            extra={"prompt_length": len(prompt), "file_count": len(file_uris or [])},
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generation_config,
        )

        # Extract text from response
        text = response.text

        # Extract usage metadata if available
        usage = {}
        if hasattr(response, "usage_metadata"):
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", None),
                "candidates_tokens": getattr(response.usage_metadata, "candidates_token_count", None),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", None),
            }

        logger.info("Content generated successfully")

        return {
            "success": True,
            "text": text,
            "model": model,
            "usage": usage,
            "message": "Content generated successfully",
        }

    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        return {
            "success": False,
            "error": f"Generation failed: {str(e)}",
        }


# ===================================================================
# Tool 3: Start Chat
# ===================================================================
def start_chat(
    session_id: str,
    initial_message: str,
    file_uris: list[str] | None = None,
    model: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    """Start a new chat session with Gemini.

    Args:
        session_id: Unique identifier for this chat session
        initial_message: First message to send
        file_uris: Optional list of Gemini file URIs to include as context
        model: Gemini model to use (default: gemini-2.5-flash)

    Returns:
        Chat session details and first response
    """
    # Ensure SDK is available (lazy import if needed)
    if not _ensure_genai_available():
        return {
            "success": False,
            "error": "google-genai SDK not available",
            "install_hint": "pip install google-genai",
        }

    client = _get_gemini_client()
    if not client:
        return {
            "success": False,
            "error": "GEMINI_API_KEY environment variable not set",
            "hint": "Set GEMINI_API_KEY before using this tool",
        }

    # Check if session already exists
    if session_id in CHAT_SESSIONS:
        return {
            "success": False,
            "error": f"Chat session already exists: {session_id}",
            "hint": "Use a different session_id or end the existing session first",
        }

    try:
        # Create chat session
        logger.info(
            f"Starting chat session: {session_id}",
            extra={"model": model, "has_files": bool(file_uris)},
        )

        chat = client.chats.create(model=model)

        # Build initial message with files if provided
        message_parts: list[Any] = []

        if file_uris:
            for file_uri in file_uris:
                file_part = types.Part.from_uri(
                    file_uri=file_uri,
                    mime_type=_get_mime_type_for_uri(file_uri),
                )
                message_parts.append(file_part)

        message_parts.append(initial_message)

        # Send initial message
        response = chat.send_message(message_parts)

        # Extract response text
        response_text = response.text

        # Create session object
        session = ChatSession(
            session_id=session_id,
            model=model,
            chat=chat,
            created_at=datetime.now(),
            message_count=1,
        )

        # Store in registry
        CHAT_SESSIONS[session_id] = session

        logger.info(
            f"Chat session started: {session_id}",
            extra={"session_id": session_id},
        )

        return {
            "success": True,
            "session_id": session_id,
            "response": response_text,
            "model": model,
            "message_count": 1,
            "message": f"Chat session started: {session_id}",
        }

    except Exception as e:
        logger.error(f"Failed to start chat: {e}")
        return {
            "success": False,
            "error": f"Chat start failed: {str(e)}",
        }


# ===================================================================
# Tool 4: Send Message
# ===================================================================
def send_message(
    session_id: str,
    message: str,
    file_uris: list[str] | None = None,
) -> dict[str, Any]:
    """Send a message in an existing chat session.

    Args:
        session_id: ID of existing chat session
        message: Message text to send
        file_uris: Optional list of Gemini file URIs to include with message

    Returns:
        Response from Gemini and updated message count
    """
    # Ensure SDK is available (lazy import if needed)
    if not _ensure_genai_available():
        return {
            "success": False,
            "error": "google-genai SDK not available",
            "install_hint": "pip install google-genai",
        }

    client = _get_gemini_client()
    if not client:
        return {
            "success": False,
            "error": "GEMINI_API_KEY environment variable not set",
            "hint": "Set GEMINI_API_KEY before using this tool",
        }

    # Check if session exists
    if session_id not in CHAT_SESSIONS:
        return {
            "success": False,
            "error": f"Chat session not found: {session_id}",
            "hint": "Use start_chat to create a new session first",
        }

    try:
        # Get session
        session = CHAT_SESSIONS[session_id]

        logger.info(
            f"Sending message to chat session: {session_id}",
            extra={
                "session_id": session_id,
                "message_length": len(message),
                "has_files": bool(file_uris),
            },
        )

        # Build message parts with optional file context
        message_parts: list[Any] = []

        if file_uris:
            for file_uri in file_uris:
                file_part = types.Part.from_uri(
                    file_uri=file_uri,
                    mime_type=_get_mime_type_for_uri(file_uri),
                )
                message_parts.append(file_part)

        message_parts.append(message)

        # Send message to existing chat
        response = session.chat.send_message(message_parts)

        # Extract response text
        response_text = response.text

        # Increment message count
        session.message_count += 1

        logger.info(
            f"Message sent successfully to session: {session_id}",
            extra={
                "session_id": session_id,
                "message_count": session.message_count,
            },
        )

        return {
            "success": True,
            "session_id": session_id,
            "response": response_text,
            "message_count": session.message_count,
            "message": f"Message sent to session: {session_id}",
        }

    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        return {
            "success": False,
            "error": f"Send message failed: {str(e)}",
        }


# ===================================================================
# Tool 5: List Files
# ===================================================================
def list_files() -> dict[str, Any]:
    """List all uploaded files with metadata.

    Returns:
        List of all non-expired uploaded files with their metadata
    """
    # Ensure SDK is available (lazy import if needed)
    if not _ensure_genai_available():
        return {
            "success": False,
            "error": "google-genai SDK not available",
            "install_hint": "pip install google-genai",
        }

    client = _get_gemini_client()
    if not client:
        return {
            "success": False,
            "error": "GEMINI_API_KEY environment variable not set",
            "hint": "Set GEMINI_API_KEY before using this tool",
        }

    try:
        logger.info("Listing uploaded files")

        # Get current time for expiration check
        now = datetime.now()

        # Collect non-expired files
        files = []
        expired_files = []

        for file_id, file_data in FILE_REGISTRY.items():
            # Parse expiration timestamp
            expires_at = datetime.fromisoformat(file_data["expires_at"])

            # Check if file is expired
            if now > expires_at:
                expired_files.append(file_id)
                continue

            # Add to result list
            files.append({
                "file_id": file_id,
                "file_name": file_data["name"],
                "display_name": file_data["display_name"],
                "size": file_data["size"],
                "mime_type": file_data["mime_type"],
                "uploaded_at": file_data["uploaded_at"],
                "expires_at": file_data["expires_at"],
                "uri": file_data["uri"],
            })

        # Remove expired files from registry
        for file_id in expired_files:
            del FILE_REGISTRY[file_id]
            logger.info(f"Removed expired file from registry: {file_id}")

        logger.info(
            f"Listed {len(files)} files",
            extra={"file_count": len(files), "expired_count": len(expired_files)},
        )

        return {
            "success": True,
            "files": files,
            "count": len(files),
            "expired_removed": len(expired_files),
            "message": f"Found {len(files)} uploaded files",
        }

    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return {
            "success": False,
            "error": f"List files failed: {str(e)}",
        }


# ===================================================================
# Tool 6: Delete File
# ===================================================================
def delete_file(file_name: str) -> dict[str, Any]:
    """Delete a file from Gemini Files API and local registry.

    Args:
        file_name: The Gemini file name (e.g., 'files/abc123')

    Returns:
        Deletion confirmation
    """
    # Ensure SDK is available (lazy import if needed)
    if not _ensure_genai_available():
        return {
            "success": False,
            "error": "google-genai SDK not available",
            "install_hint": "pip install google-genai",
        }

    client = _get_gemini_client()
    if not client:
        return {
            "success": False,
            "error": "GEMINI_API_KEY environment variable not set",
            "hint": "Set GEMINI_API_KEY before using this tool",
        }

    # Find file in registry by Gemini file name
    file_id = None
    for fid, file_data in FILE_REGISTRY.items():
        if file_data["name"] == file_name:
            file_id = fid
            break

    if file_id is None:
        return {
            "success": False,
            "error": f"File not found in registry: {file_name}",
            "hint": "Use list_files to see available files",
        }

    try:
        logger.info(
            f"Deleting file: {file_name}",
            extra={"file_id": file_id, "gemini_name": file_name},
        )

        # Delete from Gemini Files API
        client.files.delete(name=file_name)

        # Remove from local registry
        del FILE_REGISTRY[file_id]

        logger.info(
            f"File deleted successfully: {file_id}",
            extra={"file_id": file_id, "gemini_name": file_name},
        )

        return {
            "success": True,
            "file_id": file_id,
            "file_name": file_name,
            "message": f"File deleted successfully: {file_name}",
        }

    except Exception as e:
        logger.error(f"File deletion failed: {e}")
        return {
            "success": False,
            "error": f"Deletion failed: {str(e)}",
            "hint": "The file may have already been deleted from Gemini API",
        }


# ===================================================================
# Resource 1: Chat History
# ===================================================================
def get_chat_history(session_id: str) -> dict[str, Any]:
    """Get chat session history and metadata.

    Args:
        session_id: ID of the chat session

    Returns:
        Session metadata including message count and status
    """
    logger.info(f"Retrieving chat history for session: {session_id}")

    # Check if session exists
    if session_id not in CHAT_SESSIONS:
        logger.warning(f"Chat session not found: {session_id}")
        return {
            "success": False,
            "error": f"Chat session not found: {session_id}",
            "session_id": session_id,
        }

    # Get session data
    session = CHAT_SESSIONS[session_id]

    # Build response with session metadata
    # Note: Gemini SDK doesn't provide built-in message history storage,
    # so we return session metadata instead
    response = {
        "success": True,
        "session_id": session.session_id,
        "model": session.model,
        "created_at": session.created_at.isoformat(),
        "message_count": session.message_count,
        "status": "active",
        "message": (
            f"Chat session {session_id} is active "
            f"with {session.message_count} message(s)"
        ),
    }

    logger.info(
        f"Chat history retrieved for session: {session_id}",
        extra={"message_count": session.message_count},
    )

    return response


# ===================================================================
# Resource 2: File Information
# ===================================================================
def get_file_info(file_name: str) -> dict[str, Any]:
    """Get file metadata and status.

    Args:
        file_name: Name of the file (Gemini file name)

    Returns:
        File metadata including upload info and expiration status
    """
    logger.info(f"Retrieving file info for: {file_name}")

    # Search FILE_REGISTRY for matching file
    file_id = None
    file_data = None

    for fid, fdata in FILE_REGISTRY.items():
        if fdata["name"] == file_name:
            file_id = fid
            file_data = fdata
            break

    # File not found
    if file_data is None:
        logger.warning(f"File not found: {file_name}")
        return {
            "success": False,
            "error": f"File not found: {file_name}",
            "file_name": file_name,
        }

    # Determine if file is expired
    expires_at = datetime.fromisoformat(file_data["expires_at"])
    is_expired = expires_at < datetime.now()
    status = "expired" if is_expired else "active"

    # Build response
    response = {
        "success": True,
        "file_id": file_id,
        "file_name": file_data["name"],
        "file_uri": file_data["uri"],
        "display_name": file_data["display_name"],
        "size": file_data["size"],
        "mime_type": file_data["mime_type"],
        "uploaded_at": file_data["uploaded_at"],
        "expires_at": file_data["expires_at"],
        "status": status,
    }

    logger.info(
        f"File info retrieved: {file_name}",
        extra={"file_id": file_id, "status": status},
    )

    return response


# ===================================================================
# Prompt 1: Analyze Media
# ===================================================================
def analyze_media_prompt(media_type: str) -> str:
    """Generate a prompt template for media analysis.

    Args:
        media_type: Type of media - one of: audio, video, image, document

    Returns:
        Customized prompt text for the media type
    """
    templates = {
        "audio": """Analyze this audio file in detail.

Please provide:
1. Content Overview: Describe what the audio contains (speech, music, ambient sounds, etc.)
2. Speakers/Voices: Identify number of speakers and their characteristics if applicable
3. Key Points: Extract main topics, themes, or messages conveyed
4. Audio Quality: Comment on clarity, background noise, recording quality
5. Transcript Highlights: Provide key quotes or important segments if speech is present
6. Duration and Pacing: Note timing of key moments or segments
7. Recommendations: Suggest any follow-up actions or additional analysis needed

Be thorough and specific in your analysis.""",

        "video": """Analyze this video file comprehensively.

Please provide:
1. Content Overview: Describe the overall content and purpose of the video
2. Visual Elements: Detail key visuals, scenes, objects, and people shown
3. Key Moments: Identify and timestamp significant events or transitions
4. Audio Content: Describe any narration, dialogue, music, or sound effects
5. Production Quality: Comment on video quality, lighting, composition, editing
6. Text/Graphics: Note any on-screen text, captions, or graphics
7. Context and Message: Explain the intended message or story being conveyed
8. Recommendations: Suggest improvements or additional analysis needed

Provide timestamps for important moments when possible.""",

        "image": """Analyze this image in comprehensive detail.

Please provide:
1. Subject Matter: Describe the main subject(s) and focus of the image
2. Objects and Elements: List and describe all visible objects, people, or elements
3. Composition: Analyze framing, layout, perspective, and visual balance
4. Colors and Lighting: Describe the color palette, lighting conditions, and mood
5. Text Content: Transcribe and analyze any visible text, signs, or labels
6. Context and Setting: Identify the location, environment, or scenario depicted
7. Quality and Technical Aspects: Comment on resolution, clarity, and technical quality
8. Interpretation: Provide insights into the image's purpose, message, or significance

Be specific and detailed in your observations.""",

        "document": """Analyze this document thoroughly.

Please provide:
1. Document Summary: Provide a concise overview of the document's content and purpose
2. Key Points: List the main points, arguments, or information presented (3-5 bullet points)
3. Structure and Organization: Describe how the document is organized (sections, headings, format)
4. Important Details: Highlight critical data, dates, names, numbers, or facts
5. Tone and Style: Comment on the writing style, intended audience, and tone
6. Key Takeaways: Identify the most important conclusions or action items
7. Questions or Gaps: Note any unclear sections or missing information
8. Recommendations: Suggest how to use or act on this document

Provide specific page references or sections when applicable."""
    }

    template = templates.get(media_type.lower())

    if template:
        return template

    # Fallback for unknown media types
    return f"""Analyze this {media_type} file.

Please provide a comprehensive analysis including:
1. Content overview and description
2. Key elements and components
3. Quality assessment
4. Notable features or details
5. Context and interpretation
6. Recommendations for use or further analysis

Be thorough and specific in your analysis."""


# ===================================================================
# Prompt 2: Document Q&A
# ===================================================================
def document_qa_prompt(question_type: str) -> str:
    """Generate a prompt template for document Q&A.

    Args:
        question_type: Type of question - one of: summary, detailed, extraction

    Returns:
        Customized prompt text for the question type
    """
    templates = {
        "summary": """Summarize this document concisely.

Provide a summary that includes:
1. Main Purpose: What is this document about and why does it exist?
2. Key Points: List 3-5 most important points or findings (use bullet points)
3. Essential Details: Include any critical data, dates, or decisions mentioned
4. Target Audience: Who is this document intended for?
5. Action Items: List any actions, recommendations, or next steps (if applicable)

Keep the summary focused and actionable. Aim for clarity and brevity while capturing all essential information.

Format your response with clear headings and bullet points for easy scanning.""",

        "detailed": """Provide a detailed analysis of this document.

Your analysis should include:
1. Executive Summary: Brief overview of the document (2-3 sentences)
2. Background and Context: What circumstances or needs led to this document?
3. Main Content Analysis:
   - Break down each major section or argument
   - Explain key concepts, data, or findings in depth
   - Identify relationships between different parts
4. Supporting Evidence: What data, citations, or examples are used?
5. Strengths and Weaknesses: Evaluate the quality and completeness of the information
6. Implications: What are the consequences or significance of this document?
7. Questions and Considerations: What additional information might be needed?
8. Recommendations: Based on the content, what actions or decisions are suggested?

Be comprehensive and analytical. Use specific quotes and references from the document to support your analysis.""",

        "extraction": """Extract and list all key information from this document.

Please extract:
1. Factual Data:
   - Names of people, organizations, or entities
   - Dates, times, and deadlines
   - Numbers, statistics, financial figures
   - Locations and addresses
2. Key Terms and Definitions: Important concepts or terminology defined
3. Action Items and Tasks: Any assigned actions, responsibilities, or deliverables
4. Decisions and Conclusions: Explicit decisions made or conclusions reached
5. References and Citations: External sources, documents, or materials mentioned
6. Contact Information: Any emails, phone numbers, or other contact details
7. Requirements or Specifications: Lists of requirements, criteria, or specifications
8. Deadlines and Milestones: Time-sensitive information or project timelines

Present the extracted information in a structured format with clear categories and bullet points. Include page numbers or section references where the information appears.

Focus on factual extraction - do not interpret or summarize, just list what is explicitly stated."""
    }

    template = templates.get(question_type.lower())

    if template:
        return template

    # Fallback for unknown question types
    return """Answer questions about this document.

Please provide clear, accurate answers based on the document content. Support your answers with:
1. Direct quotes or references from the document
2. Specific page numbers or sections
3. Context for your interpretations
4. Acknowledgment of any ambiguities or limitations

Be precise and cite your sources within the document."""


# ===================================================================
# Prompt 3: Multimodal Analysis
# ===================================================================
def multimodal_analysis_prompt(analysis_type: str) -> str:
    """Generate a prompt template for multimodal analysis.

    Args:
        analysis_type: Analysis approach - e.g., "compare", "synthesize", "timeline"

    Returns:
        Customized prompt text for the analysis type
    """
    templates = {
        "compare": """Compare and contrast the provided files.

Your comparison should include:
1. Overview: Brief description of each file and what it contains
2. Similarities: What common themes, information, or elements appear across files?
   - Shared content or topics
   - Consistent data or findings
   - Common formats or structures
3. Differences: How do the files differ from each other?
   - Unique information in each file
   - Conflicting data or perspectives
   - Different approaches or formats
4. Complementary Aspects: How do the files work together or complement each other?
5. Gaps and Inconsistencies: Note any missing information or contradictions
6. Quality Assessment: Compare the quality, depth, or reliability of each file
7. Overall Analysis: What insights emerge from comparing these files together?
8. Recommendations: How should these files be used together?

Present your comparison in a structured format with clear sections. Use tables or side-by-side comparisons where helpful.""",

        "synthesize": """Synthesize information from all provided files into a unified analysis.

Create a comprehensive synthesis that:
1. Integrated Overview: Combine information from all files into a cohesive narrative
2. Unified Key Points: Extract and merge the most important points across all files
3. Complete Picture: What complete story or understanding emerges from all files together?
4. Cross-Referenced Information: Connect related information across different files
5. Resolved Conflicts: Address any contradictions or discrepancies between files
6. Enhanced Insights: What new insights or conclusions can be drawn from the combined information?
7. Comprehensive Timeline: If applicable, create a timeline integrating events from all files
8. Consolidated Recommendations: Synthesize action items or recommendations across files
9. Knowledge Gaps: Identify what information is still missing even after reviewing all files

Present the synthesis as a unified document that reads cohesively, clearly indicating when you're drawing from multiple sources.""",

        "timeline": """Create a timeline or sequence from the provided files.

Construct a chronological timeline that includes:
1. Timeline Overview: Describe the time span and scope covered by all files
2. Chronological Events: List events, developments, or milestones in order
   - Date/time of each event
   - Description of what happened
   - Source file for each event
   - Significance or impact
3. Key Periods: Identify important phases or periods in the timeline
4. Relationships and Dependencies: Show how events connect or depend on each other
5. Parallel Activities: Note events happening simultaneously across different files
6. Gaps in Timeline: Identify missing periods or unclear sequences
7. Visual Timeline: Create a structured representation of the timeline (use formatting)
8. Analysis: What patterns, trends, or insights emerge from the chronological view?
9. Critical Path: Identify the most important sequence of events

Format the timeline clearly with dates/times on the left and descriptions on the right. Use consistent date formatting and clear markers for different types of events."""
    }

    template = templates.get(analysis_type.lower())

    if template:
        return template

    # Fallback for unknown analysis types
    return f"""Perform a {analysis_type} analysis across all provided files.

Your multi-file analysis should:
1. Examine each file individually first
2. Identify connections and relationships between files
3. Apply the {analysis_type} approach to combine insights
4. Present findings in a clear, structured format
5. Reference specific files when making claims
6. Provide actionable conclusions

Be thorough and ensure your analysis adds value by combining information from multiple sources."""


# ===================================================================
# Server Factory Function
# ===================================================================
def create_gemini_server() -> BuildMCPServer:
    """Create and configure the Gemini MCP server.

    Returns:
        Configured BuildMCPServer instance
    """
    # Create server instance
    mcp = BuildMCPServer(
        name="gemini-server",
        version="0.1.0",
        description="Gemini MCP Server with file upload, content generation, and chat capabilities",
    )

    # Register tools
    mcp.add_tool(
        "upload_file",
        upload_file,
        description="Upload a file to Gemini Files API for use in prompts",
    )

    mcp.add_tool(
        "generate_content",
        generate_content,
        description="Generate content using Gemini with optional file context",
    )

    mcp.add_tool(
        "start_chat",
        start_chat,
        description="Start a new chat session with Gemini and send initial message",
    )

    mcp.add_tool(
        "send_message",
        send_message,
        description="Send a message in an existing chat session",
    )

    mcp.add_tool(
        "list_files",
        list_files,
        description="List all uploaded files in the Gemini Files API",
    )

    mcp.add_tool(
        "delete_file",
        delete_file,
        description="Delete a file from Gemini Files API",
    )

    # Register resources
    mcp.add_resource(
        "chat-history://{session_id}",
        get_chat_history,
        name="Get chat history",
        description="Retrieve message history for a chat session",
    )

    mcp.add_resource(
        "file-info://{file_name}",
        get_file_info,
        name="Get file information",
        description="Retrieve metadata for an uploaded file",
    )

    # Register prompts
    mcp.add_prompt(
        "analyze_media",
        analyze_media_prompt,
        description="Template for analyzing uploaded media files",
        arguments=["media_type"],
    )

    mcp.add_prompt(
        "document_qa",
        document_qa_prompt,
        description="Template for document question-answering",
        arguments=["question_type"],
    )

    mcp.add_prompt(
        "multimodal_analysis",
        multimodal_analysis_prompt,
        description="Template for analyzing multiple files together",
        arguments=["analysis_type"],
    )

    return mcp


# Module-level instance for CLI detection
# This allows the server to be discovered and packaged by simply-mcp build command
mcp = create_gemini_server()


