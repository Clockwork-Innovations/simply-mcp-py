"""Storage Configuration - Polish Layer

Centralized configuration management for storage backends.

Supports:
- SQLite (file-based, no server)
- PostgreSQL (enterprise SQL)
- MongoDB (document database)

Configuration can be loaded from:
- Environment variables
- YAML/TOML configuration files
- Programmatic settings

Usage:
    # From environment
    config = StorageConfig.from_env()
    storage = create_storage(config)

    # From file
    config = StorageConfig.from_file("config.yaml")
    storage = create_storage(config)

    # Programmatic
    config = StorageConfig(
        backend="postgresql",
        connection_string="postgresql://localhost/gemini"
    )
    storage = create_storage(config)
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

try:
    import tomli
except ImportError:
    import tomllib as tomli  # Python 3.11+

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from .base import SessionStorage

# Optional: SQLite storage
try:
    from .sqlite import SQLiteSessionStorage
    _HAS_SQLITE = True
except ImportError:
    SQLiteSessionStorage = None  # type: ignore
    _HAS_SQLITE = False

logger = logging.getLogger(__name__)


BackendType = Literal["sqlite", "postgresql", "mongodb"]


@dataclass
class StorageConfig:
    """Storage configuration.

    Attributes:
        backend: Storage backend type
        connection_string: Connection string (for PostgreSQL/MongoDB)
        database_path: Database path (for SQLite)
        min_pool_size: Minimum connection pool size
        max_pool_size: Maximum connection pool size
        auto_save: Enable auto-save
        expiry_hours: Session expiry in hours
        options: Additional backend-specific options
    """

    backend: BackendType = "sqlite"
    connection_string: Optional[str] = None
    database_path: str = "gemini_sessions.db"
    min_pool_size: int = 5
    max_pool_size: int = 20
    auto_save: bool = True
    expiry_hours: int = 72
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create configuration from environment variables.

        Environment variables:
            STORAGE_BACKEND: Backend type (sqlite, postgresql, mongodb)
            STORAGE_CONNECTION_STRING: Connection string
            STORAGE_DATABASE_PATH: Database path (SQLite)
            STORAGE_MIN_POOL_SIZE: Minimum pool size
            STORAGE_MAX_POOL_SIZE: Maximum pool size
            STORAGE_AUTO_SAVE: Enable auto-save (true/false)
            STORAGE_EXPIRY_HOURS: Session expiry hours

        Returns:
            Configuration instance
        """
        backend = os.getenv("STORAGE_BACKEND", "sqlite")
        connection_string = os.getenv("STORAGE_CONNECTION_STRING")
        database_path = os.getenv("STORAGE_DATABASE_PATH", "gemini_sessions.db")

        # Parse pool sizes
        min_pool_size = int(os.getenv("STORAGE_MIN_POOL_SIZE", "5"))
        max_pool_size = int(os.getenv("STORAGE_MAX_POOL_SIZE", "20"))

        # Parse boolean
        auto_save_str = os.getenv("STORAGE_AUTO_SAVE", "true").lower()
        auto_save = auto_save_str in ("true", "1", "yes", "on")

        # Parse expiry
        expiry_hours = int(os.getenv("STORAGE_EXPIRY_HOURS", "72"))

        return cls(
            backend=backend,  # type: ignore
            connection_string=connection_string,
            database_path=database_path,
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            auto_save=auto_save,
            expiry_hours=expiry_hours,
        )

    @classmethod
    def from_file(cls, config_path: Path | str) -> "StorageConfig":
        """Create configuration from file.

        Supports YAML and TOML formats.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration instance

        Raises:
            ValueError: If file format not supported
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load file
        if config_path.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("PyYAML required for YAML config. Install: pip install pyyaml")
            with open(config_path) as f:
                data = yaml.safe_load(f)
        elif config_path.suffix == ".toml":
            with open(config_path, "rb") as f:
                data = tomli.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        # Extract storage config
        storage_config = data.get("storage", {})

        return cls(
            backend=storage_config.get("backend", "sqlite"),
            connection_string=storage_config.get("connection_string"),
            database_path=storage_config.get("database_path", "gemini_sessions.db"),
            min_pool_size=storage_config.get("min_pool_size", 5),
            max_pool_size=storage_config.get("max_pool_size", 20),
            auto_save=storage_config.get("auto_save", True),
            expiry_hours=storage_config.get("expiry_hours", 72),
            options=storage_config.get("options", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "backend": self.backend,
            "connection_string": self.connection_string,
            "database_path": self.database_path,
            "min_pool_size": self.min_pool_size,
            "max_pool_size": self.max_pool_size,
            "auto_save": self.auto_save,
            "expiry_hours": self.expiry_hours,
            "options": self.options,
        }


def create_storage(config: StorageConfig) -> SessionStorage:
    """Create storage backend from configuration.

    Args:
        config: Storage configuration

    Returns:
        Initialized storage backend

    Raises:
        ValueError: If backend type not supported
        ImportError: If required dependencies not installed
    """
    if config.backend == "sqlite":
        if not _HAS_SQLITE or SQLiteSessionStorage is None:
            raise ImportError(
                "SQLite backend requires aiosqlite. "
                "Install with: pip install aiosqlite"
            )
        logger.info(f"Creating SQLite storage: {config.database_path}")
        return SQLiteSessionStorage(config.database_path)

    elif config.backend == "postgresql":
        if not config.connection_string:
            raise ValueError("PostgreSQL requires connection_string")

        logger.info("Creating PostgreSQL storage")
        from .postgresql import PostgreSQLSessionStorage

        return PostgreSQLSessionStorage(
            config.connection_string,
            min_pool_size=config.min_pool_size,
            max_pool_size=config.max_pool_size,
        )

    elif config.backend == "mongodb":
        if not config.connection_string:
            raise ValueError("MongoDB requires connection_string")

        logger.info("Creating MongoDB storage")
        from .mongodb import MongoDBSessionStorage

        # Extract database name from connection string or use default
        database_name = config.options.get("database_name", "gemini_mcp")

        return MongoDBSessionStorage(
            config.connection_string,
            database_name=database_name,
        )

    else:
        raise ValueError(f"Unsupported backend: {config.backend}")


async def test_connection(storage: SessionStorage) -> dict[str, Any]:
    """Test storage backend connection.

    Args:
        storage: Storage backend to test

    Returns:
        Connection test results
    """
    try:
        await storage.initialize()

        # Try basic operations
        from .base import ChatSession

        test_session = ChatSession.create("test-model")
        test_session.add_message("user", "Test message")

        # Save and load
        save_success = await storage.save_session(test_session)
        loaded_session = await storage.load_session(test_session.session_id)

        # Cleanup
        await storage.delete_session(test_session.session_id)
        await storage.close()

        return {
            "success": True,
            "save_successful": save_success,
            "load_successful": loaded_session is not None,
            "message": "Connection test passed",
        }

    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Connection test failed",
        }


def get_example_config(backend: BackendType) -> dict[str, Any]:
    """Get example configuration for a backend.

    Args:
        backend: Backend type

    Returns:
        Example configuration dictionary
    """
    if backend == "sqlite":
        return {
            "storage": {
                "backend": "sqlite",
                "database_path": "./gemini_sessions.db",
                "auto_save": True,
                "expiry_hours": 72,
            }
        }

    elif backend == "postgresql":
        return {
            "storage": {
                "backend": "postgresql",
                "connection_string": "postgresql://user:password@localhost:5432/gemini_mcp",
                "min_pool_size": 5,
                "max_pool_size": 20,
                "auto_save": True,
                "expiry_hours": 72,
            }
        }

    elif backend == "mongodb":
        return {
            "storage": {
                "backend": "mongodb",
                "connection_string": "mongodb://localhost:27017",
                "options": {"database_name": "gemini_mcp"},
                "auto_save": True,
                "expiry_hours": 72,
            }
        }

    else:
        raise ValueError(f"Unknown backend: {backend}")
