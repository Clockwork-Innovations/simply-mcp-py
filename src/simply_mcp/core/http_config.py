"""HTTP Transport Configuration System for Simply-MCP.

This module provides production-grade configuration management for HTTP transport:
- YAML/TOML configuration file support
- Environment variable precedence
- Structured validation with Pydantic
- Multi-environment support (dev/staging/prod)
- Configuration with sensible defaults

Configuration Priority (highest to lowest):
1. Environment variables
2. Config file (YAML/TOML)
3. Default values

Example:
    >>> config = HttpConfig.from_file("config.yaml")
    >>> config = HttpConfig.from_env()
    >>> config = HttpConfig()  # Use defaults
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Handle TOML library imports - for type checking assume tomllib is available
if TYPE_CHECKING:
    import tomllib as tomli
    TOMLI_AVAILABLE: bool
else:
    # At runtime: try tomli first (backport for Python <3.11), fall back to tomllib
    try:
        import tomli

        TOMLI_AVAILABLE = True
    except ImportError:
        try:
            import tomllib as tomli  # type: ignore[import-not-found]

            TOMLI_AVAILABLE = True
        except ImportError:
            TOMLI_AVAILABLE = False
            tomli = None  # type: ignore[assignment]

# Handle YAML library imports
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore[assignment]

from pydantic import BaseModel, Field, field_validator


class TLSConfig(BaseModel):
    """TLS/HTTPS configuration.

    Attributes:
        enabled: Enable HTTPS/TLS
        cert_file: Path to SSL certificate file
        key_file: Path to SSL private key file
        ca_file: Optional path to CA certificate file
        verify_client: Require client certificate verification
    """
    enabled: bool = False
    cert_file: str | None = None
    key_file: str | None = None
    ca_file: str | None = None
    verify_client: bool = False

    @field_validator("cert_file", "key_file", "ca_file")
    @classmethod
    def validate_file_exists(cls, v: str | None) -> str | None:
        """Validate that certificate files exist if provided."""
        if v and not Path(v).exists():
            raise ValueError(f"Certificate file not found: {v}")
        return v


class AuthConfig(BaseModel):
    """Authentication configuration.

    Attributes:
        enabled: Enable authentication
        keys_file: Path to API keys file (JSON/YAML)
        key_env_var: Environment variable name for single API key
        require_auth: Require authentication on all endpoints
    """
    enabled: bool = False
    keys_file: str | None = None
    key_env_var: str = "MCP_API_KEY"
    require_auth: bool = True


class RateLimitConfig(BaseModel):
    """Rate limiting configuration.

    Attributes:
        enabled: Enable rate limiting
        default_limit: Default requests per window
        window_seconds: Time window in seconds
        strategy: Rate limit strategy (token_bucket, sliding_window)
    """
    enabled: bool = False
    default_limit: int = Field(default=100, ge=1)
    window_seconds: int = Field(default=60, ge=1)
    strategy: str = Field(default="token_bucket", pattern="^(token_bucket|sliding_window)$")


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration.

    Attributes:
        prometheus_enabled: Enable Prometheus metrics
        prometheus_path: Metrics endpoint path
        health_path: Health check endpoint path
        tracing_enabled: Enable distributed tracing
        log_requests: Log all HTTP requests
        log_responses: Log all HTTP responses
    """
    prometheus_enabled: bool = True
    prometheus_path: str = "/metrics"
    health_path: str = "/health"
    tracing_enabled: bool = False
    log_requests: bool = True
    log_responses: bool = False


class CORSConfig(BaseModel):
    """CORS configuration.

    Attributes:
        enabled: Enable CORS
        allow_origins: Allowed origins (* for all)
        allow_methods: Allowed HTTP methods
        allow_headers: Allowed headers
        allow_credentials: Allow credentials
        max_age: Preflight cache duration in seconds
    """
    enabled: bool = True
    allow_origins: list[str] = ["*"]
    allow_methods: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers: list[str] = ["*"]
    allow_credentials: bool = False
    max_age: int = 3600


class SecurityConfig(BaseModel):
    """Security configuration.

    Attributes:
        security_headers: Enable security headers
        hsts_enabled: Enable HSTS header
        hsts_max_age: HSTS max age in seconds
        content_type_nosniff: Enable X-Content-Type-Options
        xss_protection: Enable X-XSS-Protection
        frame_options: X-Frame-Options value
        max_request_size: Maximum request body size in bytes
        request_timeout: Request timeout in seconds
    """
    security_headers: bool = True
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    content_type_nosniff: bool = True
    xss_protection: bool = True
    frame_options: str = "DENY"
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 30


class LoggingConfig(BaseModel):
    """Logging configuration.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format (json, text)
        enable_console: Enable console logging
        file: Optional log file path
        structured: Use structured JSON logging
    """
    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = Field(default="json", pattern="^(json|text)$")
    enable_console: bool = True
    file: str | None = None
    structured: bool = True


class ServerConfig(BaseModel):
    """Server configuration.

    Attributes:
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes (0 = auto)
        worker_connections: Max concurrent connections per worker
        backlog: Socket backlog size
        keepalive: Connection keepalive timeout in seconds
        graceful_timeout: Graceful shutdown timeout in seconds
    """
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=0, ge=0)
    worker_connections: int = Field(default=1000, ge=1)
    backlog: int = Field(default=2048, ge=1)
    keepalive: int = Field(default=5, ge=0)
    graceful_timeout: int = Field(default=30, ge=1)


class HttpConfig(BaseModel):
    """Complete HTTP transport configuration.

    This is the main configuration class that combines all configuration sections.
    It provides factory methods for loading from files and environment variables.

    Attributes:
        environment: Deployment environment (dev, staging, prod)
        server: Server configuration
        tls: TLS/HTTPS configuration
        auth: Authentication configuration
        rate_limit: Rate limiting configuration
        monitoring: Monitoring configuration
        cors: CORS configuration
        security: Security configuration
        logging: Logging configuration
    """
    environment: str = "development"
    server: ServerConfig = Field(default_factory=ServerConfig)
    tls: TLSConfig = Field(default_factory=TLSConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "HttpConfig":
        """Load configuration from YAML or TOML file.

        Args:
            path: Path to configuration file

        Returns:
            HttpConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If file format is not supported or parsing fails

        Example:
            >>> config = HttpConfig.from_file("config.yaml")
            >>> config = HttpConfig.from_file("config.toml")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        # Determine file type
        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required to load YAML config files. "
                    "Install with: pip install pyyaml"
                )
            with open(path) as f:
                data = yaml.safe_load(f)

        elif suffix == ".toml":
            if not TOMLI_AVAILABLE:
                raise ImportError(
                    "tomli is required to load TOML config files on Python <3.11. "
                    "Install with: pip install tomli"
                )
            with open(path, "rb") as f:
                data = tomli.load(f)

        else:
            raise ValueError(f"Unsupported config file format: {suffix}")

        # Apply environment variable overrides
        data = cls._apply_env_overrides(data)

        return cls(**data)

    @classmethod
    def from_env(cls, prefix: str = "MCP_HTTP_") -> "HttpConfig":
        """Load configuration from environment variables.

        Environment variables should be prefixed (default: MCP_HTTP_) and
        use double underscores for nested keys.

        Args:
            prefix: Environment variable prefix

        Returns:
            HttpConfig instance

        Example:
            >>> os.environ["MCP_HTTP_SERVER__PORT"] = "9000"
            >>> os.environ["MCP_HTTP_AUTH__ENABLED"] = "true"
            >>> config = HttpConfig.from_env()
        """
        data: dict[str, Any] = {}

        # Parse environment variables
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Remove prefix and split by double underscore
            key_path = key[len(prefix):].lower().split("__")

            # Navigate/create nested dict structure
            current = data
            for part in key_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set value with type conversion
            current[key_path[-1]] = cls._convert_env_value(value)

        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HttpConfig":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            HttpConfig instance
        """
        # Apply environment variable overrides
        data = cls._apply_env_overrides(data)
        return cls(**data)

    @classmethod
    def _apply_env_overrides(cls, data: dict[str, Any], prefix: str = "MCP_HTTP_") -> dict[str, Any]:
        """Apply environment variable overrides to config data.

        Args:
            data: Base configuration data
            prefix: Environment variable prefix

        Returns:
            Configuration data with env overrides applied
        """
        result = data.copy()

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Remove prefix and split by double underscore
            key_path = key[len(prefix):].lower().split("__")

            # Navigate/create nested dict structure
            current = result
            for part in key_path[:-1]:
                if part not in current:
                    current[part] = {}
                if not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]

            # Set value with type conversion
            current[key_path[-1]] = cls._convert_env_value(value)

        return result

    @staticmethod
    def _convert_env_value(value: str) -> Any:
        """Convert environment variable string to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Converted value (bool, int, list, or string)
        """
        # Boolean conversion
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass

        # List conversion (comma-separated)
        if "," in value:
            return [v.strip() for v in value.split(",")]

        # String
        return value

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.model_dump()

    def to_yaml(self, path: str | Path) -> None:
        """Export configuration to YAML file.

        Args:
            path: Output file path

        Raises:
            ImportError: If PyYAML is not installed
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to export YAML config. "
                "Install with: pip install pyyaml"
            )

        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


__all__ = [
    "HttpConfig",
    "ServerConfig",
    "TLSConfig",
    "AuthConfig",
    "RateLimitConfig",
    "MonitoringConfig",
    "CORSConfig",
    "SecurityConfig",
    "LoggingConfig",
]
