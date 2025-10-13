"""Core type definitions for Simply-MCP.

This module defines the fundamental types used throughout the Simply-MCP framework,
including tool configurations, prompt definitions, resource specifications, and
server metadata.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, TypedDict, Union
from typing_extensions import NotRequired

# Type aliases for common patterns
JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONDict = Dict[str, JSONValue]
HandlerFunction = Callable[..., Any]


class ToolConfig(TypedDict):
    """Configuration for a tool.

    Tools are executable functions that the MCP server exposes to clients.
    They can perform actions, computations, and have side effects.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        input_schema: JSON Schema defining the tool's input parameters
        handler: The function that implements the tool's logic
        metadata: Optional additional metadata about the tool
    """

    name: str
    description: str
    input_schema: JSONDict
    handler: HandlerFunction
    metadata: NotRequired[Dict[str, Any]]


class PromptConfig(TypedDict):
    """Configuration for a prompt template.

    Prompts define interaction templates that help structure communication
    with language models.

    Attributes:
        name: Unique identifier for the prompt
        description: Human-readable description of the prompt's purpose
        arguments: Optional list of argument names the prompt accepts
        template: The prompt template string (may include placeholders)
        handler: Optional function to dynamically generate the prompt
        metadata: Optional additional metadata about the prompt
    """

    name: str
    description: str
    arguments: NotRequired[List[str]]
    template: NotRequired[str]
    handler: NotRequired[HandlerFunction]
    metadata: NotRequired[Dict[str, Any]]


class ResourceConfig(TypedDict):
    """Configuration for a resource.

    Resources represent data or content that the server can provide,
    such as files, configurations, or computed values.

    Attributes:
        uri: Unique URI identifying the resource (e.g., "file:///path", "config://name")
        name: Human-readable name for the resource
        description: Human-readable description of what the resource provides
        mime_type: MIME type of the resource content (e.g., "application/json")
        handler: Function that returns the resource content
        metadata: Optional additional metadata about the resource
    """

    uri: str
    name: str
    description: str
    mime_type: str
    handler: HandlerFunction
    metadata: NotRequired[Dict[str, Any]]


class ServerMetadata(TypedDict):
    """Metadata about the MCP server.

    Attributes:
        name: Server name
        version: Server version (semver format recommended)
        description: Optional description of the server's purpose
        author: Optional author information
        homepage: Optional homepage URL
    """

    name: str
    version: str
    description: NotRequired[str]
    author: NotRequired[str]
    homepage: NotRequired[str]


# Transport types
TransportType = Literal["stdio", "http", "sse"]


class TransportConfig(TypedDict):
    """Configuration for a transport layer.

    Attributes:
        type: Type of transport to use
        host: Host address for network transports (http/sse)
        port: Port number for network transports (http/sse)
        path: Optional path prefix for HTTP endpoints
    """

    type: TransportType
    host: NotRequired[str]
    port: NotRequired[int]
    path: NotRequired[str]


# Progress reporting types
class ProgressUpdate(TypedDict):
    """Progress update information.

    Attributes:
        percentage: Progress as a percentage (0-100)
        message: Optional human-readable status message
        current: Optional current step number
        total: Optional total number of steps
    """

    percentage: float
    message: NotRequired[str]
    current: NotRequired[int]
    total: NotRequired[int]


class ProgressReporter(Protocol):
    """Protocol for progress reporting.

    This defines the interface for reporting progress during long-running operations.
    """

    async def update(
        self,
        percentage: float,
        message: Optional[str] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> None:
        """Update progress.

        Args:
            percentage: Progress as a percentage (0-100)
            message: Optional status message
            current: Optional current step number
            total: Optional total number of steps
        """
        ...


# Context types
class RequestContext(TypedDict):
    """Context information for a request.

    Attributes:
        request_id: Unique identifier for the request
        session_id: Optional session identifier
        user_id: Optional user identifier
        metadata: Additional request metadata
    """

    request_id: str
    session_id: NotRequired[str]
    user_id: NotRequired[str]
    metadata: NotRequired[Dict[str, Any]]


class HandlerContext(Protocol):
    """Protocol for handler execution context.

    This provides handlers with access to request information, progress reporting,
    and server utilities.
    """

    @property
    def request(self) -> RequestContext:
        """Get the current request context."""
        ...

    @property
    def progress(self) -> Optional[ProgressReporter]:
        """Get the progress reporter if available."""
        ...

    @property
    def server(self) -> Any:  # Avoid circular import
        """Get reference to the server instance."""
        ...


# API Style detection
APIStyle = Literal["decorator", "functional", "interface", "builder"]


class APIStyleInfo(TypedDict):
    """Information about detected API style.

    Attributes:
        style: The detected API style
        confidence: Confidence level (0.0-1.0)
        indicators: List of indicators that led to this detection
    """

    style: APIStyle
    confidence: float
    indicators: List[str]


# Validation types
class ValidationError(TypedDict):
    """Validation error information.

    Attributes:
        field: Field name that failed validation
        message: Error message
        code: Optional error code
        context: Optional additional context
    """

    field: str
    message: str
    code: NotRequired[str]
    context: NotRequired[Dict[str, Any]]


class ValidationResult(TypedDict):
    """Result of a validation operation.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors (if any)
    """

    valid: bool
    errors: NotRequired[List[ValidationError]]


# Security types
class RateLimitConfig(TypedDict):
    """Rate limiting configuration.

    Attributes:
        enabled: Whether rate limiting is enabled
        requests_per_minute: Maximum requests per minute
        burst_size: Maximum burst size (token bucket)
    """

    enabled: bool
    requests_per_minute: int
    burst_size: NotRequired[int]


AuthType = Literal["api_key", "oauth", "jwt", "none"]


class AuthConfig(TypedDict):
    """Authentication configuration.

    Attributes:
        type: Type of authentication
        enabled: Whether authentication is enabled
        api_keys: List of valid API keys (for api_key type)
        oauth_config: OAuth configuration (for oauth type)
        jwt_config: JWT configuration (for jwt type)
    """

    type: AuthType
    enabled: bool
    api_keys: NotRequired[List[str]]
    oauth_config: NotRequired[Dict[str, Any]]
    jwt_config: NotRequired[Dict[str, Any]]


# Logging types
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["json", "text"]


class LogConfig(TypedDict):
    """Logging configuration.

    Attributes:
        level: Log level
        format: Log format
        file: Optional log file path
        enable_console: Whether to log to console
    """

    level: LogLevel
    format: LogFormat
    file: NotRequired[str]
    enable_console: NotRequired[bool]


# Feature flags
class FeatureFlags(TypedDict):
    """Feature flags configuration.

    Attributes:
        enable_progress: Enable progress reporting
        enable_binary_content: Enable binary content support
        max_request_size: Maximum request size in bytes
    """

    enable_progress: bool
    enable_binary_content: bool
    max_request_size: int


# Complete server configuration
class ServerConfig(TypedDict):
    """Complete server configuration.

    Attributes:
        metadata: Server metadata
        transport: Transport configuration
        rate_limit: Rate limiting configuration
        auth: Authentication configuration
        logging: Logging configuration
        features: Feature flags
    """

    metadata: ServerMetadata
    transport: TransportConfig
    rate_limit: NotRequired[RateLimitConfig]
    auth: NotRequired[AuthConfig]
    logging: NotRequired[LogConfig]
    features: NotRequired[FeatureFlags]


__all__ = [
    # Type aliases
    "JSONValue",
    "JSONDict",
    "HandlerFunction",
    # Core configs
    "ToolConfig",
    "PromptConfig",
    "ResourceConfig",
    "ServerMetadata",
    # Transport
    "TransportType",
    "TransportConfig",
    # Progress
    "ProgressUpdate",
    "ProgressReporter",
    # Context
    "RequestContext",
    "HandlerContext",
    # API Style
    "APIStyle",
    "APIStyleInfo",
    # Validation
    "ValidationError",
    "ValidationResult",
    # Security
    "RateLimitConfig",
    "AuthType",
    "AuthConfig",
    # Logging
    "LogLevel",
    "LogFormat",
    "LogConfig",
    # Features
    "FeatureFlags",
    # Server
    "ServerConfig",
]
