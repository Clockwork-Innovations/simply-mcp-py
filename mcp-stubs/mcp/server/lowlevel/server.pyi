"""Type stubs for mcp.server.lowlevel.server module.

This stub file provides proper type annotations for the MCP Server class decorators,
which lack complete type hints in the MCP SDK. These stubs enable strict type checking
without modifying the third-party library.
"""

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Generic, TypeVar

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.server.models import InitializationOptions
from mcp.shared.message import SessionMessage
from mcp.types import Icon

# Type variables for Server generic parameters
LifespanResultT = TypeVar("LifespanResultT")
RequestT = TypeVar("RequestT")

# Type variable for decorator functions
F = TypeVar("F", bound=Callable[..., Awaitable[Any]])

class NotificationOptions:
    """Notification options for server initialization."""
    ...

class Server(Generic[LifespanResultT, RequestT]):
    """Type stub for MCP Server with properly typed decorators.

    This class provides type hints for the decorator methods used to register
    MCP handlers. The actual implementation is in the mcp package; this stub
    only provides type information for static analysis.
    """

    def __init__(
        self,
        name: str,
        version: str | None = None,
        instructions: str | None = None,
        website_url: str | None = None,
        icons: list[Icon] | None = None,
        lifespan: Callable[
            [Server[LifespanResultT, RequestT]],
            AbstractAsyncContextManager[LifespanResultT],
        ] = ...,
    ) -> None:
        """Initialize MCP Server.

        Args:
            name: Server name
            version: Optional server version
            instructions: Optional server instructions
            website_url: Optional website URL
            icons: Optional list of icons
            lifespan: Lifespan context manager
        """
        ...

    def list_tools(self) -> Callable[[F], F]:
        """Register a list_tools handler.

        Decorator for registering a function that lists available tools.

        Returns:
            A decorator that preserves the function's type signature.
        """
        ...

    def call_tool(self) -> Callable[[F], F]:
        """Register a call_tool handler.

        Decorator for registering a function that executes tools.

        Returns:
            A decorator that preserves the function's type signature.
        """
        ...

    def list_prompts(self) -> Callable[[F], F]:
        """Register a list_prompts handler.

        Decorator for registering a function that lists available prompts.

        Returns:
            A decorator that preserves the function's type signature.
        """
        ...

    def get_prompt(self) -> Callable[[F], F]:
        """Register a get_prompt handler.

        Decorator for registering a function that retrieves prompt content.

        Returns:
            A decorator that preserves the function's type signature.
        """
        ...

    def list_resources(self) -> Callable[[F], F]:
        """Register a list_resources handler.

        Decorator for registering a function that lists available resources.

        Returns:
            A decorator that preserves the function's type signature.
        """
        ...

    def read_resource(self) -> Callable[[F], F]:
        """Register a read_resource handler.

        Decorator for registering a function that reads resource content.

        Returns:
            A decorator that preserves the function's type signature.
        """
        ...

    def create_initialization_options(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, Any] | None = None,
    ) -> InitializationOptions:
        """Create initialization options for the server.

        Args:
            notification_options: Optional notification options
            experimental_capabilities: Optional experimental capabilities

        Returns:
            Initialization options
        """
        ...

    async def run(
        self,
        read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
        write_stream: MemoryObjectSendStream[SessionMessage],
        init_options: InitializationOptions,
        raise_exceptions: bool = False,
    ) -> None:
        """Run the MCP server with given streams.

        Args:
            read_stream: Input stream for receiving messages
            write_stream: Output stream for sending messages
            init_options: Initialization options
            raise_exceptions: Whether to raise exceptions
        """
        ...
