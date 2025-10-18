"""UI Resource Helpers for MCP-UI Foundation Layer.

Helper functions for creating and validating UI resources.
UI resources are MCP resources with special MIME types that
indicate they should be rendered as interactive UI elements.

These helpers mirror the TypeScript implementation in simple-mcp.
"""

from typing import Any
from urllib.parse import urlparse

from simply_mcp.core.ui_types import UIResource, UIResourceOptions, UIResourcePayload


def create_inline_html_resource(
    uri: str,
    html_content: str,
    options: UIResourceOptions | None = None,
) -> UIResource:
    """Create an inline HTML UI resource.

    Creates a UIResource object with text/html MIME type for rendering
    HTML content in a sandboxed iframe. The HTML should be complete and
    self-contained (no external dependencies unless handled by Layer 2+).

    Args:
        uri: Unique identifier (must start with "ui://")
        html_content: HTML string to render
        options: Optional metadata and annotations

    Returns:
        UIResource object ready for MCP response

    Raises:
        ValueError: If URI doesn't start with "ui://"

    Example:
        Basic HTML card:
        >>> ui_resource = create_inline_html_resource(
        ...     'ui://product-selector/v1',
        ...     '<div><h2>Select a product</h2><button>Widget A</button></div>'
        ... )

        With metadata:
        >>> from simply_mcp.core.ui_types import UIResourceOptions, UIResourceMetadata, PreferredFrameSize
        >>> ui_resource = create_inline_html_resource(
        ...     'ui://chart/quarterly-sales',
        ...     '<div id="chart">...</div>',
        ...     UIResourceOptions(
        ...         metadata=UIResourceMetadata(
        ...             preferred_frame_size=PreferredFrameSize(width=800, height=600),
        ...             initial_render_data={'quarter': 'Q4', 'year': 2024}
        ...         ),
        ...         annotations={'myapp.com/chart-type': 'line-chart'}
        ...     )
        ... )
    """
    # Validate URI starts with ui://
    if not uri.startswith("ui://"):
        raise ValueError(
            f'Invalid UI resource URI: "{uri}". UI resource URIs must start with "ui://"'
        )

    # Build metadata if provided
    metadata = _build_metadata(options)

    # Create payload
    payload_dict: dict[str, Any] = {
        "uri": uri,
        "mime_type": "text/html",
        "text": html_content,
    }

    if metadata:
        payload_dict["_meta"] = metadata

    return UIResource(
        type="resource",
        resource=UIResourcePayload(**payload_dict),
    )


def create_external_url_resource(
    uri: str,
    url: str,
    options: UIResourceOptions | None = None,
) -> UIResource:
    """Create an external URL UI resource (Layer 2).

    Creates a UIResource object that points to an external URL to be
    embedded in an iframe. The URL must be HTTPS (or localhost for dev).
    External URLs are rendered with more permissive sandbox settings
    to allow same-origin API calls.

    Args:
        uri: Unique identifier (must start with "ui://")
        url: HTTPS URL to embed in iframe
        options: Optional metadata and annotations

    Returns:
        UIResource object ready for MCP response

    Raises:
        ValueError: If URI doesn't start with "ui://"
        ValueError: If URL is not HTTPS or localhost

    Example:
        External dashboard:
        >>> ui_resource = create_external_url_resource(
        ...     'ui://analytics/dashboard',
        ...     'https://example.com/dashboard'
        ... )

        Localhost for development:
        >>> ui_resource = create_external_url_resource(
        ...     'ui://dev/widget',
        ...     'http://localhost:3000/widget'
        ... )
    """
    # Validate URI starts with ui://
    if not uri.startswith("ui://"):
        raise ValueError(
            f'Invalid UI resource URI: "{uri}". UI resource URIs must start with "ui://"'
        )

    # Validate URL is HTTPS or localhost
    try:
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            and parsed.hostname not in ("localhost", "127.0.0.1")
        ):
            raise ValueError(
                f'Invalid external URL: "{url}". Must be HTTPS or localhost (for development)'
            )
    except Exception as e:
        raise ValueError(f'Invalid URL: "{url}". Error: {e}') from e

    # Build metadata if provided
    metadata = _build_metadata(options)

    # Create payload
    payload_dict: dict[str, Any] = {
        "uri": uri,
        "mime_type": "text/uri-list",
        "text": url,
    }

    if metadata:
        payload_dict["_meta"] = metadata

    return UIResource(
        type="resource",
        resource=UIResourcePayload(**payload_dict),
    )


def create_remote_dom_resource(
    uri: str,
    script: str,
    framework: str = "javascript",
    options: UIResourceOptions | None = None,
) -> UIResource:
    """Create a Remote DOM UI resource (Layer 3).

    Creates a UIResource object containing JavaScript that will be executed
    in a Web Worker sandbox. The script uses the remoteDOM API to create
    native-looking React components safely and securely.

    Security: The script runs in a Web Worker with no DOM access. It can only
    communicate via a controlled postMessage protocol with whitelisted operations.

    Args:
        uri: Unique identifier (must start with "ui://")
        script: JavaScript code to execute in Web Worker sandbox
        framework: Framework type (default: 'javascript', also: 'react', 'web-components')
        options: Optional metadata and annotations

    Returns:
        UIResource object ready for MCP response

    Raises:
        ValueError: If URI doesn't start with "ui://"

    Example:
        Simple counter component:
        >>> script = '''
        ... const card = remoteDOM.createElement('div', { style: { padding: '20px' } });
        ... const title = remoteDOM.createElement('h2');
        ... remoteDOM.setTextContent(title, 'Counter');
        ... remoteDOM.appendChild(card, title);
        ...
        ... const display = remoteDOM.createElement('div', { id: 'count' });
        ... remoteDOM.setTextContent(display, '0');
        ... remoteDOM.appendChild(card, display);
        ...
        ... const button = remoteDOM.createElement('button');
        ... remoteDOM.setTextContent(button, 'Increment');
        ... remoteDOM.addEventListener(button, 'click', () => {
        ...     remoteDOM.callHost('notify', { level: 'info', message: 'Clicked!' });
        ... });
        ... remoteDOM.appendChild(card, button);
        ... '''
        >>> resource = create_remote_dom_resource(
        ...     'ui://counter/v1',
        ...     script,
        ...     'javascript'
        ... )

        With metadata for sizing hints:
        >>> from simply_mcp.core.ui_types import UIResourceOptions, UIResourceMetadata, PreferredFrameSize
        >>> resource = create_remote_dom_resource(
        ...     'ui://dashboard/widget',
        ...     dashboard_script,
        ...     'javascript',
        ...     UIResourceOptions(
        ...         metadata=UIResourceMetadata(
        ...             preferred_frame_size=PreferredFrameSize(width=800, height=600)
        ...         )
        ...     )
        ... )
    """
    # Validate URI starts with ui://
    if not uri.startswith("ui://"):
        raise ValueError(
            f'Invalid UI resource URI: "{uri}". UI resource URIs must start with "ui://"'
        )

    # Build metadata if provided
    metadata = _build_metadata(options)

    # Note: We do NOT validate script content here
    # Script is executed in Web Worker sandbox which is the security boundary
    # Invalid scripts will fail safely in the worker without crashing the host

    # Create payload
    payload_dict: dict[str, Any] = {
        "uri": uri,
        "mime_type": f"application/vnd.mcp-ui.remote-dom+{framework}",
        "text": script,
    }

    if metadata:
        payload_dict["_meta"] = metadata

    return UIResource(
        type="resource",
        resource=UIResourcePayload(**payload_dict),
    )


def is_ui_resource(resource: Any) -> bool:
    """Type guard to check if a resource is a UI resource.

    Validates that a resource object has the correct structure and
    MIME type to be considered a UI resource. Checks for:
    - Correct object structure (type: 'resource')
    - Valid UI MIME types (text/html, text/uri-list, or Remote DOM)

    Args:
        resource: Object to check

    Returns:
        True if resource is a valid UI resource

    Example:
        >>> resource = create_inline_html_resource('ui://test', '<div>Hello</div>')
        >>> if is_ui_resource(resource):
        ...     print('Valid UI resource:', resource.resource.uri)
        Valid UI resource: ui://test
    """
    # Explicitly check for None
    if resource is None:
        return False

    # Check if it's a UIResource instance (duck typing)
    if not hasattr(resource, "type") or not hasattr(resource, "resource"):
        return False

    # Check type field
    if resource.type != "resource":
        return False

    # Check resource field exists and has mime_type
    if not hasattr(resource.resource, "mime_type"):
        return False

    # Check MIME type
    mime_type = resource.resource.mime_type
    if not isinstance(mime_type, str):
        return False

    # Check if it's a UI MIME type
    return (
        mime_type == "text/html"
        or mime_type == "text/uri-list"
        or mime_type.startswith("application/vnd.mcp-ui.remote-dom")
    )


def _build_metadata(options: UIResourceOptions | None) -> dict[str, Any]:
    """Helper function to build metadata object.

    Merges user-provided metadata with MCP-UI conventions.
    Converts structured metadata options into the flat _meta object
    with namespaced keys following MCP-UI conventions.

    Args:
        options: UI resource options containing metadata

    Returns:
        Flat metadata object with namespaced keys

    Example:
        >>> from simply_mcp.core.ui_types import UIResourceOptions, UIResourceMetadata, PreferredFrameSize
        >>> options = UIResourceOptions(
        ...     metadata=UIResourceMetadata(
        ...         preferred_frame_size=PreferredFrameSize(width=800, height=600)
        ...     ),
        ...     annotations={'myapp.com/custom': 'value'}
        ... )
        >>> metadata = _build_metadata(options)
        >>> print(metadata)
        {'mcpui.dev/ui-preferred-frame-size': {'width': 800, 'height': 600}, 'myapp.com/custom': 'value'}
    """
    metadata: dict[str, Any] = {}

    if options is None:
        return metadata

    # Add preferred frame size if present
    if (
        options.metadata
        and options.metadata.preferred_frame_size
    ):
        # Convert Pydantic model to dict, excluding None values
        frame_size_dict = options.metadata.preferred_frame_size.model_dump(
            exclude_none=True
        )
        if frame_size_dict:  # Only add if not empty
            metadata["mcpui.dev/ui-preferred-frame-size"] = frame_size_dict

    # Add initial render data if present
    if (
        options.metadata
        and options.metadata.initial_render_data
    ):
        metadata["mcpui.dev/ui-initial-render-data"] = (
            options.metadata.initial_render_data
        )

    # Add custom annotations
    if options.annotations:
        metadata.update(options.annotations)

    return metadata


__all__ = [
    "create_inline_html_resource",
    "create_external_url_resource",
    "create_remote_dom_resource",
    "is_ui_resource",
]
