"""UI Resource Types for MCP-UI Foundation Layer.

This module defines the structure for UI resources that can be rendered
in MCP clients. UI resources support multiple content types:
- rawHtml: Inline HTML content rendered in sandboxed iframes
- externalUrl: External URLs loaded in iframes
- remoteDom: Remote DOM rendering (Layer 3)

These types mirror the TypeScript implementation in simple-mcp.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# UI content type - determines how content is rendered
UIContentType = Literal["rawHtml", "externalUrl", "remoteDom"]


class UIResourcePayload(BaseModel):
    """UI Resource Payload.

    The actual resource data returned by the MCP server.
    Contains the URI, MIME type, and content (text or blob).

    Attributes:
        uri: Unique identifier for the UI resource (must start with "ui://")
        mime_type: MIME type indicating content type:
            - text/html: Inline HTML content
            - text/uri-list: External URL
            - application/vnd.mcp-ui.remote-dom+javascript: Remote DOM
        text: Text content (HTML or URL)
        blob: Base64-encoded blob content (optional)
        meta: Metadata annotations (optional)
            Convention: Use namespaced keys like "mcpui.dev/ui-preferred-frame-size"
    """

    uri: str = Field(..., min_length=1, description="Resource URI (must start with ui://)")
    mime_type: str = Field(..., description="MIME type")
    text: str | None = Field(None, description="Text content")
    blob: str | None = Field(None, description="Base64-encoded blob")
    meta: dict[str, Any] | None = Field(
        None,
        alias="_meta",
        description="Metadata annotations",
    )

    model_config = ConfigDict(
        populate_by_name=True,  # Allow both meta and _meta
        extra="forbid",
    )


class UIResource(BaseModel):
    """UI Resource.

    Complete UI resource object with MCP resource envelope.
    This is the format returned by resources/read requests.

    Attributes:
        type: Resource type (always "resource")
        resource: The actual resource payload
    """

    type: Literal["resource"] = Field("resource", description="Resource type")
    resource: UIResourcePayload = Field(..., description="Resource payload")

    model_config = ConfigDict(extra="forbid")


class PreferredFrameSize(BaseModel):
    """Preferred frame size hint for UI rendering.

    Attributes:
        width: Preferred width in pixels (optional)
        height: Preferred height in pixels (optional)
    """

    width: int | None = Field(None, ge=0, description="Preferred width in pixels")
    height: int | None = Field(None, ge=0, description="Preferred height in pixels")

    model_config = ConfigDict(extra="forbid")


class UIResourceMetadata(BaseModel):
    """Metadata for UI rendering.

    Attributes:
        preferred_frame_size: Preferred iframe size (rendering hint)
            Client may ignore or adjust based on available space
        initial_render_data: Initial data passed to UI on first render
            Available in Layer 2+ for interactive UIs
    """

    preferred_frame_size: PreferredFrameSize | None = Field(
        None,
        description="Preferred iframe size",
    )
    initial_render_data: dict[str, Any] | None = Field(
        None,
        description="Initial data for UI",
    )

    model_config = ConfigDict(extra="forbid")


class UIResourceOptions(BaseModel):
    """UI Resource Options.

    Optional configuration when creating UI resources.
    Includes metadata for rendering hints and custom annotations.

    Attributes:
        metadata: Metadata for UI rendering
        annotations: Custom annotations added to _meta
            Use namespaced keys to avoid conflicts
    """

    metadata: UIResourceMetadata | None = Field(None, description="UI metadata")
    annotations: dict[str, Any] | None = Field(None, description="Custom annotations")

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "UIContentType",
    "UIResourcePayload",
    "UIResource",
    "PreferredFrameSize",
    "UIResourceMetadata",
    "UIResourceOptions",
]
