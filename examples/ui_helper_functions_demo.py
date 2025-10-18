"""MCP-UI Helper Functions Demo.

This example demonstrates using the MCP-UI helper functions:
- create_inline_html_resource()
- create_external_url_resource()
- create_remote_dom_resource() (Layer 3)

This shows the alternative approach to using add_ui_resource().
"""

import asyncio

from simply_mcp import (
    BuildMCPServer,
    UIResourceMetadata,
    UIResourceOptions,
    PreferredFrameSize,
    create_inline_html_resource,
    create_external_url_resource,
)

# Create server instance
server = BuildMCPServer(
    name="ui-helper-demo",
    version="1.0.0",
    description="MCP-UI helper functions demonstration",
)


# Example 1: Using create_inline_html_resource with options
@server.tool()
def show_status_card() -> dict:
    """Show a status card using create_inline_html_resource with metadata."""

    # Create UI resource with preferred frame size
    ui_resource = create_inline_html_resource(
        uri="ui://status/card",
        html_content="""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <style>
            body {
              font-family: system-ui, sans-serif;
              margin: 0;
              padding: 16px;
              background: #f0f0f0;
            }
            .status-card {
              background: white;
              border-radius: 8px;
              padding: 24px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .status {
              display: inline-block;
              padding: 8px 16px;
              border-radius: 4px;
              font-weight: 600;
              background: #10b981;
              color: white;
            }
            h2 {
              margin: 0 0 16px 0;
            }
          </style>
        </head>
        <body>
          <div class="status-card">
            <h2>Server Status</h2>
            <div class="status">✓ OPERATIONAL</div>
            <p>All systems running normally</p>
          </div>
        </body>
        </html>
        """,
        options=UIResourceOptions(
            metadata=UIResourceMetadata(
                preferred_frame_size=PreferredFrameSize(width=400, height=300)
            ),
            annotations={"app.example/card-type": "status"},
        ),
    )

    return {
        "message": "Status card created",
        "ui_resource": ui_resource.model_dump(by_alias=True),
    }


# Example 2: External URL resource (for Layer 2)
@server.tool()
def show_external_dashboard() -> dict:
    """Show an external dashboard using create_external_url_resource."""

    ui_resource = create_external_url_resource(
        uri="ui://dashboard/external",
        url="https://example.com/dashboard",
        options=UIResourceOptions(
            metadata=UIResourceMetadata(
                preferred_frame_size=PreferredFrameSize(width=1200, height=800)
            )
        ),
    )

    return {
        "message": "External dashboard UI resource created",
        "ui_resource": ui_resource.model_dump(by_alias=True),
        "note": "This requires Feature Layer (Layer 2) support in the client",
    }


# Example 3: Dynamic HTML generation
@server.tool()
def create_user_card(name: str, role: str = "User") -> dict:
    """Create a personalized user card with dynamic content.

    Args:
        name: User's name
        role: User's role (default: "User")
    """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body {{
          font-family: system-ui, sans-serif;
          margin: 0;
          padding: 20px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
        }}
        .user-card {{
          background: white;
          border-radius: 16px;
          padding: 32px;
          text-align: center;
          box-shadow: 0 8px 32px rgba(0,0,0,0.2);
          max-width: 300px;
        }}
        .avatar {{
          width: 80px;
          height: 80px;
          border-radius: 50%;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          margin: 0 auto 16px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 36px;
          color: white;
          font-weight: bold;
        }}
        h2 {{
          margin: 0 0 8px 0;
          color: #333;
        }}
        .role {{
          color: #666;
          font-size: 14px;
          text-transform: uppercase;
          letter-spacing: 1px;
        }}
      </style>
    </head>
    <body>
      <div class="user-card">
        <div class="avatar">{name[0].upper()}</div>
        <h2>{name}</h2>
        <div class="role">{role}</div>
      </div>
    </body>
    </html>
    """

    ui_resource = create_inline_html_resource(
        uri=f"ui://user/card/{name.lower().replace(' ', '-')}",
        html_content=html_content,
        options=UIResourceOptions(
            metadata=UIResourceMetadata(
                preferred_frame_size=PreferredFrameSize(width=400, height=400),
                initial_render_data={"name": name, "role": role},
            )
        ),
    )

    return {
        "message": f"User card created for {name}",
        "ui_resource": ui_resource.model_dump(by_alias=True),
    }


# Example 4: Validate UI resources
@server.tool()
def validate_ui_resource_examples() -> dict:
    """Demonstrate is_ui_resource validation."""
    from simply_mcp import is_ui_resource

    # Create test resources
    valid_resource = create_inline_html_resource(
        uri="ui://test/valid",
        html_content="<div>Test</div>",
    )

    # Test validation
    results = {
        "valid_html_resource": is_ui_resource(valid_resource),
        "invalid_none": is_ui_resource(None),
        "invalid_dict": is_ui_resource({"type": "not-a-resource"}),
        "test_passed": is_ui_resource(valid_resource) is True,
    }

    return {
        "message": "UI resource validation tests completed",
        "results": results,
    }


async def main():
    """Run the MCP server."""
    print("Starting MCP-UI Helper Functions Demo...")
    print("\nAvailable Tools:")
    for tool in server.list_tools():
        print(f"  - {tool}")

    print("\nInitializing server...")
    await server.initialize()

    print("Server ready. Running with stdio transport...")
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
