"""MCP-UI Foundation Layer Demo - Inline HTML UI Resources.

This example demonstrates the Foundation Layer of MCP-UI support:
- Static HTML UI resources
- Sandboxed iframe rendering
- Basic UI resource creation

This mirrors the TypeScript ui-inline-html-demo.ts example.
"""

import asyncio

from simply_mcp import BuildMCPServer

# Create server instance
server = BuildMCPServer(
    name="ui-inline-html-demo",
    version="1.0.0",
    description="Foundation Layer: Inline HTML UI resources",
)


# Example 1: Product Card UI Resource
# This demonstrates a static HTML card with styling
server.add_ui_resource(
    uri="ui://product-card/demo",
    name="Product Card",
    description="Displays a product information card",
    mime_type="text/html",
    content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          margin: 0;
          padding: 16px;
          background: #f5f5f5;
        }
        .card {
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          padding: 24px;
          max-width: 400px;
        }
        .card h2 {
          margin: 0 0 8px 0;
          font-size: 20px;
          font-weight: 600;
        }
        .card p {
          margin: 0 0 16px 0;
          color: #666;
          line-height: 1.5;
        }
        .badge {
          display: inline-block;
          background: #007bff;
          color: white;
          padding: 4px 12px;
          border-radius: 4px;
          font-size: 12px;
          font-weight: 600;
          margin-bottom: 16px;
        }
        .info-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
          margin: 16px 0;
        }
        .info-item {
          border-left: 2px solid #007bff;
          padding-left: 12px;
        }
        .info-label {
          font-size: 12px;
          color: #999;
          text-transform: uppercase;
          margin-bottom: 4px;
        }
        .info-value {
          font-size: 16px;
          font-weight: 600;
          color: #333;
        }
      </style>
    </head>
    <body>
      <div class="card">
        <div class="badge">Foundation Layer Demo</div>
        <h2>Widget Pro X</h2>
        <p>A demonstration of static HTML UI resources in MCP-UI Foundation Layer.</p>

        <div class="info-grid">
          <div class="info-item">
            <div class="info-label">Price</div>
            <div class="info-value">$299</div>
          </div>
          <div class="info-item">
            <div class="info-label">In Stock</div>
            <div class="info-value">✓ Yes</div>
          </div>
          <div class="info-item">
            <div class="info-label">Rating</div>
            <div class="info-value">4.8★</div>
          </div>
          <div class="info-item">
            <div class="info-label">Reviews</div>
            <div class="info-value">1.2K</div>
          </div>
        </div>

        <p style="font-size: 14px; color: #999;">
          This is a static HTML demo. Interactivity will be added in the Feature Layer
          with postMessage callbacks. Remote DOM support comes in Layer 3.
        </p>
      </div>
    </body>
    </html>
  """,
)


# Example 2: Simple configuration panel
server.add_ui_resource(
    uri="ui://config/panel",
    name="Configuration Panel",
    description="A simple configuration display panel",
    mime_type="text/html",
    content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body {
          font-family: system-ui, -apple-system, sans-serif;
          margin: 0;
          padding: 20px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          min-height: 100vh;
        }
        .panel {
          background: white;
          border-radius: 12px;
          padding: 32px;
          max-width: 600px;
          margin: 0 auto;
          box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        h1 {
          margin: 0 0 24px 0;
          color: #333;
          font-size: 28px;
        }
        .setting {
          padding: 16px;
          border-bottom: 1px solid #eee;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .setting:last-child {
          border-bottom: none;
        }
        .setting-label {
          font-weight: 500;
          color: #555;
        }
        .setting-value {
          color: #007bff;
          font-weight: 600;
        }
      </style>
    </head>
    <body>
      <div class="panel">
        <h1>Server Configuration</h1>
        <div class="setting">
          <div class="setting-label">Server Name</div>
          <div class="setting-value">ui-inline-html-demo</div>
        </div>
        <div class="setting">
          <div class="setting-label">Version</div>
          <div class="setting-value">1.0.0</div>
        </div>
        <div class="setting">
          <div class="setting-label">MCP-UI Layer</div>
          <div class="setting-value">Foundation (Layer 1)</div>
        </div>
        <div class="setting">
          <div class="setting-label">Features</div>
          <div class="setting-value">Static HTML</div>
        </div>
      </div>
    </body>
    </html>
  """,
)


# Example 3: Dynamic content with a handler function
def get_welcome_html() -> str:
    """Generate dynamic HTML content.

    This demonstrates that UI resource content can be generated
    dynamically using handler functions.
    """
    import datetime

    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%B %d, %Y")

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body {{
          font-family: 'Courier New', monospace;
          margin: 0;
          padding: 32px;
          background: #1a1a1a;
          color: #0f0;
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        .terminal {{
          border: 2px solid #0f0;
          padding: 24px;
          border-radius: 4px;
          background: #000;
          box-shadow: 0 0 20px rgba(0,255,0,0.3);
          max-width: 600px;
        }}
        .prompt {{
          color: #0f0;
          margin-bottom: 16px;
        }}
        .output {{
          color: #0f0;
          line-height: 1.6;
        }}
        .blink {{
          animation: blink 1s infinite;
        }}
        @keyframes blink {{
          0%, 49% {{ opacity: 1; }}
          50%, 100% {{ opacity: 0; }}
        }}
      </style>
    </head>
    <body>
      <div class="terminal">
        <div class="prompt">$ mcp-ui --status</div>
        <div class="output">
          MCP-UI Foundation Layer v1.0.0<br>
          Status: <span style="color: #0f0;">ONLINE</span><br>
          <br>
          Server: ui-inline-html-demo<br>
          Time: {time_str}<br>
          Date: {date_str}<br>
          <br>
          Available Resources: 3<br>
          - ui://product-card/demo<br>
          - ui://config/panel<br>
          - ui://welcome/terminal<br>
          <br>
          $ <span class="blink">_</span>
        </div>
      </div>
    </body>
    </html>
    """


server.add_ui_resource(
    uri="ui://welcome/terminal",
    name="Welcome Terminal",
    description="Dynamic terminal-style welcome UI",
    mime_type="text/html",
    content=get_welcome_html,
)


# Add a tool to get demo information
@server.tool()
def get_demo_info() -> dict:
    """Returns information about the Foundation Layer demo."""
    return {
        "layer": "Foundation",
        "features": [
            "Static HTML UI resources",
            "Sandboxed iframe rendering",
            "Security: sandbox attribute",
            "Dynamic content generation",
        ],
        "next_layer": "Interactive callbacks (Feature Layer)",
        "resources": [
            {"uri": "ui://product-card/demo", "name": "Product Card"},
            {"uri": "ui://config/panel", "name": "Configuration Panel"},
            {"uri": "ui://welcome/terminal", "name": "Welcome Terminal"},
        ],
    }


# Add a tool to list available UI resources
@server.tool()
def list_ui_resources() -> dict:
    """List all available UI resources in this demo."""
    resources = server.list_resources()
    ui_resources = [uri for uri in resources if uri.startswith("ui://")]

    return {
        "count": len(ui_resources),
        "ui_resources": ui_resources,
        "total_resources": len(resources),
    }


async def main():
    """Run the MCP server."""
    print("Starting MCP-UI Foundation Layer Demo...")
    print("\nAvailable UI Resources:")
    for uri in server.list_resources():
        if uri.startswith("ui://"):
            print(f"  - {uri}")

    print("\nInitializing server...")
    await server.initialize()

    print("Server ready. Running with stdio transport...")
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
