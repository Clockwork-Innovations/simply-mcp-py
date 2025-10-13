#!/usr/bin/env python3
"""Basic Builder API Example.

This example demonstrates the basic usage of the SimplyMCP builder API
for creating an MCP server with tools, prompts, and resources.
"""

import asyncio
from simply_mcp import SimplyMCP

# Create a server using the builder API
mcp = SimplyMCP(name="calculator", version="1.0.0", description="A simple calculator server")


# Register tools using decorators
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


# Or register tools directly
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Result of division
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


mcp.add_tool("divide", divide, description="Divide two numbers")


# Add prompts
@mcp.prompt()
def math_tutor(topic: str = "algebra") -> str:
    """Generate a math tutoring prompt.

    Args:
        topic: The math topic to focus on

    Returns:
        A tutoring prompt
    """
    return f"""You are a helpful math tutor specializing in {topic}.
Help the student understand concepts clearly with examples.
Be patient and encouraging."""


# Add resources
@mcp.resource(uri="config://calculator")
def get_config() -> dict:
    """Get calculator configuration.

    Returns:
        Configuration dictionary
    """
    return {
        "name": "calculator",
        "version": "1.0.0",
        "supported_operations": ["add", "multiply", "divide"],
        "precision": 10
    }


async def main():
    """Run the calculator server."""
    print("Starting Calculator MCP Server...")
    print(f"Registered tools: {mcp.list_tools()}")
    print(f"Registered prompts: {mcp.list_prompts()}")
    print(f"Registered resources: {mcp.list_resources()}")

    # Initialize and run
    await mcp.initialize()
    await mcp.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
