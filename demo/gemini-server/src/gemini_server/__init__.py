"""Gemini MCP Server - Google Gemini API integration for Model Context Protocol.

This package provides an MCP server that integrates with Google's Gemini API,
enabling AI assistants to:
- Upload and manage files
- Generate content with optional file context
- Start and manage chat sessions
- Analyze media and documents
"""

__version__ = "0.1.0"
__author__ = "Clockwork Innovations"
__license__ = "MIT"

from gemini_server.server import create_gemini_server

__all__ = ["create_gemini_server"]
