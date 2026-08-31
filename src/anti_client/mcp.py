"""Model Context Protocol (MCP) adapter and session integration."""

from __future__ import annotations

from typing import Any, Protocol

from .types import Tool


class MCPToolProtocol(Protocol):
    """Protocol for an MCP Tool definition."""

    name: str
    description: str | None
    inputSchema: dict[str, Any] | None


class MCPSessionProtocol(Protocol):
    """Protocol representing an active Model Context Protocol (MCP) ClientSession."""

    async def list_tools(self) -> Any:
        """Lists tools exposed by the MCP server."""
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Executes a specific tool on the MCP server."""
        ...


async def load_mcp_tools(session: MCPSessionProtocol) -> list[Tool]:
    """Fetches all tools from an MCP (Model Context Protocol) session and converts them to anti_client Tools.

    Args:
        session (MCPSessionProtocol): An initialized MCP ClientSession instance.

    Returns:
        List[Tool]: A list of anti_client Tool objects corresponding to the MCP session tools.
    """
    result = await session.list_tools()
    mcp_tools = getattr(result, "tools", result)
    tools: list[Tool] = []
    if isinstance(mcp_tools, list):
        for tool in mcp_tools:
            tools.append(Tool.from_mcp(tool, session))
    return tools


class MCPAdapter:
    """Adapter for integrating Model Context Protocol (MCP) tools and sessions with anti_client."""

    def __init__(self, session: MCPSessionProtocol):
        """Initializes the MCPAdapter.

        Args:
            session (MCPSessionProtocol): An active MCP ClientSession instance.
        """
        self.session = session

    async def get_tools(self) -> list[Tool]:
        """Retrieves all tools from the MCP server session as anti_client Tools.

        Returns:
            List[Tool]: The converted list of tools.
        """
        return await load_mcp_tools(self.session)
