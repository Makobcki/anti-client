from __future__ import annotations

from typing import Any, List

from .types import Tool


async def load_mcp_tools(session: Any) -> List[Tool]:
    """Fetches all tools from an MCP (Model Context Protocol) session and converts them to anti_client Tools.

    Args:
        session (Any): An initialized MCP ClientSession instance.

    Returns:
        List[Tool]: A list of anti_client Tool objects corresponding to the MCP session tools.
    """
    result = await session.list_tools()
    mcp_tools = getattr(result, "tools", result)
    tools = []
    for tool in mcp_tools:
        tools.append(Tool.from_mcp(tool, session))
    return tools


class MCPAdapter:
    """Adapter for integrating Model Context Protocol (MCP) tools and sessions with anti_client."""

    def __init__(self, session: Any):
        """Initializes the MCPAdapter.

        Args:
            session (Any): An active MCP ClientSession instance.
        """
        self.session = session

    async def get_tools(self) -> List[Tool]:
        """Retrieves all tools from the MCP server session as anti_client Tools.

        Returns:
            List[Tool]: The converted list of tools.
        """
        return await load_mcp_tools(self.session)
