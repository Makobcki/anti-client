import pytest

from anti_client.mcp import MCPAdapter
from anti_client.types import Tool


class MockMCPTool:
    def __init__(self, name: str, description: str, schema: dict):
        self.name = name
        self.description = description
        self.inputSchema = schema


class MockMCPSession:
    async def list_tools(self):
        class Result:
            tools = [
                MockMCPTool(
                    name="calculate_sum",
                    description="Calculates sum of two numbers",
                    schema={
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    },
                )
            ]

        return Result()

    async def call_tool(self, name: str, arguments: dict):
        class ContentItem:
            text = f"Result of {name}: {arguments.get('a', 0) + arguments.get('b', 0)}"

        class Result:
            content = [ContentItem()]

        return Result()


@pytest.mark.asyncio
async def test_mcp_adapter():
    session = MockMCPSession()
    adapter = MCPAdapter(session)
    tools = await adapter.get_tools()

    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, Tool)
    assert tool.name == "calculate_sum"
    assert tool.description == "Calculates sum of two numbers"

    # Test executing tool
    exec_result = await tool.func(a=5, b=7)
    assert "Result of calculate_sum: 12" in exec_result
