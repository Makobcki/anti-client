import os
import tempfile

import pytest

from anti_client.mcp import MCPAdapter
from anti_client.types import CountTokensResult, FileAttachment


def test_file_attachment_adversarial():
    # 1. Non-existent file on from_file
    with pytest.raises(FileNotFoundError):
        FileAttachment.from_file("/path/to/nonexistent/file/12345.bin")

    import binascii

    # 2. Corrupted base64 in to_bytes
    corrupted = FileAttachment(mime_type="image/png", data="!!!not-valid-base64-string!!!")
    with pytest.raises((binascii.Error, ValueError)):
        corrupted.to_bytes()

    # 3. Save auto-creates parent directory
    with tempfile.TemporaryDirectory() as tmpdir:
        deep_target = os.path.join(tmpdir, "a", "b", "c", "out.bin")
        valid = FileAttachment.from_bytes(b"payload", "application/octet-stream")
        valid.save(deep_target)
        assert os.path.exists(deep_target)
        with open(deep_target, "rb") as f:
            assert f.read() == b"payload"


def test_count_tokens_result_type_errors():
    res = CountTokensResult(total_tokens=10)
    # Comparison with non-numeric should raise TypeError on ordering or False on equality
    assert res != "10"
    assert res is not None
    with pytest.raises(TypeError):
        _ = res < "5"
    with pytest.raises(TypeError):
        _ = res > [1, 2]
    with pytest.raises(TypeError):
        _ = res + "5"


@pytest.mark.asyncio
async def test_mcp_adapter_adversarial():
    # MCP session that returns None or empty tool lists
    class EmptyMCPSession:
        async def list_tools(self):
            class Res:
                tools = []

            return Res()

    adapter = MCPAdapter(EmptyMCPSession())
    tools = await adapter.get_tools()
    assert tools == []

    # MCP tool that returns content with non-text objects or empty content
    class BrokenMCPTool:
        name = "broken_tool"
        description = "tool with non text items"
        inputSchema = None

    class BrokenMCPSession:
        async def list_tools(self):
            class Res:
                tools = [BrokenMCPTool()]

            return Res()

        async def call_tool(self, name, arguments):
            class NonTextItem:
                pass

            class Res:
                content = [NonTextItem()]

            return Res()

    adapter2 = MCPAdapter(BrokenMCPSession())
    tools2 = await adapter2.get_tools()
    assert len(tools2) == 1
    out = await tools2[0].func()
    assert "NonTextItem" in out
