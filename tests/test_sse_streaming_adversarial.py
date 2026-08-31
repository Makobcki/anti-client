from unittest.mock import AsyncMock, patch

import httpx
import pytest

from anti_client.client import Client
from anti_client.types import ChatResponse, Message


@pytest.mark.asyncio
async def test_sse_streaming_corrupted_json_lines_and_abrupt_done():
    # Stream with corrupted json lines, empty lines, and premature DONE
    sse_data = (
        "data: {corrupted json line}\n\n"
        ": comment line to ignore\n\n"
        "\n\n"
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Valid part 1 "}]}}]}}\n\n'
        "data: [DONE]\n\n"
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Should not appear"}]}}]}}\n\n'
    )
    mock_resp = httpx.Response(status_code=200, text=sse_data)

    client = Client(api_key="test_tok", project_id="proj")
    with patch.object(
        client, "_post_with_fallback", new_callable=AsyncMock, return_value=mock_resp
    ):
        res: ChatResponse = await client.generate(
            model="gemini-2.5-flash",
            messages=[Message(role="user", content="Hi")],
            stream=False,
        )
        assert res.text == "Valid part 1 "


@pytest.mark.asyncio
async def test_sse_stream_parallel_multiple_tool_calls():
    # Single response returning 3 parallel function calls
    sse_data = (
        'data: {"response": {"candidates": [{"content": {"parts": ['
        '{"functionCall": {"name": "tool_a", "args": {"p": 1}}},'
        '{"functionCall": {"name": "tool_b", "args": {"q": "hello"}}},'
        '{"functionCall": {"name": "tool_c", "args": {}}}'
        "]}}]}}\n\n"
    )
    mock_resp = httpx.Response(status_code=200, text=sse_data)

    client = Client(api_key="test_tok", project_id="proj")
    with patch.object(
        client, "_post_with_fallback", new_callable=AsyncMock, return_value=mock_resp
    ):
        res: ChatResponse = await client.generate(
            model="gemini-2.5-flash",
            messages=[Message(role="user", content="Call tools")],
            stream=False,
        )
        assert res.tool_calls is not None
        assert len(res.tool_calls) == 3
        assert [tc.name for tc in res.tool_calls] == ["tool_a", "tool_b", "tool_c"]
        assert res.tool_calls[1].arguments == {"q": "hello"}


@pytest.mark.asyncio
async def test_sse_streaming_thinking_signature_and_details():
    sse_data = (
        'data: {"response": {"candidates": [{"content": {"parts": ['
        '{"thought": true, "text": "I am thinking deeply...", "thoughtSignature": "sig_abc123"},'
        '{"text": "Here is the final answer."}'
        "]}}], "
        '"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 50, "totalTokenCount": 70, "thoughtsTokenCount": 35}}}\n\n'
    )
    mock_resp = httpx.Response(status_code=200, text=sse_data)

    client = Client(api_key="test_tok", project_id="proj")
    with patch.object(
        client, "_post_with_fallback", new_callable=AsyncMock, return_value=mock_resp
    ):
        res: ChatResponse = await client.generate(
            model="gemini-2.5-flash",
            messages=[Message(role="user", content="Question")],
            stream=False,
        )
        assert res.thought == "I am thinking deeply..."
        assert res.thought_signature == "sig_abc123"
        assert res.text == "Here is the final answer."
        assert res.usage.thoughts_tokens == 35


@pytest.mark.asyncio
async def test_sse_stream_iterator_chunks():
    # Test async generator yielding StreamChunk
    sse_data = (
        'data: {"response": {"candidates": [{"content": {"parts": [{"thought": true, "text": "Step 1 reasoning "}]}}]}}\n\n'
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]}}\n\n'
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "World!"}]}}], "usageMetadata": {"totalTokenCount": 10}}}\n\n'
    )

    client = Client(api_key="test_tok", project_id="proj")

    async def mock_stream(*args, **kwargs):
        resp = httpx.Response(status_code=200, text=sse_data)
        async for chunk in client._stream_response(resp):
            yield chunk

    with patch.object(client, "_stream_with_fallback", side_effect=mock_stream):
        stream = await client.generate(
            model="gemini-2.5-flash",
            messages=[Message(role="user", content="Hi")],
            stream=True,
        )
        chunks = [c async for c in stream]

        assert len(chunks) == 3
        assert chunks[0].thought == "Step 1 reasoning "
        assert chunks[1].text == "Hello "
        assert chunks[2].text == "World!"
        assert chunks[2].usage is not None
        assert chunks[2].usage.total_tokens == 10
