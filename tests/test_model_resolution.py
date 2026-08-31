from unittest.mock import patch

import httpx
import pytest

from anti_client.client import (
    Client,
    get_clean_display_name,
    parse_effort_from_suffix,
)
from anti_client.types import Message


def test_clean_display_name():
    assert get_clean_display_name("Gemini 3.7 Flash (High)") == "Gemini 3.7 Flash"
    assert get_clean_display_name("Gemini 3.6 Flash (Low)") == "Gemini 3.6 Flash"
    assert get_clean_display_name("Gemini 3.1 Pro (Low)") == "Gemini 3.1 Pro"
    assert get_clean_display_name("Claude Sonnet 4.6 (Thinking)") == "Claude Sonnet 4.6"
    assert get_clean_display_name("GPT-OSS 120B (Medium)") == "GPT-OSS 120B"
    assert get_clean_display_name("Gemini 2.5 Flash (Extra Low)") == "Gemini 2.5 Flash"
    assert get_clean_display_name("Custom Model") == "Custom Model"


def test_parse_effort_from_suffix():
    assert parse_effort_from_suffix("gemini-3.7-flash-high") == "high"
    assert parse_effort_from_suffix("gemini-3.7-flash-medium") == "medium"
    assert parse_effort_from_suffix("gemini-3.7-flash-low") == "low"
    assert parse_effort_from_suffix("gemini-2.5-flash-extra-low") == "extra-low"
    assert parse_effort_from_suffix("claude-sonnet-4-6") is None


@pytest.mark.asyncio
async def test_tiered_model_mapping_in_generate():
    client = Client(api_key="test_token", project_id="aicode-consumers")

    sse_lines = (
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}}\n\n'
    )
    mock_resp = httpx.Response(status_code=200, text=sse_lines)

    recorded_payloads = []

    async def mock_post(url, json_data, headers, **kwargs):
        recorded_payloads.append((url, json_data))
        return mock_resp

    with patch.object(client, "_post_with_fallback", side_effect=mock_post):
        # Call with gemini-3.7-flash-high
        await client.generate(
            model="gemini-3.7-flash-high",
            messages=[Message(role="user", content="Hi")],
        )

        gen_payloads = [p for url, p in recorded_payloads if "streamGenerateContent" in url]
        assert len(gen_payloads) == 1
        p = gen_payloads[0]
        # Should be mapped to gemini-3.7-flash-tiered with thinkingLevel high
        assert p["model"] == "gemini-3.7-flash-tiered"
        assert p["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"


@pytest.mark.asyncio
async def test_claude_thinking_budget_mapping():
    client = Client(api_key="test_token", project_id="aicode-consumers")

    sse_lines = (
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}}\n\n'
    )
    mock_resp = httpx.Response(status_code=200, text=sse_lines)

    recorded_payloads = []

    async def mock_post(url, json_data, headers, **kwargs):
        recorded_payloads.append((url, json_data))
        return mock_resp

    with patch.object(client, "_post_with_fallback", side_effect=mock_post):
        # Call claude model with thinking_budget
        await client.generate(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="Hi")],
            thinking_budget=4096,
        )

        gen_payloads = [p for url, p in recorded_payloads if "streamGenerateContent" in url]
        assert len(gen_payloads) == 1
        p = gen_payloads[0]
        assert p["model"] == "claude-sonnet-4-6"
        assert p["request"]["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 4096
