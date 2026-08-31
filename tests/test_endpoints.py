from unittest.mock import AsyncMock, patch

import httpx
import pytest

from anti_client.client import Client
from anti_client.types import (
    CodeAssistInfo,
    CountTokensResult,
    FileAttachment,
    Message,
    ModelsCatalog,
    QuotaSummary,
    Tool,
)


@pytest.mark.asyncio
async def test_load_code_assist_endpoint():
    mock_resp = httpx.Response(
        status_code=200,
        json={
            "currentTier": {
                "id": "free-tier",
                "name": "Antigravity",
                "description": "Free tier",
                "upgradeSubscriptionUri": "https://codeassist.google.com/upgrade",
            },
            "allowedTiers": [
                {"id": "free-tier", "name": "Antigravity", "description": "Free", "isDefault": True}
            ],
            "cloudaicompanionProject": "aicode-consumers",
            "gcpManaged": False,
            "upgradeSubscriptionUri": "https://codeassist.google.com/upgrade",
        },
    )

    client = Client(api_key="test_token", project_id="aicode-consumers")
    with patch.object(client.http_client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        info = await client.load_code_assist()

        assert isinstance(info, CodeAssistInfo)
        assert info.current_tier is not None
        assert info.current_tier.id == "free-tier"
        assert info.companion_project_id == "aicode-consumers"
        assert len(info.allowed_tiers) == 1
        assert info.allowed_tiers[0].is_default is True


@pytest.mark.asyncio
async def test_fetch_available_models_catalog():
    mock_resp = httpx.Response(
        status_code=200,
        json={
            "models": {
                "gemini-3.5-flash-low": {
                    "model": "MODEL_PLACEHOLDER_M20",
                    "displayName": "Gemini 3.5 Flash (Low)",
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "modelProvider": "MODEL_PROVIDER_GOOGLE",
                    "maxTokens": 1048576,
                    "maxOutputTokens": 65536,
                    "supportsImages": True,
                    "supportsThinking": True,
                    "thinkingBudget": 4000,
                    "minThinkingBudget": 32,
                    "quotaInfo": {"remainingFraction": 0.92, "resetTime": "2026-08-31T02:34:57Z"},
                    "supportedMimeTypes": {"text/plain": True, "image/png": True},
                }
            },
            "defaultAgentModelId": "gemini-3.5-flash-low",
            "agentModelSorts": ["gemini-3.5-flash-low"],
            "commandModelIds": ["gemini-2.5-flash"],
            "imageGenerationModelIds": ["gemini-3.1-flash-image"],
            "webSearchModelIds": ["gemini-2.5-pro"],
            "deprecatedModelIds": ["old-model"],
        },
    )

    client = Client(api_key="test_token", project_id="aicode-consumers")
    with patch.object(client.http_client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        catalog = await client.models.get_catalog()

        assert isinstance(catalog, ModelsCatalog)
        assert catalog.default_agent_model_id == "gemini-3.5-flash-low"
        assert len(catalog.models) == 1
        m = catalog.models[0]
        assert m.id == "gemini-3.5-flash-low"
        assert m.clean_display_name == "Gemini 3.5 Flash"
        assert m.thinking_level == "low"
        assert m.quota_info is not None
        assert m.quota_info.remaining_fraction == 0.92
        assert "image/png" in m.supported_mime_types
        assert catalog.web_search_model_ids == ["gemini-2.5-pro"]
        assert catalog.image_generation_model_ids == ["gemini-3.1-flash-image"]


@pytest.mark.asyncio
async def test_fetch_catalog_unlimited_thinking_budget():
    mock_resp = httpx.Response(
        status_code=200,
        json={
            "models": {
                "gemini-3.7-flash": {
                    "model": "MODEL_GEMINI_37_FLASH",
                    "displayName": "Gemini 3.7 Flash",
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "modelProvider": "MODEL_PROVIDER_GOOGLE",
                    "maxTokens": 1048576,
                    "maxOutputTokens": -1,
                    "supportsImages": True,
                    "supportsThinking": True,
                    "thinkingBudget": -1,
                    "minThinkingBudget": -1,
                }
            },
            "defaultAgentModelId": "gemini-3.7-flash",
        },
    )

    client = Client(api_key="test_token", project_id="aicode-consumers")
    with patch.object(client.http_client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        catalog = await client.models.get_catalog()

        m = catalog.get_model("gemini-3.7-flash")
        assert m is not None
        assert m.thinking_budget is None
        assert m.min_thinking_budget is None
        assert m.max_output_tokens is None


@pytest.mark.asyncio
async def test_retrieve_user_quota_summary():
    mock_resp = httpx.Response(
        status_code=200,
        json={
            "groups": [
                {
                    "displayName": "Gemini Models",
                    "description": "Models: Flash, Pro",
                    "buckets": [
                        {
                            "bucketId": "gemini-weekly",
                            "displayName": "Weekly Limit",
                            "window": "weekly",
                            "resetTime": "2026-09-04T19:56:36Z",
                            "description": "Weekly refresh",
                            "remainingFraction": 0.45,
                        },
                        {
                            "bucketId": "gemini-5h",
                            "displayName": "5h Limit",
                            "window": "5h",
                            "resetTime": "2026-08-31T02:34:57Z",
                            "description": "5h refresh",
                            "remainingFraction": 0.95,
                        },
                    ],
                }
            ],
            "description": "Shared quotas",
        },
    )

    client = Client(api_key="test_token", project_id="aicode-consumers")
    with patch.object(client.http_client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        quota_summary = await client.get_quota_summary()

        assert isinstance(quota_summary, QuotaSummary)
        assert quota_summary.gemini is not None
        assert quota_summary.gemini.weekly.remaining_fraction == 0.45
        assert quota_summary.gemini.five_hour.remaining_fraction == 0.95


@pytest.mark.asyncio
async def test_count_tokens():
    mock_resp = httpx.Response(
        status_code=200,
        json={"totalTokens": 42},
    )

    client = Client(api_key="test_token", project_id="aicode-consumers")
    with patch.object(client.http_client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        result = await client.count_tokens("Hello world from testing")

        assert isinstance(result, CountTokensResult)
        assert result.total_tokens == 42


@pytest.mark.asyncio
async def test_backend_fallback_on_429():
    daily_429 = httpx.Response(status_code=429, text="Resource exhausted on daily")
    prod_200 = httpx.Response(status_code=200, json={"totalTokens": 10})

    client = Client(api_key="test_token", project_id="aicode-consumers")

    async def mock_post(url, *args, **kwargs):
        if "daily" in url:
            return daily_429
        return prod_200

    with patch.object(client.http_client, "post", side_effect=mock_post):
        result = await client.count_tokens("Test fallback")
        assert result.total_tokens == 10


@pytest.mark.asyncio
async def test_count_tokens_with_messages_tools_and_system():
    recorded_requests = []
    mock_resp = httpx.Response(status_code=200, json={"totalTokens": 128})
    client = Client(api_key="test_token", project_id="aicode-consumers")

    async def mock_post(url, json_data, headers, **kwargs):
        recorded_requests.append((url, json_data))
        return mock_resp

    with patch.object(client, "_post_with_fallback", side_effect=mock_post):
        tool = Tool(
            name="get_time",
            description="Returns current time",
            parameters={"type": "object", "properties": {"timezone": {"type": "string"}}},
        )
        att = FileAttachment(mime_type="image/png", data="iVBORw0KGgo...")
        msg = Message(role="user", content="Describe this image", attachments=[att])

        result = await client.count_tokens(
            messages=[msg],
            model="gemini-3.7-flash-high",
            tools=[tool],
            system_instruction="You are a vision assistant.",
        )

        assert result == 128
        count_reqs = [r for r in recorded_requests if "countTokens" in r[0]]
        assert len(count_reqs) == 1
        url, body = count_reqs[0]
        assert "countTokens" in url
        assert body["model"] == "gemini-3.7-flash-tiered"
        req = body["request"]
        assert "systemInstruction" in req
        assert req["systemInstruction"]["parts"][0]["text"] == "You are a vision assistant."
        assert "tools" in req
        assert len(req["tools"][0]["functionDeclarations"]) == 1
        assert req["tools"][0]["functionDeclarations"][0]["name"] == "get_time"
        assert len(req["contents"]) == 1
        parts = req["contents"][0]["parts"]
        assert parts[0]["text"] == "Describe this image"
        assert parts[1]["inlineData"]["mimeType"] == "image/png"


@pytest.mark.asyncio
async def test_thought_parts_serialization_in_payload_and_count_tokens():
    recorded_requests = []
    mock_resp = httpx.Response(status_code=200, json={"totalTokens": 64})
    client = Client(api_key="test_token", project_id="aicode-consumers")

    async def mock_post(url, json_data, headers, **kwargs):
        recorded_requests.append((url, json_data))
        return mock_resp

    with patch.object(client, "_post_with_fallback", side_effect=mock_post):
        msg = Message(
            role="assistant",
            content="Final answer",
            thought="Thinking about solution...",
            thought_signature="sig_12345",
        )
        await client.count_tokens(messages=[msg])

        count_reqs = [r for r in recorded_requests if "countTokens" in r[0]]
        assert len(count_reqs) == 1
        url, body = count_reqs[0]
        contents = body["request"]["contents"]
        assert len(contents) == 1
        parts = contents[0]["parts"]
        # Thought part must be first and contain text and thought: True (TYPE_BOOL)
        assert parts[0]["text"] == "Thinking about solution..."
        assert parts[0]["thought"] is True
        assert parts[0]["thoughtSignature"] == "sig_12345"
        # Content part must follow
        assert parts[1]["text"] == "Final answer"

    # Test generate payload
    gen_mock_resp = httpx.Response(
        status_code=200,
        text='data: {"response": {"candidates": [{"content": {"parts": [{"text": "World"}]}}]}}\n\n',
    )
    gen_requests = []

    async def mock_gen_post(url, json_data, headers, **kwargs):
        gen_requests.append((url, json_data))
        return gen_mock_resp

    with patch.object(client, "_post_with_fallback", side_effect=mock_gen_post):
        await client.generate(
            model="gemini-3.1-pro-low",
            messages=[
                Message(role="user", content="Hi"),
                Message(
                    role="assistant",
                    content="Hello!",
                    thought="Thinking...",
                    thought_signature="sig_abc",
                ),
            ],
        )

        gen_reqs = [r for r in gen_requests if "streamGenerateContent" in r[0]]
        assert len(gen_reqs) == 1
        url, body = gen_reqs[0]
        req_contents = body["request"]["contents"]
        assert len(req_contents) == 2
        model_turn_parts = req_contents[1]["parts"]
        assert model_turn_parts[0]["text"] == "Thinking..."
        assert model_turn_parts[0]["thought"] is True
        assert model_turn_parts[0]["thoughtSignature"] == "sig_abc"
        assert model_turn_parts[1]["text"] == "Hello!"


