from unittest.mock import AsyncMock, patch

import httpx
import pytest

from anti_client.client import Client, _extract_project_id
from anti_client.exceptions import (
    AgentAPIError,
    AuthError,
    ModelNotFoundError,
    RateLimitError,
)
from anti_client.types import ModelsCatalog, QuotaSummary


def test_extract_project_id_adversarial_inputs():
    assert _extract_project_id(None) is None
    assert _extract_project_id("not a dict") is None
    assert _extract_project_id([]) is None
    assert _extract_project_id({}) is None
    assert _extract_project_id({"project": "   "}) is None
    assert (
        _extract_project_id({"cloudaicompanionProject": "projects/my-project-123"})
        == "my-project-123"
    )
    assert _extract_project_id({"projectId": "projects/nested-proj"}) == "nested-proj"
    assert _extract_project_id({"project": {"id": "projects/deep-id"}}) == "deep-id"
    assert _extract_project_id({"project": {"name": "projects/named-id"}}) == "named-id"


def test_raise_for_status_all_codes():
    # 200 -> no exception
    Client._raise_for_status(200, "OK")

    # 401 -> AuthError
    with pytest.raises(AuthError, match="Unauthorized"):
        Client._raise_for_status(401, "Invalid API Key")

    # 404 -> ModelNotFoundError
    with pytest.raises(ModelNotFoundError, match="Model or resource not found"):
        Client._raise_for_status(404, "Model xyz not found")

    # 429 -> RateLimitError
    error_429_body = '{"error": {"message": "Quota exceeded", "details": [{"quotaResetDelay": "3600s", "quotaResetTimeStamp": "2026-08-31T12:00:00Z"}]}}'
    with pytest.raises(RateLimitError) as exc_429:
        Client._raise_for_status(429, error_429_body)
    assert exc_429.value.quota_reset_delay == "3600s"
    assert exc_429.value.quota_reset_timestamp == "2026-08-31T12:00:00Z"

    # 500 -> AgentAPIError
    with pytest.raises(AgentAPIError, match="Internal Server Error"):
        Client._raise_for_status(500, "Internal Server Error")


@pytest.mark.asyncio
async def test_double_429_daily_and_prod_raises_rate_limit_error():
    daily_429 = httpx.Response(status_code=429, text='{"error": {"quotaResetDelay": "120s"}}')
    prod_429 = httpx.Response(status_code=429, text='{"error": {"quotaResetDelay": "300s"}}')

    client = Client(api_key="test_tok", project_id="proj")

    call_urls = []

    async def mock_post(url, *args, **kwargs):
        call_urls.append(url)
        if "daily" in url:
            return daily_429
        return prod_429

    with patch.object(client.http_client, "post", side_effect=mock_post):
        with pytest.raises(RateLimitError) as exc:
            await client.count_tokens("test")
        assert exc.value.quota_reset_delay == "300s"
        assert len(call_urls) == 2


@pytest.mark.asyncio
async def test_network_timeout_and_connect_error_fallback():
    client = Client(api_key="test_tok", project_id="proj")

    # 1. ConnectError on Daily -> succeeds on Prod
    prod_200 = httpx.Response(status_code=200, json={"totalTokens": 99})
    call_count = 0

    async def mock_post_connect_err(url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if "daily" in url:
            raise httpx.ConnectError("Connection refused to daily")
        return prod_200

    with patch.object(client.http_client, "post", side_effect=mock_post_connect_err):
        res = await client.count_tokens("test network fail")
        assert res.total_tokens == 99
        assert call_count == 2


@pytest.mark.asyncio
async def test_load_code_assist_malformed_responses():
    client = Client(api_key="test_tok", project_id="proj")

    # 1. Response without currentTier and without allowedTiers
    empty_resp = httpx.Response(status_code=200, json={})
    with patch.object(client.http_client, "post", new_callable=AsyncMock, return_value=empty_resp):
        info = await client.load_code_assist()
        assert info.current_tier is None
        assert info.allowed_tiers == []
        assert info.gcp_managed is False

    # 2. Server returns 403 Forbidden
    resp_403 = httpx.Response(status_code=403, text="Forbidden: companion project not permitted")
    with patch.object(client.http_client, "post", new_callable=AsyncMock, return_value=resp_403):
        with pytest.raises(AgentAPIError, match="Forbidden"):
            await client.load_code_assist()


@pytest.mark.asyncio
async def test_models_catalog_malformed_and_empty():
    client = Client(api_key="test_tok", project_id="proj")

    # Models response is empty dictionary
    empty_models_resp = httpx.Response(status_code=200, json={"models": {}})
    with patch.object(
        client.http_client, "post", new_callable=AsyncMock, return_value=empty_models_resp
    ):
        catalog = await client.models.get_catalog(force=True)
        assert isinstance(catalog, ModelsCatalog)
        assert catalog.models == []
        assert catalog.get_model("any") is None


@pytest.mark.asyncio
async def test_quota_summary_malformed_and_empty():
    client = Client(api_key="test_tok", project_id="proj")

    # Quota response has groups without buckets or empty groups
    weird_quota_resp = httpx.Response(
        status_code=200,
        json={
            "groups": [
                {"displayName": "Unknown Group 1", "buckets": []},
                {
                    "displayName": "Another Weird Group",
                    "buckets": [{"bucketId": "weird-1", "remainingFraction": -0.5}],
                },
            ]
        },
    )
    with patch.object(
        client.http_client, "post", new_callable=AsyncMock, return_value=weird_quota_resp
    ):
        summary = await client.get_quota_summary()
        assert isinstance(summary, QuotaSummary)
        assert summary.gemini is None
        assert summary.claude is None
        assert len(summary.groups) == 2
