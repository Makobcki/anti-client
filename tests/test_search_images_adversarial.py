from unittest.mock import AsyncMock, patch

import httpx
import pytest

from anti_client.client import Client
from anti_client.types import (
    ImageGenerationResponse,
    ModelInfo,
    ModelsCatalog,
    SearchResponse,
)


@pytest.fixture
def test_catalog():
    return ModelsCatalog(
        models=[
            ModelInfo(
                id="gemini-2.5-pro",
                display_name="Gemini 2.5 Pro",
                clean_display_name="Gemini 2.5 Pro",
                internal_model_id="M1",
                model_provider="MODEL_PROVIDER_GOOGLE",
                api_provider="API_PROVIDER_GOOGLE_GEMINI",
                max_tokens=1048576,
            ),
            ModelInfo(
                id="gemini-3.1-flash-image",
                display_name="Gemini 3.1 Flash Image",
                clean_display_name="Gemini 3.1 Flash Image",
                internal_model_id="M2",
                model_provider="MODEL_PROVIDER_GOOGLE",
                api_provider="API_PROVIDER_GOOGLE_GEMINI",
                max_tokens=1048576,
            ),
        ],
        default_agent_model_id="gemini-2.5-pro",
        web_search_model_ids=["gemini-2.5-pro"],
        image_generation_model_ids=["gemini-3.1-flash-image"],
    )


@pytest.mark.asyncio
async def test_search_zero_candidates_and_missing_grounding(test_catalog):
    # Response has no grounding metadata at all
    sse_lines = 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Answer without web sources"}]}}]}}\n\n'
    mock_resp = httpx.Response(status_code=200, text=sse_lines)

    client = Client(api_key="test_tok", project_id="proj")
    with patch.object(
        client.models, "get_catalog", new_callable=AsyncMock, return_value=test_catalog
    ):
        with patch.object(
            client, "_post_with_fallback", new_callable=AsyncMock, return_value=mock_resp
        ):
            res = await client.search("query without web hits")
            assert isinstance(res, SearchResponse)
            assert res.text == "Answer without web sources"
            assert res.sources == []


@pytest.mark.asyncio
async def test_generate_image_missing_inline_data_and_text_only(test_catalog):
    # Response has only text parts instead of inlineData (e.g. model refused or failed generation)
    sse_lines = 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "I cannot draw this image due to safety policies."}]}}]}}\n\n'
    mock_resp = httpx.Response(status_code=200, text=sse_lines)

    client = Client(api_key="test_tok", project_id="proj")
    with patch.object(
        client.models, "get_catalog", new_callable=AsyncMock, return_value=test_catalog
    ):
        with patch.object(
            client, "_post_with_fallback", new_callable=AsyncMock, return_value=mock_resp
        ):
            res = await client.generate_image("A forbidden prompt")
            assert isinstance(res, ImageGenerationResponse)
            assert res.images == []
            assert res.image is None
