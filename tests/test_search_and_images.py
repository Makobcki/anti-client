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
def mock_catalog():
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
async def test_search_grounding(mock_catalog):
    sse_lines = (
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Python 3.14 was released recently."}]}, '
        '"groundingMetadata": {"groundingChunks": [{"web": {"title": "Python Official", "uri": "https://python.org"}}]}}], '
        '"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30}}}\n\n'
    )
    mock_resp = httpx.Response(status_code=200, text=sse_lines)

    client = Client(api_key="test_token", project_id="aicode-consumers")
    with patch.object(
        client.models, "get_catalog", new_callable=AsyncMock, return_value=mock_catalog
    ):
        with patch.object(
            client, "_post_with_fallback", new_callable=AsyncMock, return_value=mock_resp
        ):
            res = await client.search("Python 3.14 version info")

            assert isinstance(res, SearchResponse)
            assert "Python 3.14 was released recently." in res.text
            assert len(res.sources) == 1
            assert res.sources[0].title == "Python Official"
            assert res.sources[0].uri == "https://python.org"


@pytest.mark.asyncio
async def test_generate_image(mock_catalog):
    raw_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    sse_lines = (
        f'data: {{"response": {{"candidates": [{{"content": {{"parts": [{{"inlineData": {{"mimeType": "image/png", "data": "{raw_b64}"}}}}]}}}}], '
        f'"usageMetadata": {{"promptTokenCount": 15, "candidatesTokenCount": 1024, "totalTokenCount": 1039}}, '
        f'"modelVersion": "gemini-3.1-flash-image"}}}}\n\n'
    )
    mock_resp = httpx.Response(status_code=200, text=sse_lines)

    client = Client(api_key="test_token", project_id="aicode-consumers")
    with patch.object(
        client.models, "get_catalog", new_callable=AsyncMock, return_value=mock_catalog
    ):
        with patch.object(
            client, "_post_with_fallback", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            res = await client.generate_image("A futuristic city skyline", aspect_ratio="16:9")

            assert isinstance(res, ImageGenerationResponse)
            assert len(res.images) == 1
            img = res.image
            assert img is not None
            assert img.mime_type == "image/png"
            assert img.data == raw_b64
            assert len(img.to_bytes()) > 0

            # Verify payload structure
            call_kwargs = mock_post.call_args.kwargs
            payload = call_kwargs["json_data"]
            assert payload["request"]["contents"][0]["parts"][0]["text"] == (
                "A futuristic city skyline\nAspect ratio: 16:9"
            )
            assert "aspectRatio" not in payload["request"]["generationConfig"]
            assert "safetySettings" in payload["request"]


@pytest.mark.asyncio
async def test_search_non_default_model_warning(mock_catalog):
    sse_lines = (
        'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Found"}]}}]}}\n\n'
    )
    mock_resp = httpx.Response(status_code=200, text=sse_lines)

    client = Client(api_key="test_token", project_id="aicode-consumers")
    with patch.object(
        client.models, "get_catalog", new_callable=AsyncMock, return_value=mock_catalog
    ):
        with patch.object(
            client, "_post_with_fallback", new_callable=AsyncMock, return_value=mock_resp
        ):
            with pytest.warns(UserWarning, match="is not in webSearchModelIds"):
                res = await client.search("Search query", model="custom-search-model")
                assert res.text == "Found"


@pytest.mark.asyncio
async def test_image_generation_non_default_model_warning(mock_catalog):
    raw_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    sse_lines = f'data: {{"response": {{"candidates": [{{"content": {{"parts": [{{"inlineData": {{"mimeType": "image/png", "data": "{raw_b64}"}}}}]}}}}]}}}}\n\n'
    mock_resp = httpx.Response(status_code=200, text=sse_lines)

    client = Client(api_key="test_token", project_id="aicode-consumers")
    with patch.object(
        client.models, "get_catalog", new_callable=AsyncMock, return_value=mock_catalog
    ):
        with patch.object(
            client, "_post_with_fallback", new_callable=AsyncMock, return_value=mock_resp
        ):
            with pytest.warns(UserWarning, match="is not in imageGenerationModelIds"):
                res = await client.generate_image("Draw a cat", model="custom-image-model")
                assert len(res.images) == 1
