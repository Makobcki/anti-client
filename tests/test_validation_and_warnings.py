import warnings

import pytest

from anti_client.client import Client
from anti_client.types import (
    FileAttachment,
    Message,
    ModelInfo,
    ModelsCatalog,
)


@pytest.fixture
def sample_catalog():
    model_gemini = ModelInfo(
        id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        clean_display_name="Gemini 2.5 Flash",
        internal_model_id="M1",
        model_provider="MODEL_PROVIDER_GOOGLE",
        api_provider="API_PROVIDER_GOOGLE_GEMINI",
        max_tokens=1048576,
        max_output_tokens=8192,
        supports_images=True,
        supports_video=True,
        supports_thinking=True,
        thinking_budget=4000,
        min_thinking_budget=32,
        supported_mime_types={
            "text/plain": True,
            "application/json": True,
            "image/png": True,
            "image/jpeg": True,
        },
    )

    model_text_only = ModelInfo(
        id="text-only-model",
        display_name="Text Only",
        clean_display_name="Text Only",
        internal_model_id="M2",
        model_provider="MODEL_PROVIDER_GOOGLE",
        api_provider="API_PROVIDER_GOOGLE_GEMINI",
        max_tokens=32000,
        max_output_tokens=2048,
        supports_images=False,
        supports_video=False,
        supports_thinking=False,
        supported_mime_types={"text/plain": True},
    )

    model_no_images = ModelInfo(
        id="no-images-model",
        display_name="No Images",
        clean_display_name="No Images",
        internal_model_id="M3",
        model_provider="MODEL_PROVIDER_GOOGLE",
        api_provider="API_PROVIDER_GOOGLE_GEMINI",
        max_tokens=32000,
        max_output_tokens=2048,
        supports_images=False,
        supports_video=False,
        supports_thinking=False,
        supported_mime_types={"image/png": True},  # explicitly test supports_images check
    )

    return ModelsCatalog(
        models=[model_gemini, model_text_only, model_no_images],
        default_agent_model_id="gemini-2.5-flash",
        deprecated_model_ids=["old-deprecated-model"],
        deprecated_model_map={"old-deprecated-model": "gemini-2.5-flash"},
        web_search_model_ids=["gemini-2.5-flash"],
        image_generation_model_ids=["gemini-3.1-flash-image"],
    )


def test_mime_type_validation_error(sample_catalog):
    client = Client(api_key="dummy_key", project_id="dummy_proj")
    invalid_att = FileAttachment(mime_type="application/zip", data="UEsDB...")
    msg = Message(role="user", content="Hello", attachments=[invalid_att])

    with pytest.raises(ValueError, match="MIME type 'application/zip' is not supported"):
        client._validate_request_params(
            model="gemini-2.5-flash",
            catalog=sample_catalog,
            messages=[msg],
            max_output_tokens=1024,
            thinking_budget=None,
        )


def test_image_support_validation_error(sample_catalog):
    client = Client(api_key="dummy_key", project_id="dummy_proj")
    img_att = FileAttachment(mime_type="image/png", data="iVBORw0KGgo...")
    msg = Message(role="user", content="Here is an image", attachments=[img_att])

    with pytest.raises(ValueError, match="does not support image inputs"):
        client._validate_request_params(
            model="no-images-model",
            catalog=sample_catalog,
            messages=[msg],
            max_output_tokens=1024,
            thinking_budget=None,
        )


def test_max_output_tokens_limit_validation(sample_catalog):
    client = Client(api_key="dummy_key", project_id="dummy_proj")
    msg = Message(role="user", content="Hello")

    with pytest.raises(ValueError, match="max_output_tokens .* exceeds model maximum limit"):
        client._validate_request_params(
            model="text-only-model",
            catalog=sample_catalog,
            messages=[msg],
            max_output_tokens=5000,  # exceeds 2048
            thinking_budget=None,
        )


def test_thinking_budget_validation(sample_catalog):
    client = Client(api_key="dummy_key", project_id="dummy_proj")
    msg = Message(role="user", content="Hello")

    # Model does not support thinking
    with pytest.raises(ValueError, match="does not support reasoning/thinking"):
        client._validate_request_params(
            model="text-only-model",
            catalog=sample_catalog,
            messages=[msg],
            max_output_tokens=1024,
            thinking_budget=100,
        )

    # Thinking budget below min
    with pytest.raises(ValueError, match="thinking_budget .* is below model minimum"):
        client._validate_request_params(
            model="gemini-2.5-flash",
            catalog=sample_catalog,
            messages=[msg],
            max_output_tokens=1024,
            thinking_budget=10,  # min is 32
        )

    # Thinking budget exceeds max
    with pytest.raises(ValueError, match="thinking_budget .* exceeds model maximum"):
        client._validate_request_params(
            model="gemini-2.5-flash",
            catalog=sample_catalog,
            messages=[msg],
            max_output_tokens=1024,
            thinking_budget=10000,  # max is 4000
        )


def test_deprecated_model_warning(sample_catalog):
    client = Client(api_key="dummy_key", project_id="dummy_proj")
    msg = Message(role="user", content="Hello")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client._validate_request_params(
            model="old-deprecated-model",
            catalog=sample_catalog,
            messages=[msg],
            max_output_tokens=1024,
            thinking_budget=None,
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "is deprecated. Recommended alternative: 'gemini-2.5-flash'" in str(w[-1].message)
