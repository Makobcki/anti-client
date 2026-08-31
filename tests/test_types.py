import os
import tempfile

from anti_client.types import (
    CountTokensResult,
    FileAttachment,
    GenerateOptions,
    ModelInfo,
    ModelsCatalog,
    QuotaBucket,
    QuotaGroup,
    QuotaSummary,
)


def test_file_attachment_from_bytes_and_save():
    raw_data = b"Hello, test image data"
    attachment = FileAttachment.from_bytes(raw_data, mime_type="image/png")
    assert attachment.mime_type == "image/png"
    assert attachment.to_bytes() == raw_data

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp_path = tmp.name

    try:
        attachment.save(tmp_path)
        with open(tmp_path, "rb") as f:
            saved_bytes = f.read()
        assert saved_bytes == raw_data

        from_file_att = FileAttachment.from_file(tmp_path)
        assert from_file_att.mime_type == "image/png"
        assert from_file_att.to_bytes() == raw_data
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_quota_summary_properties():
    gemini_buckets = [
        QuotaBucket(
            bucket_id="gemini-weekly",
            display_name="Weekly Limit",
            window="weekly",
            reset_time="2026-09-04T19:56:36Z",
            description="Weekly refresh",
            remaining_fraction=0.42,
        ),
        QuotaBucket(
            bucket_id="gemini-5h",
            display_name="Five Hour Limit",
            window="5h",
            reset_time="2026-08-31T02:34:57Z",
            description="5h refresh",
            remaining_fraction=0.92,
        ),
    ]

    claude_buckets = [
        QuotaBucket(
            bucket_id="3p-weekly",
            display_name="Weekly Limit",
            window="weekly",
            reset_time="2026-09-06T11:54:50Z",
            description="Weekly 3p refresh",
            remaining_fraction=0.88,
        ),
        QuotaBucket(
            bucket_id="3p-5h",
            display_name="Five Hour Limit",
            window="5h",
            reset_time="2026-08-31T02:11:33Z",
            description="5h 3p refresh",
            remaining_fraction=0.99,
        ),
    ]

    summary = QuotaSummary(
        groups=[
            QuotaGroup(
                display_name="Gemini Models", description="Gemini models", buckets=gemini_buckets
            ),
            QuotaGroup(
                display_name="Claude and GPT models",
                description="3P models",
                buckets=claude_buckets,
            ),
        ],
        description="User quota limits",
    )

    assert summary.gemini is not None
    assert summary.gemini.weekly is not None
    assert summary.gemini.weekly.remaining_fraction == 0.42
    assert summary.gemini.five_hour is not None
    assert summary.gemini.five_hour.remaining_fraction == 0.92

    assert summary.claude is not None
    assert summary.claude.weekly is not None
    assert summary.claude.weekly.remaining_fraction == 0.88
    assert summary.claude.five_hour is not None
    assert summary.claude.five_hour.remaining_fraction == 0.99


def test_models_catalog_lookup():
    m1 = ModelInfo(
        id="gemini-3.5-flash-low",
        display_name="Gemini 3.5 Flash (Low)",
        clean_display_name="Gemini 3.5 Flash",
        internal_model_id="M20",
        model_provider="MODEL_PROVIDER_GOOGLE",
        api_provider="API_PROVIDER_GOOGLE_GEMINI",
        max_tokens=1048576,
    )
    catalog = ModelsCatalog(
        models=[m1],
        default_agent_model_id="gemini-3.5-flash-low",
        agent_model_sorts=["gemini-3.5-flash-low"],
    )

    assert catalog.get_model("gemini-3.5-flash-low") == m1
    assert catalog.get_model("M20") == m1
    assert catalog.get_model("nonexistent") is None


def test_generate_options_defaults():
    opts = GenerateOptions(temperature=0.5, thinking_level="high", max_steps=10)
    assert opts.temperature == 0.5
    assert opts.thinking_level == "high"
    assert opts.max_steps == 10


def test_count_tokens_result_integer_operations():
    res = CountTokensResult(total_tokens=42)
    assert int(res) == 42
    assert res == 42
    assert res == CountTokensResult(total_tokens=42)
    assert res != 40
    assert res > 40
    assert res >= 42
    assert res < 50
    assert res <= 42
    assert res + 8 == 50
    assert 8 + res == 50
    assert res - 2 == 40
    assert repr(res) == "CountTokensResult(total_tokens=42)"
