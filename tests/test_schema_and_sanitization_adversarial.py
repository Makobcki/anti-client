from anti_client.client import sanitize_params_for_google


def test_sanitize_params_for_google_deep_nesting_and_unsupported_keywords():
    # Complex JSON schema with keywords that Google Gemini API rejects ($schema, additionalProperties, definitions, etc.)
    raw_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://example.com/schema.json",
        "title": "ComplexToolParams",
        "type": "object",
        "additionalProperties": False,
        "required": ["user_id", "settings"],
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Unique user ID",
            },
            "settings": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "theme": {
                        "type": "string",
                        "enum": ["light", "dark"],
                    },
                    "notifications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "channel": {"type": "string"},
                                "enabled": {"type": "boolean"},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
            },
        },
        "definitions": {"extra": {"type": "string"}},
        "$defs": {"other": {"type": "number"}},
        "allOf": [{"type": "object"}],
        "anyOf": [{"type": "string"}],
        "oneOf": [{"type": "integer"}],
    }

    clean = sanitize_params_for_google(raw_schema)

    # Verify disallowed top-level keys are removed
    assert "$schema" not in clean
    assert "$id" not in clean
    assert "$ref" not in clean
    assert "definitions" not in clean
    assert "$defs" not in clean
    assert "additionalProperties" not in clean
    assert "allOf" not in clean
    assert "anyOf" not in clean
    assert "oneOf" not in clean

    # Verify nested object
    settings_prop = clean["properties"]["settings"]
    assert "additionalProperties" not in settings_prop
    assert settings_prop["type"] == "object"

    # Verify nested array of objects
    notif_prop = settings_prop["properties"]["notifications"]
    assert notif_prop["type"] == "array"
    assert "additionalProperties" not in notif_prop["items"]
    assert notif_prop["items"]["type"] == "object"
    assert notif_prop["items"]["properties"]["enabled"]["type"] == "boolean"


def test_sanitize_params_for_google_edge_cases():
    assert sanitize_params_for_google(None) is None
    assert sanitize_params_for_google({}) == {}
    assert sanitize_params_for_google("non-dict") == "non-dict"
    assert sanitize_params_for_google([{"type": "string", "$schema": "url"}]) == [
        {"type": "string"}
    ]
