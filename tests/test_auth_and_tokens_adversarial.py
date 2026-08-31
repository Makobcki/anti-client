import json
import os
import tempfile
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from anti_client.client import Client, authenticate


def test_corrupted_accounts_json_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = os.path.join(tmpdir, "accounts.json")

        # 1. Invalid JSON syntax
        with open(fake_path, "w") as f:
            f.write("{invalid json syntax, trailing comma,}")

        with patch.dict(os.environ, {"ANTI_API_DATA_DIR": tmpdir, "MY_API_KEY": ""}):
            client = Client.__new__(Client)
            info = client._load_account_info()
            assert info == {}

        # 2. File containing non-dict JSON (e.g., list or integer)
        with open(fake_path, "w") as f:
            f.write(json.dumps(["not", "a", "dict"]))

        with patch.dict(os.environ, {"ANTI_API_DATA_DIR": tmpdir, "MY_API_KEY": ""}):
            client = Client.__new__(Client)
            info = client._load_account_info()
            assert info == {}

        # 3. File with empty accounts list
        with open(fake_path, "w") as f:
            f.write(json.dumps({"accounts": []}))

        with patch.dict(os.environ, {"ANTI_API_DATA_DIR": tmpdir, "MY_API_KEY": ""}):
            client = Client.__new__(Client)
            info = client._load_account_info()
            assert info == {}


@pytest.mark.asyncio
async def test_token_refresh_adversarial_failures():
    client = Client(api_key="expired_tok", project_id="proj")
    client.refresh_token = "bad_refresh_tok"
    client.expires_at = 0  # Force expiry

    # 1. Refresh endpoint returns 400 invalid_grant (revoked token)
    mock_400 = httpx.Response(
        status_code=400,
        json={"error": "invalid_grant", "error_description": "Token has been revoked"},
    )
    with patch.object(client.http_client, "post", new_callable=AsyncMock, return_value=mock_400):
        await client._check_and_refresh_token()
        # Should not crash, keeps existing state gracefully

    # 2. Refresh endpoint returns 200 with new token
    mock_200 = httpx.Response(
        status_code=200, json={"access_token": "fresh_tok", "expires_in": 3600}
    )
    with patch.object(client.http_client, "post", new_callable=AsyncMock, return_value=mock_200):
        await client._check_and_refresh_token()
        assert client.api_key == "fresh_tok"


def test_save_account_info_auto_creates_missing_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        deep_dir = os.path.join(tmpdir, "subdir1", "subdir2")
        fake_account_file = os.path.join(deep_dir, "accounts.json")

        with patch.dict(os.environ, {"ANTI_API_DATA_DIR": deep_dir}):
            client = Client(api_key="new_tok", project_id="new_proj")
            client.refresh_token = "new_rf"
            client.expires_at = 123456789.0
            client._save_account_info()

            assert os.path.exists(fake_account_file)
            with open(fake_account_file) as f:
                data = json.load(f)
            assert data["accounts"][0]["accessToken"] == "new_tok"
            assert data["accounts"][0]["refreshToken"] == "new_rf"
            assert data["accounts"][0]["projectId"] == "new_proj"
            assert data["accounts"][0]["expiresAt"] == 123456789.0


def test_authenticate_flow_with_simulated_callback():
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_account_file = os.path.join(tmpdir, "accounts.json")

        # Mock browser open to trigger callback via real socket / urllib
        def fake_open(url):
            import urllib.parse

            parsed = urllib.parse.urlparse(url)
            params = urllib.parse.parse_qs(parsed.query)
            redir = params["redirect_uri"][0]
            port = urllib.parse.urlparse(redir).port

            time.sleep(0.05)
            # Use raw urllib.request to bypass any httpx patches
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/oauth-callback?code=mock_auth_code&state={params['state'][0]}"
            )
            try:
                urllib.request.urlopen(req)
            except Exception:
                pass

        mock_token_resp = httpx.Response(
            status_code=200,
            json={
                "access_token": "exchanged_tok",
                "refresh_token": "exchanged_rf",
                "expires_in": 3600,
            },
        )
        mock_userinfo_resp = httpx.Response(status_code=200, json={"email": "test@google.com"})
        mock_proj_resp = httpx.Response(
            status_code=200, json={"cloudaicompanionProject": "aicode-consumers"}
        )

        def mock_httpx_post(url, *args, **kwargs):
            if "oauth2.googleapis.com/token" in url:
                return mock_token_resp
            if "loadCodeAssist" in url:
                return mock_proj_resp
            return httpx.Response(status_code=404)

        def mock_httpx_get(url, *args, **kwargs):
            if "userinfo" in url:
                return mock_userinfo_resp
            return httpx.Response(status_code=404)

        with patch.dict(os.environ, {"ANTI_API_DATA_DIR": tmpdir}):
            with patch("webbrowser.open", side_effect=fake_open):
                with patch("httpx.post", side_effect=mock_httpx_post):
                    with patch("httpx.get", side_effect=mock_httpx_get):
                        authenticate()

            assert os.path.exists(fake_account_file)
            with open(fake_account_file) as f:
                saved = json.load(f)
            assert saved["accounts"][0]["accessToken"] == "exchanged_tok"
            assert saved["accounts"][0]["projectId"] == "aicode-consumers"
