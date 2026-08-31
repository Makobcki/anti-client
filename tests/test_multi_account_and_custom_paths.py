import json
import os
import tempfile
import time
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from anti_client.client import (
    Client,
    authenticate,
    get_account,
    list_accounts,
    logout,
    remove_account,
    resolve_accounts_path,
    save_account,
    set_active_account,
)
from anti_client.exceptions import AuthError


def test_manual_api_key_initialization():
    # Pass api_key directly without any local accounts.json file
    with tempfile.TemporaryDirectory() as empty_dir:
        with patch.dict(os.environ, {"ANTI_API_DATA_DIR": empty_dir, "HOME": empty_dir}, clear=False):
            client = Client(api_key="manual_secret_token_123")
            assert client.api_key == "manual_secret_token_123"
            assert client.refresh_token is None
            assert client.expires_at is None
            assert client.email is None
            assert client.project_id == "aicode-consumers"


def test_manual_api_key_with_custom_project_and_refresh_token():
    client = Client(
        api_key="tok_abc",
        refresh_token="ref_xyz",
        email="developer@company.org",
        project_id="custom-project-id",
        auto_save=False,
    )
    assert client.api_key == "tok_abc"
    assert client.refresh_token == "ref_xyz"
    assert client.email == "developer@company.org"
    assert client.project_id == "custom-project-id"


def test_credentials_dict_initialization():
    creds = {
        "accessToken": "dict_token",
        "refreshToken": "dict_refresh",
        "projectId": "dict_project",
        "email": "dict_user@example.com",
        "expiresAt": 1700000000.0,
    }
    client = Client(credentials=creds, auto_save=False)
    assert client.api_key == "dict_token"
    assert client.refresh_token == "dict_refresh"
    assert client.project_id == "dict_project"
    assert client.email == "dict_user@example.com"
    assert client.expires_at == 1700000000.0


def test_project_level_storage_discovery():
    with tempfile.TemporaryDirectory() as project_dir:
        dot_anti = os.path.join(project_dir, ".anti-api")
        os.makedirs(dot_anti, exist_ok=True)
        acc_file = os.path.join(dot_anti, "accounts.json")
        with open(acc_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accounts": [
                        {
                            "accessToken": "project_local_token",
                            "refreshToken": "project_local_refresh",
                            "email": "project@example.com",
                            "projectId": "local-project",
                        }
                    ]
                },
                f,
            )

        orig_cwd = os.getcwd()
        try:
            os.chdir(project_dir)
            with patch.dict(os.environ, {"ANTI_API_DATA_DIR": ""}):
                resolved = resolve_accounts_path(for_saving=False)
                assert resolved == os.path.abspath(acc_file)

                client = Client()
                assert client.api_key == "project_local_token"
                assert client.email == "project@example.com"
                assert client.project_id == "local-project"
        finally:
            os.chdir(orig_cwd)


def test_custom_credentials_path_and_env_var():
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_file = os.path.join(tmpdir, "custom_creds.json")
        with open(custom_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accounts": [
                        {
                            "accessToken": "custom_file_tok",
                            "email": "custom@example.com",
                            "projectId": "custom-proj",
                        }
                    ]
                },
                f,
            )

        # 1. Via credentials_path argument
        client = Client(credentials_path=custom_file)
        assert client.api_key == "custom_file_tok"
        assert client.email == "custom@example.com"

        # 2. Via ANTI_ACCOUNTS_PATH env var
        with patch.dict(os.environ, {"ANTI_ACCOUNTS_PATH": custom_file}):
            client2 = Client()
            assert client2.api_key == "custom_file_tok"


def test_multi_account_save_list_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        acc_file = os.path.join(tmpdir, "accounts.json")

        # Save account 1
        save_account(
            {
                "accessToken": "acc1_tok",
                "refreshToken": "acc1_ref",
                "email": "alice@gmail.com",
                "projectId": "proj-alice",
            },
            credentials_path=acc_file,
            make_active=True,
        )

        # Save account 2
        save_account(
            {
                "accessToken": "acc2_tok",
                "refreshToken": "acc2_ref",
                "email": "bob@gmail.com",
                "projectId": "proj-bob",
            },
            credentials_path=acc_file,
            make_active=False,
        )

        accounts = list_accounts(credentials_path=acc_file)
        assert len(accounts) == 2
        assert accounts[0]["email"] == "alice@gmail.com"
        assert accounts[1]["email"] == "bob@gmail.com"

        # get_account by email
        acc_bob = get_account(email="bob@gmail.com", credentials_path=acc_file)
        assert acc_bob is not None
        assert acc_bob["accessToken"] == "acc2_tok"

        # get_account by index
        acc_0 = get_account(index=0, credentials_path=acc_file)
        assert acc_0 is not None
        assert acc_0["email"] == "alice@gmail.com"


def test_multi_account_client_selection_and_errors():
    with tempfile.TemporaryDirectory() as tmpdir:
        acc_file = os.path.join(tmpdir, "accounts.json")
        save_account(
            {
                "accessToken": "alice_tok",
                "refreshToken": "alice_ref",
                "email": "alice@gmail.com",
                "projectId": "proj-alice",
            },
            credentials_path=acc_file,
        )
        save_account(
            {
                "accessToken": "bob_tok",
                "refreshToken": "bob_ref",
                "email": "bob@gmail.com",
                "projectId": "proj-bob",
            },
            credentials_path=acc_file,
        )

        # Select Bob by email
        client_bob = Client(account_email="bob@gmail.com", credentials_path=acc_file)
        assert client_bob.email == "bob@gmail.com"
        assert client_bob.api_key == "bob_tok"
        assert client_bob.project_id == "proj-bob"

        # Select Alice by index
        client_alice = Client(account_index=0, credentials_path=acc_file)
        assert client_alice.email == "alice@gmail.com"
        assert client_alice.api_key == "alice_tok"

        # Select via ANTI_ACCOUNT_EMAIL env var
        with patch.dict(os.environ, {"ANTI_ACCOUNT_EMAIL": "bob@gmail.com"}):
            client_env = Client(credentials_path=acc_file)
            assert client_env.email == "bob@gmail.com"

        # Non-existent email raises AuthError
        with pytest.raises(AuthError, match="not found"):
            Client(account_email="nonexistent@gmail.com", credentials_path=acc_file)

        # Out-of-range index raises AuthError
        with pytest.raises(AuthError, match="out of range"):
            Client(account_index=99, credentials_path=acc_file)


@pytest.mark.asyncio
async def test_token_refresh_preserves_email_and_other_accounts():
    with tempfile.TemporaryDirectory() as tmpdir:
        acc_file = os.path.join(tmpdir, "accounts.json")

        save_account(
            {
                "accessToken": "alice_old_tok",
                "refreshToken": "alice_ref",
                "email": "alice@gmail.com",
                "projectId": "proj-alice",
            },
            credentials_path=acc_file,
        )
        save_account(
            {
                "accessToken": "bob_old_tok",
                "refreshToken": "bob_ref",
                "email": "bob@gmail.com",
                "projectId": "proj-bob",
            },
            credentials_path=acc_file,
        )

        # Load bob
        client_bob = Client(account_email="bob@gmail.com", credentials_path=acc_file)
        assert client_bob.email == "bob@gmail.com"
        client_bob.expires_at = 0  # Force expiry

        mock_refresh_resp = httpx.Response(
            status_code=200,
            json={"access_token": "bob_fresh_token", "expires_in": 3600},
        )

        with patch.object(client_bob.http_client, "post", new_callable=AsyncMock, return_value=mock_refresh_resp):
            await client_bob._check_and_refresh_token()

        # In memory check
        assert client_bob.api_key == "bob_fresh_token"
        assert client_bob.email == "bob@gmail.com"

        # On disk check
        accounts = list_accounts(credentials_path=acc_file)
        assert len(accounts) == 2

        # Alice is completely untouched
        alice = get_account(email="alice@gmail.com", credentials_path=acc_file)
        assert alice["accessToken"] == "alice_old_tok"
        assert alice["email"] == "alice@gmail.com"

        # Bob has fresh token and preserved email!
        bob = get_account(email="bob@gmail.com", credentials_path=acc_file)
        assert bob["accessToken"] == "bob_fresh_token"
        assert bob["email"] == "bob@gmail.com"
        assert bob["refreshToken"] == "bob_ref"
        assert bob["projectId"] == "proj-bob"


def test_remove_and_set_active_account():
    with tempfile.TemporaryDirectory() as tmpdir:
        acc_file = os.path.join(tmpdir, "accounts.json")
        save_account({"accessToken": "tok1", "email": "a@a.com"}, credentials_path=acc_file)
        save_account({"accessToken": "tok2", "email": "b@b.com"}, credentials_path=acc_file)

        assert set_active_account("b@b.com", credentials_path=acc_file) is True

        with open(acc_file) as f:
            data = json.load(f)
        assert data["activeAccount"] == "b@b.com"

        # Remove account
        assert remove_account(email="a@a.com", credentials_path=acc_file) is True
        accounts = list_accounts(credentials_path=acc_file)
        assert len(accounts) == 1
        assert accounts[0]["email"] == "b@b.com"


@pytest.mark.asyncio
async def test_get_user_info_updates_and_saves_email():
    with tempfile.TemporaryDirectory() as tmpdir:
        acc_file = os.path.join(tmpdir, "accounts.json")
        save_account(
            {"accessToken": "tok_no_email", "projectId": "proj-1"},
            credentials_path=acc_file,
        )

        client = Client(credentials_path=acc_file)
        assert client.email is None

        mock_userinfo = httpx.Response(status_code=200, json={"email": "discovered@gmail.com"})
        with patch.object(client.http_client, "get", new_callable=AsyncMock, return_value=mock_userinfo):
            info = await client.get_user_info()
            assert info["email"] == "discovered@gmail.com"
            assert client.email == "discovered@gmail.com"

        # Check saved on disk
        acc = get_account(index=0, credentials_path=acc_file)
        assert acc["email"] == "discovered@gmail.com"


def test_logout_functions():
    with tempfile.TemporaryDirectory() as tmpdir:
        acc_file = os.path.join(tmpdir, "accounts.json")
        save_account({"accessToken": "tok1", "email": "a@a.com"}, credentials_path=acc_file)
        save_account({"accessToken": "tok2", "email": "b@b.com"}, credentials_path=acc_file)

        # 1. Logout a single account by email (mocking revocation request)
        with patch("httpx.post") as mock_post:
            assert logout(email="a@a.com", credentials_path=acc_file, revoke=True) is True
            mock_post.assert_called_once()

        accounts = list_accounts(credentials_path=acc_file)
        assert len(accounts) == 1
        assert accounts[0]["email"] == "b@b.com"

        # 2. Logout all accounts
        with patch("httpx.post") as mock_post:
            assert logout(all_accounts=True, credentials_path=acc_file, revoke=True) is True

        accounts = list_accounts(credentials_path=acc_file)
        assert len(accounts) == 0


@pytest.mark.asyncio
async def test_client_logout_method():
    with tempfile.TemporaryDirectory() as tmpdir:
        acc_file = os.path.join(tmpdir, "accounts.json")
        save_account(
            {
                "accessToken": "client_tok",
                "refreshToken": "client_ref",
                "email": "client@gmail.com",
            },
            credentials_path=acc_file,
        )

        client = Client(credentials_path=acc_file)
        assert client.email == "client@gmail.com"

        with patch.object(client.http_client, "post", new_callable=AsyncMock) as mock_post:
            success = await client.logout(revoke=True)
            assert success is True
            mock_post.assert_called_once()

        assert client.api_key is None
        assert client.refresh_token is None
        assert list_accounts(credentials_path=acc_file) == []

