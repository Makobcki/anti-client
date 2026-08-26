from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import queue
import secrets
import threading
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union, overload

import httpx

from .exceptions import (
    AgentAPIError,
    AuthError,
    ModelNotFoundError,
    RateLimitError,
)
from .types import (
    ChatResponse,
    Message,
    ModelInfo,
    QuotaInfo,
    StreamChunk,
    Tool,
    ToolCall,
    UsageStats,
)

logger = logging.getLogger(__name__)

API_ENDPOINTS = {
    "production": "https://daily-cloudcode-pa.googleapis.com",
    "prod_fallback": "https://cloudcode-pa.googleapis.com",
    "sandbox": "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "autopush": "https://autopush-cloudcode-pa.sandbox.googleapis.com",
}

DEFAULT_PROJECT_ID = os.environ.get("ANTI_PROJECT_ID", "rising-fact-p41fc")

DEFAULT_CLIENT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Antigravity/1.19.6 Chrome/138.0.7204.235 Electron/37.3.1 Safari/537.36"
    ),
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": json.dumps(
        {"ideType": "ANTIGRAVITY", "platform": "WINDOWS", "pluginType": "GEMINI"}
    ),
}


def sanitize_params_for_google(schema: Any) -> Any:
    """Recursively removes Draft-07 JSON Schema fields unsupported by Google Cloud Code API.

    Fields like `patternProperties`, `additionalProperties`, `$schema`, `allOf`,
    `anyOf`, `definitions`, and `$defs` cause HTTP 400 Bad Request on the Google backend.
    """
    if isinstance(schema, dict):
        unsupported_keys = {
            "patternProperties",
            "additionalProperties",
            "$schema",
            "allOf",
            "anyOf",
            "definitions",
            "$defs",
        }
        return {
            k: sanitize_params_for_google(v)
            for k, v in schema.items()
            if k not in unsupported_keys
        }
    elif isinstance(schema, list):
        return [sanitize_params_for_google(item) for item in schema]
    return schema


def _extract_project_id(data: Any) -> Optional[str]:
    """Extracts and normalizes the project ID from an API response.

    Handles string IDs, dictionary structures, resource paths (e.g. 'projects/123'),
    and ignores empty dictionaries.
    """
    if not isinstance(data, dict):
        return None
    for key in ("cloudaicompanionProject", "projectId", "project"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().removeprefix("projects/")
        if isinstance(val, dict):
            proj_id = val.get("id") or val.get("projectId") or val.get("name")
            if isinstance(proj_id, str) and proj_id.strip():
                return proj_id.strip().removeprefix("projects/")
    return None


OAUTH_CONFIG = {
    "client_id": os.environ.get(
        "ANTI_CLIENT_ID",
        "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com",
    ),
    "client_secret": os.environ.get(
        "ANTI_CLIENT_SECRET", "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"
    ),
    "callback_port": 51121,
    "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_url": "https://oauth2.googleapis.com/token",
    "project_url": f"{API_ENDPOINTS['production']}/v1internal:loadCodeAssist",
    "scopes": [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    ],
}


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handles the local HTTP callback for OAuth 2.0 authorization."""

    def do_GET(self):
        """Processes the GET request from the OAuth callback.
        Extracts the authorization code or error from the URL query parameters
        and pushes it to the server's result queue.
        """
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path == "/oauth-callback":
            query_params = urllib.parse.parse_qs(parsed_path.query)

            # Передаем результат через очередь, привязанную к экземпляру сервера
            if "error" in query_params:
                self.server.oauth_result_queue.put({"error": query_params["error"][0]})
            elif "code" in query_params:
                self.server.oauth_result_queue.put({"code": query_params["code"][0]})
            else:
                self.server.oauth_result_queue.put(
                    {"error": "No valid parameters received."}
                )

            self.send_response(302)
            self.send_header("Location", "https://antigravity.google/auth-success")
            self.end_headers()
            threading.Thread(target=self.server.shutdown).start()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        """Suppresses standard HTTP server log messages."""
        pass


def authenticate():
    """Authenticates the user via Google OAuth 2.0 with PKCE.
    Starts a local server, opens the user's browser for authorization,
    waits for the OAuth callback, exchanges the authorization code for access
    and refresh tokens, retrieves the user's project ID, and saves the
    credentials to `~/.anti-api/accounts.json`.
    """
    state = secrets.token_urlsafe(16)
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("ascii")).digest())
        .decode("ascii")
        .rstrip("=")
    )

    result_queue = queue.Queue()
    server_ready = threading.Event()
    server_port_box = []

    def start_server():
        port = OAUTH_CONFIG["callback_port"]
        max_offset = 10
        httpd = None

        for offset in range(max_offset + 1):
            try:
                httpd = HTTPServer(("127.0.0.1", port + offset), OAuthCallbackHandler)
                break
            except OSError:
                continue

        if not httpd:
            result_queue.put({"error": "Could not bind to any port for OAuth callback"})
            server_ready.set()
            return

        httpd.oauth_result_queue = result_queue
        server_port_box.append(httpd.server_port)

        server_ready.set()
        httpd.serve_forever()

    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    server_ready.wait()

    if not server_port_box:
        print("Failed to start the local HTTP server.")
        return

    actual_port = server_port_box[0]
    redirect_uri = f"http://localhost:{actual_port}/oauth-callback"

    params = {
        "client_id": OAUTH_CONFIG["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(OAUTH_CONFIG["scopes"]),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    auth_url = f"{OAUTH_CONFIG['auth_url']}?{urllib.parse.urlencode(params)}"
    print(
        f"Opening browser for authorization...\nIf the browser does not open automatically, please visit:\n{auth_url}\n"
    )
    webbrowser.open(auth_url)

    try:
        result = result_queue.get(timeout=300)
    except queue.Empty:
        print("Authorization timed out.")
        return

    auth_error = result.get("error")
    auth_code = result.get("code")

    if auth_error:
        print(f"Authorization error: {auth_error}")
        return

    if not auth_code:
        print("Authorization was not completed.")
        return

    print("Authorization code received. Exchanging for tokens...")

    token_data = {
        "code": auth_code,
        "client_id": OAUTH_CONFIG["client_id"],
        "client_secret": OAUTH_CONFIG["client_secret"],
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "code_verifier": code_verifier,
    }

    response = httpx.post(OAUTH_CONFIG["token_url"], data=token_data, timeout=60.0)
    if response.status_code != 200:
        print(f"Token exchange error: {response.status_code}\n{response.text}")
        return

    tokens = response.json()
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    expires_in = tokens.get("expires_in", 3600)

    print("Retrieving user profile...")
    user_email = None
    try:
        userinfo_resp = httpx.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            timeout=15.0,
        )
        if userinfo_resp.status_code == 200:
            user_data = userinfo_resp.json()
            user_email = user_data.get("email")
            if user_email:
                print(f"Logged in as: {user_email}")
    except Exception:
        pass

    print("Retrieving Project ID (loadCodeAssist)...")
    project_id = None
    try:
        project_response = httpx.post(
            OAUTH_CONFIG["project_url"],
            headers={
                **DEFAULT_CLIENT_HEADERS,
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"metadata": {"ideType": "ANTIGRAVITY"}},
            timeout=60.0,
        )
        if project_response.status_code == 200:
            project_data = project_response.json()
            project_id = _extract_project_id(project_data)
    except Exception:
        pass

    # Auto-onboarding if companion project is not assigned
    if not project_id:
        print("Project not assigned, performing auto-onboarding (onboardUser)...")
        try:
            onboard_url = f"{API_ENDPOINTS['production']}/v1internal:onboardUser"
            for tier in ("free-tier", "standard-tier"):
                onboard_resp = httpx.post(
                    onboard_url,
                    headers={
                        **DEFAULT_CLIENT_HEADERS,
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "tierId": tier,
                        "metadata": {"ideType": "ANTIGRAVITY"},
                    },
                    timeout=60.0,
                )
                if onboard_resp.status_code == 200:
                    onboard_data = onboard_resp.json()
                    project_id = _extract_project_id(
                        onboard_data.get("response")
                    ) or _extract_project_id(onboard_data)
                    if project_id:
                        break
        except Exception:
            pass

    if not project_id or project_id == "unknown":
        project_id = DEFAULT_PROJECT_ID

    print(f"Active Project ID: {project_id}")

    data_dir = os.environ.get("ANTI_API_DATA_DIR")
    if not data_dir:
        home = (
            os.environ.get("HOME")
            or os.environ.get("USERPROFILE")
            or os.path.expanduser("~")
        )
        data_dir = os.path.join(home, ".anti-api")

    os.makedirs(data_dir, exist_ok=True)

    import time

    expires_at = time.time() + expires_in
    accounts_file = os.path.join(data_dir, "accounts.json")

    account_info = {
        "accessToken": access_token,
        "projectId": project_id,
        "expiresAt": expires_at,
    }
    if refresh_token:
        account_info["refreshToken"] = refresh_token
    if user_email:
        account_info["email"] = user_email

    with open(accounts_file, "w", encoding="utf-8") as f:
        json.dump({"accounts": [account_info]}, f, indent=2)

    print("Authorization completed and saved successfully!")


class ModelsResource:
    """Provides methods for querying available AI models and their quotas."""

    def __init__(self, client: Client):
        """Initializes the ModelsResource.
        Args:
            client (Client): The main client instance used for authentication.
        """
        self._client = client

    async def list(self) -> List[ModelInfo]:
        """Fetches a list of all available models asynchronously.
        Returns:
            List[ModelInfo]: A list of available models and their metadata.
        Raises:
            AgentAPIError: If the API request to fetch models fails.
        """
        await self._client.fetch_project_id()
        url = f"{self._client.base_url}/v1internal:fetchAvailableModels"
        await self._client._check_and_refresh_token()
        headers = {
            **DEFAULT_CLIENT_HEADERS,
            "Authorization": f"Bearer {self._client.api_key}",
            "Content-Type": "application/json",
        }
        data = {"project": self._client.project_id}
        response = await self._client.http_client.post(url, json=data, headers=headers)
        if response.status_code != 200:
            Client._raise_for_status(response.status_code, response.text)
        response_data = response.json()
        models_dict = response_data.get("models", {})
        parsed_models = []
        for name, info in models_dict.items():
            quota_raw = info.get("quotaInfo", {})
            quota_info = (
                QuotaInfo(
                    remaining_fraction=quota_raw.get("remainingFraction", 0.0),
                    reset_time=quota_raw.get("resetTime"),
                )
                if quota_raw
                else None
            )
            parsed_models.append(
                ModelInfo(
                    id=name,
                    internal_model_id=info.get("model", ""),
                    display_name=info.get("displayName", name),
                    api_provider=info.get("apiProvider", ""),
                    model_provider=info.get("modelProvider", ""),
                    max_tokens=info.get("maxTokens", 0),
                    max_output_tokens=None,
                    is_internal=info.get("isInternal", False),
                    supports_images=info.get("supportsImages", False),
                    supports_thinking=info.get("supportsThinking", False),
                    quota_info=quota_info,
                )
            )
        return parsed_models

    async def get(self) -> Dict[str, QuotaInfo]:
        """Gets a mapping of model IDs to their quota information asynchronously.
        Returns:
            Dict[str, QuotaInfo]: A dictionary mapping model IDs to their quotas.
        """
        models = await self.list()
        return {m.id: m.quota_info for m in models if m.quota_info}


class Client:
    """The main async client for interacting with the Cloud Code (Gemini) API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initializes the client.
        If `api_key` or `project_id` are not provided, it will attempt to load
        them from the local cache file or environment variables.
        Args:
            api_key (Optional[str], optional): The OAuth access token. Defaults to None.
            project_id (Optional[str], optional): The Google Cloud project ID. Defaults to None.
            base_url (Optional[str], optional): The API base URL. Defaults to Production.
        Raises:
            AuthError: If authentication tokens or the project ID are missing.
        """
        self.base_url = (
            base_url
            or os.environ.get("ANTI_API_BASE_URL")
            or API_ENDPOINTS["production"]
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("MY_API_KEY")
        self.project_id = project_id
        self.refresh_token = None
        self.expires_at = None
        self.http_client = httpx.AsyncClient(timeout=60.0)
        info = self._load_account_info()
        if not self.api_key:
            self.api_key = info.get("accessToken")
        if not self.project_id:
            self.project_id = (
                os.environ.get("ANTI_PROJECT_ID")
                or info.get("projectId")
                or DEFAULT_PROJECT_ID
            )
        if not self.project_id or self.project_id == "unknown":
            self.project_id = DEFAULT_PROJECT_ID
        self.refresh_token = info.get("refreshToken")
        self.expires_at = info.get("expiresAt")
        if not self.api_key or self.api_key == "YOUR_ACCESS_TOKEN":
            raise AuthError("API key not provided. Run anti_client.authenticate()")
        self.models = ModelsResource(self)

    async def close(self) -> None:
        """Closes the underlying HTTP client session."""
        await self.http_client.aclose()

    async def get_user_info(self) -> dict:
        """Retrieves user profile information (email, name, picture) using the current OAuth token.

        Returns:
            dict: User information returned by the Google OAuth2 userinfo endpoint.
        Raises:
            AgentAPIError: If the request fails.
        """
        await self._check_and_refresh_token()
        url = "https://www.googleapis.com/oauth2/v2/userinfo"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        response = await self.http_client.get(url, headers=headers)
        if response.status_code != 200:
            self._raise_for_status(response.status_code, response.text)
        return response.json()

    async def onboard_user(self, tier_id: str = "free-tier") -> dict:
        """Onboards the user and provisions a companion project if not already allocated.

        Args:
            tier_id (str): The tier ID to enroll in (default: 'free-tier').
        Returns:
            dict: The response from the onboardUser endpoint.
        """
        await self._check_and_refresh_token()
        url = f"{self.base_url}/v1internal:onboardUser"
        headers = {
            **DEFAULT_CLIENT_HEADERS,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "tierId": tier_id,
            "metadata": {
                "ideType": "ANTIGRAVITY",
            },
        }
        response = await self.http_client.post(url, json=data, headers=headers)
        if response.status_code != 200:
            self._raise_for_status(response.status_code, response.text)
        resp_json = response.json()
        new_project_id = _extract_project_id(
            resp_json.get("response")
        ) or _extract_project_id(resp_json)
        if new_project_id:
            self.project_id = new_project_id
            self._save_account_info()
        return resp_json

    async def load_code_assist(self) -> dict:
        """Calls loadCodeAssist to retrieve account tier and project info.

        Returns:
            dict: The response JSON from loadCodeAssist.
        """
        await self._check_and_refresh_token()
        url = f"{self.base_url}/v1internal:loadCodeAssist"
        headers = {
            **DEFAULT_CLIENT_HEADERS,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"metadata": {"ideType": "ANTIGRAVITY"}}
        response = await self.http_client.post(url, json=data, headers=headers)
        if response.status_code != 200:
            self._raise_for_status(response.status_code, response.text)
        return response.json()

    async def fetch_project_id(self, force: bool = False) -> str:
        """Retrieves and updates the user's active companion project ID via loadCodeAssist.

        Args:
            force (bool): If True, re-fetches the project ID even if already set.
        Returns:
            str: The active project ID.
        """
        if self.project_id and self.project_id != DEFAULT_PROJECT_ID and not force:
            return self.project_id

        try:
            code_assist_data = await self.load_code_assist()
            proj_id = _extract_project_id(code_assist_data)
            if proj_id:
                self.project_id = proj_id
                self._save_account_info()
                return proj_id
        except Exception:
            pass
        return self.project_id

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @staticmethod
    def _raise_for_status(status_code: int, error_text: str) -> None:
        """Raises a specific custom exception based on the HTTP status code.

        Args:
            status_code (int): The HTTP status code returned by the API.
            error_text (str): The error response body text.

        Raises:
            AuthError: If status code is 401.
            ModelNotFoundError: If status code is 404.
            RateLimitError: If status code is 429.
            AgentAPIError: For any other non-200 status code.
        """
        if status_code == 200:
            return
        if status_code == 401:
            raise AuthError(f"Unauthorized (401): {error_text}")
        if status_code == 404:
            raise ModelNotFoundError(f"Model or resource not found (404): {error_text}")
        if status_code == 429:
            raise RateLimitError(f"Rate limit exceeded (429): {error_text}")
        raise AgentAPIError(f"Error {status_code}: {error_text}")

    @property
    def _lock(self):
        """Returns the asyncio Lock, creating it if it doesn't exist yet.
        Returns:
            asyncio.Lock: The lock used to prevent race conditions during token refresh.
        """
        if not hasattr(self, "_async_lock"):
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _save_account_info(self) -> None:
        """Saves or updates account information in the local credentials file."""
        data_dir = os.environ.get("ANTI_API_DATA_DIR")
        if not data_dir:
            home = (
                os.environ.get("HOME")
                or os.environ.get("USERPROFILE")
                or os.path.expanduser("~")
            )
            data_dir = os.path.join(home, ".anti-api")
        accounts_file = os.path.join(data_dir, "accounts.json")
        account_info = self._load_account_info()
        account_info.update(
            {
                "accessToken": self.api_key,
                "refreshToken": self.refresh_token,
                "projectId": self.project_id,
                "expiresAt": self.expires_at,
            }
        )
        os.makedirs(data_dir, exist_ok=True)
        try:
            with open(accounts_file, "w", encoding="utf-8") as f:
                json.dump({"accounts": [account_info]}, f, indent=2)
        except Exception:
            pass

    async def _check_and_refresh_token(self):
        """Checks if the access token has expired and refreshes it if possible."""
        if not self.refresh_token:
            return
        import time

        if self.expires_at:
            exp = self.expires_at / 1000.0 if self.expires_at > 1e11 else self.expires_at
            if time.time() < (exp - 60):
                return
        async with self._lock:
            if self.expires_at:
                exp = self.expires_at / 1000.0 if self.expires_at > 1e11 else self.expires_at
                if time.time() < (exp - 60):
                    return
            url = OAUTH_CONFIG["token_url"]
            data = {
                "client_id": OAUTH_CONFIG["client_id"],
                "client_secret": OAUTH_CONFIG["client_secret"],
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token",
            }
            try:
                response = await self.http_client.post(url, data=data)
                if response.status_code == 200:
                    tokens = response.json()
                    self.api_key = tokens.get("access_token")
                    expires_in = tokens.get("expires_in", 3600)
                    self.expires_at = time.time() + expires_in
                    self._save_account_info()
            except Exception:
                pass

    def _load_account_info(self) -> dict:
        """Loads account information from the local credentials file.
        Returns:
            dict: The first account dictionary, or an empty dictionary if not found.
        """
        data_dir = os.environ.get("ANTI_API_DATA_DIR")
        if not data_dir:
            home = (
                os.environ.get("HOME")
                or os.environ.get("USERPROFILE")
                or os.path.expanduser("~")
            )
            data_dir = os.path.join(home, ".anti-api")
        accounts_file = os.path.join(data_dir, "accounts.json")
        if os.path.exists(accounts_file):
            try:
                with open(accounts_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    accounts = data.get("accounts", [])
                    if accounts:
                        return accounts[0]
            except Exception:
                pass
        return {}

    @staticmethod
    def _parse_part(part: dict) -> tuple[Optional[str], Optional[str]]:
        """Extracts text and thought content from a candidate content part.
        Args:
            part (dict): The dictionary representation of a content part.
        Returns:
            tuple[Optional[str], Optional[str]]: A tuple containing the text
                and the thought content, respectively. Either or both can be None.
        """
        is_thought = part.get("thought") is True or part.get("isThought") is True
        if is_thought:
            thought_text = part.get("text") or (
                part.get("thought") if isinstance(part.get("thought"), str) else None
            )
            return None, thought_text
        if (
            "thought" in part
            and isinstance(part["thought"], str)
            and "text" not in part
        ):
            return None, part["thought"]
        if "text" in part:
            return part["text"], None
        return None, None

    @overload
    async def generate(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.8,
        stream: Literal[False] = False,
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatResponse: ...

    @overload
    async def generate(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.8,
        stream: Literal[True] = True,
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]: ...

    async def generate(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.8,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> Union[ChatResponse, AsyncIterator[StreamChunk]]:
        """Generates a response from the model asynchronously.
        Args:
            model (str): The ID of the model to use for generation.
            messages (List[Message]): The conversation history.
            temperature (float, optional): The sampling temperature. Defaults to 0.8.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            tools (Optional[List[Tool]], optional): A list of tools available to the model. Defaults to None.
            **kwargs: Additional generation parameters.
        Returns:
            Union[ChatResponse, AsyncIterator[StreamChunk]]: A complete ChatResponse if stream
                is False, or an async iterator yielding StreamChunks if stream is True.
        Raises:
            AgentAPIError: If the generation API request fails.
        """
        await self.fetch_project_id()
        await self._check_and_refresh_token()
        import time

        url = f"{self.base_url}/v1internal:streamGenerateContent?alt=sse"
        contents = []
        system_instruction = None
        for raw_msg in messages:
            msg = Message(**raw_msg) if isinstance(raw_msg, dict) else raw_msg
            if msg.role == "system":
                system_instruction = {"role": "system", "parts": [{"text": msg.content}]}
                continue
            role = "model" if msg.role == "assistant" else "user"
            parts = []
            if getattr(msg, "attachments", None):
                for attachment in msg.attachments:
                    parts.append(
                        {
                            "inlineData": {
                                "mimeType": attachment.mime_type,
                                "data": attachment.data,
                            }
                        }
                    )
            if getattr(msg, "thought", None):
                parts.append({"text": msg.thought, "thought": True})
            if msg.content:
                parts.append({"text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    part_dict = {
                        "functionCall": {
                            "id": tc.id,
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                    }
                    if tc.thought_signature:
                        part_dict["thoughtSignature"] = tc.thought_signature
                    parts.append(part_dict)
            if msg.role == "tool":
                try:
                    resp_data = json.loads(msg.content)
                except Exception:
                    resp_data = msg.content
                parts = [
                    {
                        "functionResponse": {
                            "name": (
                                getattr(msg, "name", None)
                                or msg.tool_call_id
                                or "unknown_tool"
                            ),
                            "response": (
                                resp_data
                                if isinstance(resp_data, dict)
                                else {"output": resp_data}
                            ),
                        }
                    }
                ]
                role = "user"
            contents.append({"role": role, "parts": parts})

        if "claude" in model.lower():
            thinking_config = {
                "includeThoughts": True,
                "thinkingBudget": kwargs.get("thinking_budget", 4096),
            }
        else:
            thinking_config = {
                "includeThoughts": True,
                "thinkingLevel": kwargs.get("thinking_level", "high"),
            }

        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": kwargs.get("max_tokens") or kwargs.get("max_output_tokens", 8192),
        }
        if "stop_sequences" in kwargs:
            generation_config["stopSequences"] = kwargs["stop_sequences"]
        if thinking_config:
            generation_config["thinkingConfig"] = thinking_config

        safety_settings = kwargs.get(
            "safety_settings",
            [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )

        payload = {
            "project": self.project_id,
            "model": model,
            "requestType": "agent",
            "userAgent": "antigravity",
            "requestId": f"agent-{int(time.time() * 1000)}",
            "request": {
                "contents": contents,
                "generationConfig": generation_config,
                "safetySettings": safety_settings,
            },
        }
        if system_instruction:
            payload["request"]["systemInstruction"] = system_instruction
        if tools:
            func_decls = []
            for t in tools:
                func_decls.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": sanitize_params_for_google(t.parameters),
                    }
                )
            payload["request"]["tools"] = [{"functionDeclarations": func_decls}]
            payload["request"]["toolConfig"] = {
                "functionCallingConfig": {"mode": "AUTO"}
            }
        headers = {
            **DEFAULT_CLIENT_HEADERS,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }

        async def _do_stream() -> AsyncIterator[StreamChunk]:
            async with self.http_client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    self._raise_for_status(
                        response.status_code,
                        error_text.decode("utf-8", errors="replace"),
                    )
                async for chunk in self._stream_response(response):
                    yield chunk

        async def _do_sync() -> ChatResponse:
            response = await self.http_client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                self._raise_for_status(response.status_code, response.text)
            return self._sync_response(response)

        if stream:
            return _do_stream()
        else:
            return await _do_sync()

    def _sync_response(self, response: httpx.Response) -> ChatResponse:
        """Parses a synchronous (non-streaming) response from the API.
        Args:
            response (httpx.Response): The raw HTTP response object.
        Returns:
            ChatResponse: The fully parsed response containing text, thoughts, usage, and tool calls.
        """
        full_text = ""
        full_thought = ""
        tool_calls = []
        finish_reason = "stop"
        usage = UsageStats(0, 0, 0)

        for line_str in response.iter_lines():
            line_str = line_str.strip()
            if line_str:
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunks = json.loads(data_str)
                        if not isinstance(chunks, list):
                            chunks = [chunks]
                        for chunk in chunks:
                            resp = chunk.get("response", {})
                            if "usageMetadata" in resp:
                                meta = resp["usageMetadata"]
                                usage = UsageStats(
                                    prompt_tokens=meta.get("promptTokenCount", 0),
                                    completion_tokens=meta.get(
                                        "candidatesTokenCount", 0
                                    ),
                                    total_tokens=meta.get("totalTokenCount", 0),
                                )

                            candidates = resp.get("candidates", [])
                            for candidate in candidates:
                                if "finishReason" in candidate:
                                    finish_reason = candidate["finishReason"]
                                parts = candidate.get("content", {}).get("parts", [])
                                for part in parts:
                                    text_val, thought_val = self._parse_part(part)
                                    if thought_val:
                                        full_thought += thought_val
                                    if text_val:
                                        full_text += text_val
                                    if "functionCall" in part:
                                        fc = part["functionCall"]
                                        tool_calls.append(
                                            ToolCall(
                                                id=fc.get("id", fc.get("name")),
                                                name=fc.get("name"),
                                                arguments=fc.get("args", {}),
                                                thought_signature=part.get(
                                                    "thoughtSignature"
                                                ),
                                            )
                                        )
                    except json.JSONDecodeError:
                        logger.warning("Failed to decode chunk: %s", data_str)

        return ChatResponse(
            text=full_text if full_text else None,
            thought=full_thought if full_thought else None,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls if tool_calls else None,
        )

    async def _stream_response(
        self, response: httpx.Response
    ) -> AsyncIterator[StreamChunk]:
        """Parses an async streaming response from the API.
        Args:
            response (httpx.Response): The raw HTTP response object.
        Yields:
            StreamChunk: Chunks of generated text, thoughts, or a list of tool calls.
        """
        tool_calls = []

        async for line_str in response.aiter_lines():
            line_str = line_str.strip()
            if line_str:
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunks = json.loads(data_str)
                        if not isinstance(chunks, list):
                            chunks = [chunks]
                        for chunk in chunks:
                            candidates = chunk.get("response", {}).get("candidates", [])
                            for candidate in candidates:
                                parts = candidate.get("content", {}).get("parts", [])
                                for part in parts:
                                    text_val, thought_val = self._parse_part(part)
                                    if thought_val:
                                        yield StreamChunk(thought=thought_val)
                                    if text_val:
                                        yield StreamChunk(text=text_val)
                                    if "functionCall" in part:
                                        fc = part["functionCall"]
                                        tool_calls.append(
                                            ToolCall(
                                                id=fc.get("id", fc.get("name")),
                                                name=fc.get("name"),
                                                arguments=fc.get("args", {}),
                                                thought_signature=part.get(
                                                    "thoughtSignature"
                                                ),
                                            )
                                        )
                    except json.JSONDecodeError:
                        logger.warning("Failed to decode chunk: %s", data_str)

        if tool_calls:
            yield StreamChunk(tool_calls=tool_calls)
