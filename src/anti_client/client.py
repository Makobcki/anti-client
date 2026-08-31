"""Asynchronous client and resource management for Google Cloud Code / Antigravity API."""

from __future__ import annotations

import asyncio
import base64
import builtins
import hashlib
import json
import logging
import os
import platform
import queue
import re
import secrets
import sys
import threading
import urllib.parse
import warnings
import webbrowser
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Literal, overload

import httpx

from .exceptions import (
    AgentAPIError,
    AuthError,
    ModelNotFoundError,
    RateLimitError,
)
from .types import (
    ChatResponse,
    Citation,
    CodeAssistInfo,
    CountTokensResult,
    FileAttachment,
    GenerateOptions,
    ImageGenerationResponse,
    Message,
    ModelInfo,
    ModelQuota,
    ModelsCatalog,
    QuotaBucket,
    QuotaGroup,
    QuotaSummary,
    SafetyRating,
    SearchResponse,
    SearchSource,
    StreamChunk,
    TierInfo,
    TokenDetail,
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

DEFAULT_PROJECT_ID = os.environ.get("ANTI_PROJECT_ID", "aicode-consumers")


def get_default_headers() -> dict[str, str]:
    """Generates dynamic client headers mimicking the official Antigravity CLI."""
    os_name = sys.platform
    if os_name.startswith("linux"):
        plat = "LINUX"
        ua_os = "linux"
    elif os_name.startswith("darwin"):
        plat = "DARWIN"
        ua_os = "darwin"
    elif os_name.startswith("win"):
        plat = "WINDOWS"
        ua_os = "windows"
    else:
        plat = "LINUX"
        ua_os = "linux"

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif "arm" in machine or "aarch64" in machine:
        arch = "arm64"
    else:
        arch = machine or "amd64"

    return {
        "User-Agent": f"Antigravity/1.19.6 ({ua_os}; {arch})",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": json.dumps(
            {"ideType": "ANTIGRAVITY", "platform": plat, "pluginType": "GEMINI"}
        ),
    }


DEFAULT_CLIENT_HEADERS = get_default_headers()


def get_clean_display_name(raw_name: str) -> str:
    """Removes effort/thinking suffixes like (High), (Low), (Medium), (Extra Low), (Thinking)."""
    return re.sub(
        r"\s*\((High|Low|Medium|Extra Low|Thinking)\)$",
        "",
        raw_name,
        flags=re.IGNORECASE,
    ).strip()


def parse_effort_from_suffix(model_id: str) -> str | None:
    """Parses thinking effort level from a model identifier suffix."""
    for suffix, level in [
        ("-extra-low", "extra-low"),
        ("-low", "low"),
        ("-medium", "medium"),
        ("-high", "high"),
    ]:
        if model_id.endswith(suffix):
            return level
    return None


def sanitize_params_for_google(schema: Any) -> Any:
    """Recursively removes Draft-07 JSON Schema fields unsupported by Google Cloud Code API.

    Fields like `patternProperties`, `additionalProperties`, `$schema`, `$id`, `$ref`,
    `allOf`, `anyOf`, `oneOf`, `definitions`, and `$defs` cause HTTP 400 Bad Request on the Google backend.
    """
    if isinstance(schema, dict):
        unsupported_keys = {
            "patternProperties",
            "additionalProperties",
            "$schema",
            "$id",
            "$ref",
            "allOf",
            "anyOf",
            "oneOf",
            "definitions",
            "$defs",
        }
        return {
            k: sanitize_params_for_google(v) for k, v in schema.items() if k not in unsupported_keys
        }
    elif isinstance(schema, list):
        return [sanitize_params_for_google(item) for item in schema]
    return schema


def _extract_project_id(data: Any) -> str | None:
    """Extracts and normalizes the project ID from an API response."""
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
    "client_secret": os.environ.get("ANTI_CLIENT_SECRET", "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"),
    "fallback_client_id": "884354919052-36trc1jjb3tguiac32ov6cod268c5blh.apps.googleusercontent.com",
    "fallback_client_secret": "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",
    "callback_port": 51121,
    "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_url": "https://oauth2.googleapis.com/token",
    "project_url": f"{API_ENDPOINTS['production']}/v1internal:loadCodeAssist",
    "scopes": [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/aicode",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    ],
}


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handles the local HTTP callback for OAuth 2.0 authorization."""

    def do_GET(self):
        """Processes the GET request from the OAuth callback."""
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path == "/oauth-callback":
            query_params = urllib.parse.parse_qs(parsed_path.query)

            if "error" in query_params:
                self.server.oauth_result_queue.put({"error": query_params["error"][0]})
            elif "code" in query_params:
                self.server.oauth_result_queue.put({"code": query_params["code"][0]})
            else:
                self.server.oauth_result_queue.put({"error": "No valid parameters received."})

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


def authenticate() -> None:
    """Authenticates the user via Google OAuth 2.0 with PKCE and stores credentials locally.

    Spawns a local HTTP server callback, opens the default web browser to the Google OAuth consent
    screen, exchanges the authorization code for access and refresh tokens, retrieves the user's
    companion Cloud project ID via loadCodeAssist, and saves credentials to ~/.anti-api/accounts.json.
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
        # Fallback to secondary credentials if primary failed
        token_data["client_id"] = OAUTH_CONFIG["fallback_client_id"]
        token_data["client_secret"] = OAUTH_CONFIG["fallback_client_secret"]
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
        home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or os.path.expanduser("~")
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
    """Provides methods for querying available AI models, quotas, and specialized category lists."""

    def __init__(self, client: Client):
        self._client = client
        self._cached_catalog: ModelsCatalog | None = None

    async def get_catalog(self, force: bool = False) -> ModelsCatalog:
        """Fetches and parses the full catalog of available models and category listings."""
        if self._cached_catalog and not force:
            return self._cached_catalog

        await self._client.fetch_project_id()
        url = f"{self._client.base_url}/v1internal:fetchAvailableModels"
        await self._client._check_and_refresh_token()
        headers = {
            **DEFAULT_CLIENT_HEADERS,
            "Authorization": f"Bearer {self._client.api_key}",
            "Content-Type": "application/json",
        }
        data = {"project": self._client.project_id}
        response = await self._client._post_with_fallback(url, json_data=data, headers=headers)
        if response.status_code != 200:
            Client._raise_for_status(response.status_code, response.text)

        response_data = response.json()
        models_dict = response_data.get("models", {})
        parsed_models: list[ModelInfo] = []

        for name, info in models_dict.items():
            quota_raw = info.get("quotaInfo", {})
            quota_info = (
                ModelQuota(
                    remaining_fraction=quota_raw.get("remainingFraction", 0.0),
                    reset_time=quota_raw.get("resetTime"),
                )
                if quota_raw
                else None
            )

            supports_thinking = bool(info.get("supportsThinking", False))
            effort_suffix = parse_effort_from_suffix(name)

            supported_efforts: list[str] | None = None
            if not supports_thinking:
                supported_efforts = []

            raw_display_name = info.get("displayName", name)
            clean_display_name = get_clean_display_name(raw_display_name)

            parsed_models.append(
                ModelInfo(
                    id=name,
                    internal_model_id=info.get("model", ""),
                    display_name=raw_display_name,
                    clean_display_name=clean_display_name,
                    api_provider=info.get("apiProvider", ""),
                    model_provider=info.get("modelProvider", ""),
                    max_tokens=info.get("maxTokens", 0),
                    max_output_tokens=info.get("maxOutputTokens"),
                    is_internal=bool(info.get("isInternal", False)),
                    supports_images=bool(info.get("supportsImages", False)),
                    supports_video=bool(info.get("supportsVideo", False)),
                    supports_thinking=supports_thinking,
                    supported_efforts=supported_efforts,
                    thinking_level=effort_suffix,
                    thinking_budget=info.get("thinkingBudget"),
                    min_thinking_budget=info.get("minThinkingBudget"),
                    quota_info=quota_info,
                    recommended=bool(info.get("recommended", False)),
                    tag_title=info.get("tagTitle"),
                    tag_description=info.get("tagDescription"),
                    supported_mime_types=info.get("supportedMimeTypes"),
                    model_experiments=info.get("modelExperiments"),
                )
            )

        # Parse deprecated model IDs and mapping
        raw_deprecated = response_data.get("deprecatedModelIds", [])
        deprecated_ids: list[str] = []
        deprecated_map: dict[str, str] = {}
        if isinstance(raw_deprecated, list):
            deprecated_ids = raw_deprecated
        elif isinstance(raw_deprecated, dict):
            deprecated_ids = list(raw_deprecated.keys())
            deprecated_map = raw_deprecated

        catalog = ModelsCatalog(
            models=parsed_models,
            default_agent_model_id=response_data.get("defaultAgentModelId", "gemini-3.1-pro-low"),
            agent_model_sorts=response_data.get("agentModelSorts", []),
            deprecated_model_ids=deprecated_ids,
            deprecated_model_map=deprecated_map,
            web_search_model_ids=response_data.get("webSearchModelIds", []),
            image_generation_model_ids=response_data.get("imageGenerationModelIds", []),
            command_model_ids=response_data.get("commandModelIds", []),
            mquery_model_ids=response_data.get("mqueryModelIds", []),
            commit_message_model_ids=response_data.get("commitMessageModelIds", []),
            audio_transcription_model_ids=response_data.get("audioTranscriptionModelIds", []),
            tab_model_ids=response_data.get("tabModelIds", []),
            tiered_model_ids=response_data.get("tieredModelIds"),
        )
        self._cached_catalog = catalog
        return catalog

    async def list(self, force: bool = False) -> builtins.list[ModelInfo]:
        """Fetches a list of all available models asynchronously.

        Args:
            force (bool, optional): If True, ignores cached models and queries backend. Defaults to False.

        Returns:
            List[ModelInfo]: The list of discovered model metadata objects.
        """
        catalog = await self.get_catalog(force=force)
        return catalog.models

    async def get(self) -> dict[str, ModelQuota | None]:
        """Gets a mapping of model IDs to their quota information asynchronously.

        Returns:
            Dict[str, Optional[ModelQuota]]: Mapping from model IDs to their remaining quota info.
        """
        catalog = await self.get_catalog()
        return {m.id: m.quota_info for m in catalog.models}


class QuotaResource:
    """Provides methods for querying detailed user quota summaries."""

    def __init__(self, client: Client):
        self._client = client

    async def get_summary(self) -> QuotaSummary:
        """Retrieves user quota status broken down by model groups (Gemini vs Claude/GPT) and windows (weekly vs 5h).

        Returns:
            QuotaSummary: Parsed user quota status broken down into Gemini and Claude groups.

        Raises:
            AuthError: If authentication fails.
            RateLimitError: If quota check is rate-limited.
            AgentAPIError: If the server returns an unexpected error.
        """
        await self._client.fetch_project_id()
        await self._client._check_and_refresh_token()
        url = f"{self._client.base_url}/v1internal:retrieveUserQuotaSummary"
        headers = {
            **DEFAULT_CLIENT_HEADERS,
            "Authorization": f"Bearer {self._client.api_key}",
            "Content-Type": "application/json",
        }
        data = {"project": self._client.project_id}
        response = await self._client._post_with_fallback(url, json_data=data, headers=headers)
        if response.status_code != 200:
            Client._raise_for_status(response.status_code, response.text)

        resp_data = response.json()
        raw_groups = resp_data.get("groups", [])
        groups: list[QuotaGroup] = []

        for g in raw_groups:
            buckets: list[QuotaBucket] = []
            for b in g.get("buckets", []):
                buckets.append(
                    QuotaBucket(
                        bucket_id=b.get("bucketId", ""),
                        display_name=b.get("displayName", ""),
                        window=b.get("window", ""),
                        reset_time=b.get("resetTime", ""),
                        description=b.get("description", ""),
                        remaining_fraction=float(b.get("remainingFraction", 0.0)),
                    )
                )
            groups.append(
                QuotaGroup(
                    display_name=g.get("displayName", ""),
                    description=g.get("description", ""),
                    buckets=buckets,
                )
            )

        return QuotaSummary(
            groups=groups,
            description=resp_data.get("description", ""),
        )


class Client:
    """The main async client for interacting with the Cloud Code / Antigravity API."""

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        base_url: str | None = None,
    ):
        self.base_url = (
            base_url or os.environ.get("ANTI_API_BASE_URL") or API_ENDPOINTS["production"]
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
                os.environ.get("ANTI_PROJECT_ID") or info.get("projectId") or DEFAULT_PROJECT_ID
            )
        if not self.project_id or self.project_id == "unknown":
            self.project_id = DEFAULT_PROJECT_ID
        self.refresh_token = info.get("refreshToken")
        self.expires_at = info.get("expiresAt")
        if not self.api_key or self.api_key == "YOUR_ACCESS_TOKEN":
            raise AuthError("API key not provided. Run anti_client.authenticate()")

        self.models = ModelsResource(self)
        self.quota = QuotaResource(self)

    async def close(self) -> None:
        """Closes the underlying HTTP client session."""
        await self.http_client.aclose()

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def get_user_info(self) -> dict:
        """Retrieves user profile information (email, name, picture) using the OAuth token.

        Returns:
            dict: Raw profile data returned by Google OAuth userinfo endpoint.

        Raises:
            AuthError: If the user token is invalid or missing required scopes.
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
            tier_id (str, optional): The target subscription tier. Defaults to "free-tier".

        Returns:
            dict: The raw JSON onboarding response from the backend.

        Raises:
            AuthError: If authentication fails.
            AgentAPIError: If onboarding request is rejected.
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
            "metadata": {"ideType": "ANTIGRAVITY"},
        }
        response = await self._post_with_fallback(url, json_data=data, headers=headers)
        if response.status_code != 200:
            self._raise_for_status(response.status_code, response.text)
        resp_json = response.json()
        new_project_id = _extract_project_id(resp_json.get("response")) or _extract_project_id(
            resp_json
        )
        if new_project_id:
            self.project_id = new_project_id
            self._save_account_info()
        return resp_json

    async def load_code_assist(self) -> CodeAssistInfo:
        """Calls loadCodeAssist to retrieve account tier and project info.

        Returns:
            CodeAssistInfo: Structured subscription tier and companion project status.

        Raises:
            AuthError: If unauthenticated.
            AgentAPIError: If request fails.
        """
        await self._check_and_refresh_token()
        url = f"{self.base_url}/v1internal:loadCodeAssist"
        headers = {
            **DEFAULT_CLIENT_HEADERS,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"metadata": {"ideType": "ANTIGRAVITY"}}
        response = await self._post_with_fallback(url, json_data=data, headers=headers)
        if response.status_code != 200:
            self._raise_for_status(response.status_code, response.text)

        raw = response.json()

        def _parse_tier(t: dict) -> TierInfo:
            return TierInfo(
                id=t.get("id", ""),
                name=t.get("name", ""),
                description=t.get("description", ""),
                upgrade_subscription_uri=t.get("upgradeSubscriptionUri"),
                upgrade_subscription_text=t.get("upgradeSubscriptionText"),
                is_default=bool(t.get("isDefault", False)),
                user_defined_companion_project=bool(
                    t.get("userDefinedCloudaicompanionProject", False)
                ),
                uses_gcp_tos=bool(t.get("usesGcpTos", False)),
            )

        current_tier = _parse_tier(raw["currentTier"]) if "currentTier" in raw else None
        allowed_tiers = [_parse_tier(t) for t in raw.get("allowedTiers", [])]
        proj_id = _extract_project_id(raw) or self.project_id

        return CodeAssistInfo(
            current_tier=current_tier,
            allowed_tiers=allowed_tiers,
            companion_project_id=proj_id,
            gcp_managed=bool(raw.get("gcpManaged", False)),
            paid_tier=raw.get("paidTier"),
            upgrade_subscription_uri=raw.get("upgradeSubscriptionUri"),
        )

    async def fetch_project_id(self, force: bool = False) -> str:
        """Retrieves and updates the user's active companion project ID via loadCodeAssist.

        Args:
            force (bool, optional): If True, re-fetches project ID even if already set. Defaults to False.

        Returns:
            str: The active companion project ID.
        """
        if self.project_id and self.project_id != DEFAULT_PROJECT_ID and not force:
            return self.project_id

        try:
            code_assist_info = await self.load_code_assist()
            if code_assist_info.companion_project_id:
                self.project_id = code_assist_info.companion_project_id
                self._save_account_info()
                return self.project_id
        except Exception:
            pass
        return self.project_id

    async def get_quota_summary(self) -> QuotaSummary:
        """Convenience method to retrieve user quota status across Gemini and Claude model families.

        Returns:
            QuotaSummary: Parsed user quota status broken down into Gemini and Claude groups.
        """
        return await self.quota.get_summary()

    async def count_tokens(
        self,
        messages: list[Message] | list[dict] | Message | str,
        model: str | None = None,
        tools: list[Tool | dict[str, Any]] | None = None,
        system_instruction: str | None = None,
    ) -> CountTokensResult:
        """Counts context tokens for given messages, tools, and system instructions via POST /v1internal:countTokens.

        Args:
            messages (Union[List[Message], List[dict], Message, str]): Context prompt or history.
            model (Optional[str], optional): Model ID to evaluate token rules against. Defaults to None.
            tools (Optional[List[Union[Tool, Dict[str, Any]]]], optional): Function schemas to include. Defaults to None.
            system_instruction (Optional[str], optional): System prompt text. Defaults to None.

        Returns:
            CountTokensResult: Result object containing the exact token count with integer operation support.

        Raises:
            AuthError: If unauthenticated.
            RateLimitError: If rate limit is exceeded.
            AgentAPIError: If counting fails on backend.
        """
        await self.fetch_project_id()
        await self._check_and_refresh_token()
        url = f"{self.base_url}/v1internal:countTokens"
        headers = {
            **DEFAULT_CLIENT_HEADERS,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Normalize messages into list
        msg_list: list[Any]
        if isinstance(messages, str):
            msg_list = [Message(role="user", content=messages)]
        elif isinstance(messages, Message):
            msg_list = [messages]
        elif isinstance(messages, list):
            msg_list = messages
        else:
            msg_list = [messages]

        contents = []
        extracted_sys_instruction = system_instruction

        for m in msg_list:
            if isinstance(m, str):
                contents.append({"role": "user", "parts": [{"text": m}]})
            elif isinstance(m, dict):
                contents.append(m)
            elif isinstance(m, Message):
                if m.role == "system":
                    if not extracted_sys_instruction and m.content:
                        extracted_sys_instruction = m.content
                    continue

                parts = []
                if m.thought:
                    parts.append({"thought": m.thought})
                if m.content:
                    parts.append({"text": m.content})
                if m.attachments:
                    for att in m.attachments:
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": att.mime_type,
                                    "data": att.data,
                                }
                            }
                        )
                if m.tool_calls:
                    for tc in m.tool_calls:
                        parts.append(
                            {
                                "functionCall": {
                                    "name": tc.name,
                                    "args": tc.arguments,
                                }
                            }
                        )
                if m.role == "tool":
                    parts.append(
                        {
                            "functionResponse": {
                                "name": m.name or "tool",
                                "response": {
                                    "name": m.name or "tool",
                                    "content": m.content,
                                },
                            }
                        }
                    )

                role = (
                    "model" if m.role == "assistant" else ("tool" if m.role == "tool" else "user")
                )
                contents.append({"role": role, "parts": parts})

        request_body: dict[str, Any] = {"contents": contents}

        if extracted_sys_instruction:
            request_body["systemInstruction"] = {"parts": [{"text": extracted_sys_instruction}]}

        if tools:
            formatted_tools: list[dict[str, Any]] = []
            func_decls: list[dict[str, Any]] = []
            for t in tools:
                if isinstance(t, Tool):
                    clean_params = (
                        sanitize_params_for_google(t.parameters) if t.parameters else None
                    )
                    decl: dict[str, Any] = {
                        "name": t.name,
                        "description": t.description,
                    }
                    if clean_params:
                        decl["parameters"] = clean_params
                    func_decls.append(decl)
                elif isinstance(t, dict):
                    if "googleSearch" in t or "codeExecution" in t or "functionDeclarations" in t:
                        formatted_tools.append(t)
                    else:
                        func_decls.append(t)
            if func_decls:
                formatted_tools.append({"functionDeclarations": func_decls})
            if formatted_tools:
                request_body["tools"] = formatted_tools

        payload: dict[str, Any] = {"request": request_body}
        if model:
            target_model = model
            if (
                target_model.startswith("gemini-3.7-flash")
                and target_model != "gemini-3.7-flash-tiered"
            ):
                target_model = "gemini-3.7-flash-tiered"
            payload["model"] = target_model

        response = await self._post_with_fallback(url, json_data=payload, headers=headers)
        if response.status_code != 200:
            self._raise_for_status(response.status_code, response.text)

        resp_data = response.json()
        return CountTokensResult(total_tokens=int(resp_data.get("totalTokens", 0)))

    async def search(
        self,
        query: str,
        model: str | None = None,
        **kwargs,
    ) -> SearchResponse:
        """Performs a web search grounding query via Google Search Grounding API.

        Args:
            query (str): The search query to ground against live web information.
            model (Optional[str], optional): Model ID to use. Defaults to the first in webSearchModelIds.
            **kwargs: Additional options passed to Client.generate.

        Returns:
            SearchResponse: Grounded response text accompanied by extracted SearchSource citations.
        """
        catalog = await self.models.get_catalog()

        if not model:
            if catalog.web_search_model_ids:
                model = catalog.web_search_model_ids[0]
            else:
                model = "gemini-2.5-flash"
        elif catalog.web_search_model_ids and model not in catalog.web_search_model_ids:
            warnings.warn(
                f"Model '{model}' is not in webSearchModelIds: {catalog.web_search_model_ids}.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "Model '%s' is not in webSearchModelIds: %s",
                model,
                catalog.web_search_model_ids,
            )

        messages = [Message(role="user", content=query)]
        kwargs["tools"] = [{"googleSearch": {}}]
        chat_resp: ChatResponse = await self.generate(
            model=model,
            messages=messages,
            stream=False,
            **kwargs,
        )

        sources: list[SearchSource] = []
        if getattr(chat_resp, "_grounding_metadata", None):
            gm = chat_resp._grounding_metadata
            for chunk in gm.get("groundingChunks", []):
                web = chunk.get("web", {})
                if web and web.get("uri"):
                    sources.append(
                        SearchSource(
                            title=web.get("title", "Source"),
                            uri=web.get("uri", ""),
                        )
                    )

        return SearchResponse(
            text=chat_resp.text or "",
            sources=sources,
            grounding_metadata=getattr(chat_resp, "_grounding_metadata", None),
            usage=chat_resp.usage,
            model_version=chat_resp.model_version,
        )

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        aspect_ratio: str | None = None,
        negative_prompt: str | None = None,
        number_of_images: int = 1,
        image_format: str = "image/png",
        **kwargs,
    ) -> ImageGenerationResponse:
        """Generates images natively via the image generation model (e.g. gemini-3.1-flash-image).

        Args:
            prompt (str): Detailed text description of the image to generate.
            model (Optional[str], optional): Image generation model ID. Defaults to first in imageGenerationModelIds.
            aspect_ratio (Optional[str], optional): Desired aspect ratio (e.g. '1:1', '16:9', '4:3').
            negative_prompt (Optional[str], optional): Unwanted elements to exclude from generation.
            number_of_images (int, optional): Number of image candidates to generate. Defaults to 1.
            image_format (str, optional): Target MIME type. Defaults to 'image/png'.
            **kwargs: Additional generation settings.

        Returns:
            ImageGenerationResponse: Response containing generated FileAttachment images.

        Raises:
            AuthError: If authentication fails.
            AgentAPIError: If backend image generation fails.
        """
        catalog = await self.models.get_catalog()

        if not model:
            if catalog.image_generation_model_ids:
                model = catalog.image_generation_model_ids[0]
            else:
                model = "gemini-3.1-flash-image"
        elif catalog.image_generation_model_ids and model not in catalog.image_generation_model_ids:
            warnings.warn(
                f"Model '{model}' is not in imageGenerationModelIds: {catalog.image_generation_model_ids}.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "Model '%s' is not in imageGenerationModelIds: %s",
                model,
                catalog.image_generation_model_ids,
            )

        await self.fetch_project_id()
        await self._check_and_refresh_token()

        import time

        url = f"{self.base_url}/v1internal:streamGenerateContent?alt=sse"

        gen_config: dict[str, Any] = {
            "responseModalities": ["IMAGE"],
            "temperature": kwargs.get("temperature", 0.7),
        }
        if aspect_ratio:
            gen_config["aspectRatio"] = aspect_ratio
        if negative_prompt:
            gen_config["negativePrompt"] = negative_prompt
        if number_of_images > 1:
            gen_config["candidateCount"] = number_of_images

        payload = {
            "project": self.project_id,
            "model": model,
            "requestType": "agent",
            "userAgent": "antigravity",
            "requestId": f"agent-image-{int(time.time() * 1000)}",
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": gen_config,
            },
        }

        headers = {
            **DEFAULT_CLIENT_HEADERS,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }

        response = await self._post_with_fallback(url, json_data=payload, headers=headers)
        if response.status_code != 200:
            self._raise_for_status(response.status_code, response.text)

        images: list[FileAttachment] = []
        usage = UsageStats(0, 0, 0)
        model_version = None

        for line_str in response.iter_lines():
            line_str = line_str.strip()
            if line_str and line_str.startswith("data: "):
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunks = json.loads(data_str)
                    if not isinstance(chunks, list):
                        chunks = [chunks]
                    for chunk in chunks:
                        resp = chunk.get("response", {})
                        if "modelVersion" in resp:
                            model_version = resp["modelVersion"]
                        if "usageMetadata" in resp:
                            meta = resp["usageMetadata"]
                            usage = UsageStats(
                                prompt_tokens=meta.get("promptTokenCount", 0),
                                completion_tokens=meta.get("candidatesTokenCount", 0),
                                total_tokens=meta.get("totalTokenCount", 0),
                            )
                        for candidate in resp.get("candidates", []):
                            for part in candidate.get("content", {}).get("parts", []):
                                if "inlineData" in part:
                                    inline = part["inlineData"]
                                    images.append(
                                        FileAttachment(
                                            mime_type=inline.get("mimeType", image_format),
                                            data=inline.get("data", ""),
                                        )
                                    )
                except Exception:
                    pass

        return ImageGenerationResponse(
            images=images,
            usage=usage,
            model_version=model_version,
        )

    def _validate_request_params(
        self,
        model: str,
        catalog: ModelsCatalog | None,
        messages: list[Message],
        max_output_tokens: int | None,
        thinking_budget: int | None,
    ) -> None:
        """Validates MIME types, token boundaries, and model capability settings."""
        if not catalog:
            return

        # Check deprecated models
        if model in catalog.deprecated_model_ids:
            alternative = catalog.deprecated_model_map.get(model, catalog.default_agent_model_id)
            warnings.warn(
                f"Model '{model}' is deprecated. Recommended alternative: '{alternative}'.",
                DeprecationWarning,
                stacklevel=3,
            )
            logger.warning(
                "Model '%s' is deprecated. Recommended alternative: '%s'.",
                model,
                alternative,
            )

        model_info = catalog.get_model(model)
        if not model_info:
            return

        # Validate MIME types and media capabilities
        for msg in messages:
            if getattr(msg, "attachments", None):
                for att in msg.attachments:
                    if (
                        model_info.supported_mime_types
                        and att.mime_type not in model_info.supported_mime_types
                    ):
                        raise ValueError(
                            f"MIME type '{att.mime_type}' is not supported by model '{model}'. "
                            f"Supported MIME types: {list(model_info.supported_mime_types.keys())}"
                        )
                    if att.mime_type.startswith("image/") and not model_info.supports_images:
                        raise ValueError(f"Model '{model}' does not support image inputs.")
                    if att.mime_type.startswith("video/") and not model_info.supports_video:
                        raise ValueError(f"Model '{model}' does not support video inputs.")

        # Validate max_output_tokens
        if (
            max_output_tokens
            and model_info.max_output_tokens
            and max_output_tokens > model_info.max_output_tokens
        ):
            raise ValueError(
                f"max_output_tokens ({max_output_tokens}) exceeds model maximum limit ({model_info.max_output_tokens})."
            )

        # Validate thinking limits
        if thinking_budget:
            if not model_info.supports_thinking:
                raise ValueError(f"Model '{model}' does not support reasoning/thinking.")
            if model_info.min_thinking_budget and thinking_budget < model_info.min_thinking_budget:
                raise ValueError(
                    f"thinking_budget ({thinking_budget}) is below model minimum ({model_info.min_thinking_budget})."
                )
            if model_info.thinking_budget and thinking_budget > model_info.thinking_budget:
                raise ValueError(
                    f"thinking_budget ({thinking_budget}) exceeds model maximum ({model_info.thinking_budget})."
                )

    @overload
    async def generate(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.8,
        stream: Literal[False] = False,
        tools: list[Tool | dict[str, Any]] | None = None,
        options: GenerateOptions | None = None,
        **kwargs,
    ) -> ChatResponse: ...

    @overload
    async def generate(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.8,
        stream: Literal[True] = True,
        tools: list[Tool | dict[str, Any]] | None = None,
        options: GenerateOptions | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]: ...

    async def generate(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.8,
        stream: bool = False,
        tools: list[Tool | dict[str, Any]] | None = None,
        options: GenerateOptions | None = None,
        **kwargs,
    ) -> ChatResponse | AsyncIterator[StreamChunk]:
        """Generates a response from the model asynchronously using streamGenerateContent.

        Args:
            model (str): The identifier of the model to query (e.g. 'gemini-3.1-pro-low').
            messages (List[Message]): The list of conversational turns and attachments.
            temperature (float, optional): Sampling temperature. Defaults to 0.8.
            stream (bool, optional): Whether to yield chunks as an AsyncIterator. Defaults to False.
            tools (Optional[List[Union[Tool, Dict[str, Any]]]], optional): Tools available for function calling.
            options (Optional[GenerateOptions], optional): Structured generation parameters.
            **kwargs: Additional parameters passed to the generation config.

        Returns:
            Union[ChatResponse, AsyncIterator[StreamChunk]]: A complete ChatResponse if stream=False,
                or an AsyncIterator of StreamChunk objects if stream=True.

        Raises:
            AuthError: If the user is unauthenticated or token is expired.
            ModelNotFoundError: If the requested model does not exist or is unavailable.
            RateLimitError: If the API quota or rate limit is exceeded.
            AgentAPIError: If a backend server error occurs.
        """
        await self.fetch_project_id()
        await self._check_and_refresh_token()
        import time

        catalog = None
        try:
            catalog = await self.models.get_catalog()
        except Exception:
            pass

        # Merge GenerateOptions if provided
        if options:
            if options.temperature is not None:
                temperature = options.temperature
            if options.max_output_tokens is not None:
                kwargs["max_output_tokens"] = options.max_output_tokens
            if options.top_p is not None:
                kwargs["top_p"] = options.top_p
            if options.top_k is not None:
                kwargs["top_k"] = options.top_k
            if options.stop_sequences is not None:
                kwargs["stop_sequences"] = options.stop_sequences
            if options.thinking_budget is not None:
                kwargs["thinking_budget"] = options.thinking_budget
            if options.thinking_level is not None:
                kwargs["thinking_level"] = options.thinking_level
            if options.safety_settings is not None:
                kwargs["safety_settings"] = options.safety_settings

        # Parse model thinking & tiered mapping
        effective_model = model
        thinking_level = kwargs.get("thinking_level") or parse_effort_from_suffix(model)
        max_output_tokens = kwargs.get("max_tokens") or kwargs.get("max_output_tokens", 8192)
        thinking_budget = kwargs.get("thinking_budget")

        # Reverse mapping for -tiered models (e.g. gemini-3.7-flash-high -> gemini-3.7-flash-tiered)
        if "3.7-flash" in model.lower() or "3.6-flash" in model.lower():
            if not effective_model.endswith("-tiered"):
                for sfx in ("-extra-low", "-low", "-medium", "-high"):
                    if effective_model.endswith(sfx):
                        thinking_level = sfx[1:]
                        effective_model = effective_model[: -len(sfx)]
                if not effective_model.endswith("-tiered"):
                    effective_model = f"{effective_model}-tiered"
            if not thinking_level:
                thinking_level = "high"

        # Validate request parameters against catalog
        self._validate_request_params(
            model=model,
            catalog=catalog,
            messages=messages,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
        )

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
            if msg.content:
                parts.append({"text": msg.content})
            if msg.thought:
                parts.append({"thought": msg.thought})
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
                func_resp = {
                    "name": (getattr(msg, "name", None) or msg.tool_call_id or "unknown_tool"),
                    "response": (
                        resp_data if isinstance(resp_data, dict) else {"output": resp_data}
                    ),
                }
                if msg.tool_call_id:
                    func_resp["id"] = msg.tool_call_id
                parts = [{"functionResponse": func_resp}]
                role = "user"

            if parts:
                is_tool = msg.role == "tool"
                if contents and contents[-1]["role"] == role:
                    prev_has_tool = any("functionResponse" in p for p in contents[-1]["parts"])
                    if is_tool == prev_has_tool:
                        contents[-1]["parts"].extend(parts)
                    else:
                        if prev_has_tool and not is_tool:
                            contents.append({"role": "model", "parts": [{"text": ""}]})
                        contents.append({"role": role, "parts": parts})
                else:
                    contents.append({"role": role, "parts": parts})

        thinking_config = None
        if kwargs.get("thinking_config") is not False:
            if isinstance(kwargs.get("thinking_config"), dict):
                thinking_config = kwargs["thinking_config"]
            elif "claude" in model.lower():
                budget = thinking_budget or (
                    1024
                    if thinking_level == "low"
                    else (16384 if thinking_level == "high" else 4096)
                )
                if budget >= max_output_tokens:
                    budget = max(1024, max_output_tokens - 1024)
                thinking_config = {
                    "includeThoughts": True,
                    "thinkingBudget": budget,
                }
            elif any(k in model.lower() for k in ("gpt", "oss", "openai", "tab_", "chat_")):
                thinking_config = None
            else:
                thinking_config = {
                    "includeThoughts": True,
                    "thinkingLevel": thinking_level or "high",
                }

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        if "top_p" in kwargs:
            generation_config["topP"] = kwargs["top_p"]
        if "top_k" in kwargs:
            generation_config["topK"] = kwargs["top_k"]
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
            "model": effective_model,
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
            direct_tools = []
            for t in tools:
                if isinstance(t, Tool):
                    func_decls.append(
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": sanitize_params_for_google(t.parameters),
                        }
                    )
                elif isinstance(t, dict):
                    if "functionDeclarations" in t or "googleSearch" in t:
                        direct_tools.append(t)
                    else:
                        direct_tools.append(t)

            req_tools = list(direct_tools)
            if func_decls:
                req_tools.append({"functionDeclarations": func_decls})
            payload["request"]["tools"] = req_tools
            payload["request"]["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        headers = {
            **DEFAULT_CLIENT_HEADERS,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }

        async def _do_stream() -> AsyncIterator[StreamChunk]:
            async for chunk in self._stream_with_fallback(url, payload, headers):
                yield chunk

        async def _do_sync() -> ChatResponse:
            response = await self._post_with_fallback(url, json_data=payload, headers=headers)
            if response.status_code != 200:
                self._raise_for_status(response.status_code, response.text)
            return self._sync_response(response)

        if stream:
            return _do_stream()
        else:
            return await _do_sync()

    async def _post_with_fallback(
        self,
        url: str,
        json_data: Any,
        headers: dict[str, str],
        timeout: float = 60.0,
    ) -> httpx.Response:
        """Executes a POST request to primary backend with automatic fallback to secondary."""
        try:
            response = await self.http_client.post(
                url, json=json_data, headers=headers, timeout=timeout
            )
            if response.status_code != 429:
                return response
            logger.warning(
                "Primary backend rate limit (429) on %s. Attempting fallback to Production...",
                url,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning(
                "Primary backend network error on %s (%s). Attempting fallback to Production...",
                url,
                str(exc),
            )

        fallback_url = url.replace(API_ENDPOINTS["production"], API_ENDPOINTS["prod_fallback"])
        if fallback_url == url:
            fallback_url = f"{API_ENDPOINTS['prod_fallback']}" + url[len(self.base_url) :]

        fb_response = await self.http_client.post(
            fallback_url, json=json_data, headers=headers, timeout=timeout
        )
        return fb_response

    async def _stream_with_fallback(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[StreamChunk]:
        """Executes an async SSE streaming request with automatic fallback on 429."""
        try:
            async with self.http_client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                if response.status_code != 429:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        self._raise_for_status(
                            response.status_code,
                            error_text.decode("utf-8", errors="replace"),
                        )
                    async for chunk in self._stream_response(response):
                        yield chunk
                    return
                logger.warning(
                    "Primary streaming backend rate limited (429). Falling back to Production..."
                )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning(
                "Primary streaming backend network error (%s). Falling back to Production...",
                str(exc),
            )

        fallback_url = url.replace(API_ENDPOINTS["production"], API_ENDPOINTS["prod_fallback"])
        if fallback_url == url:
            fallback_url = f"{API_ENDPOINTS['prod_fallback']}" + url[len(self.base_url) :]

        async with self.http_client.stream(
            "POST", fallback_url, json=payload, headers=headers
        ) as fb_response:
            if fb_response.status_code != 200:
                error_text = await fb_response.aread()
                self._raise_for_status(
                    fb_response.status_code,
                    error_text.decode("utf-8", errors="replace"),
                )
            async for chunk in self._stream_response(fb_response):
                yield chunk

    @staticmethod
    def _parse_part(
        part: dict,
    ) -> tuple[str | None, str | None, str | None]:
        """Extracts text, thought content, and thought signature from a candidate content part."""
        signature = part.get("thoughtSignature")
        is_thought = part.get("thought") is True or part.get("isThought") is True
        if is_thought:
            thought_text = part.get("text") or (
                part.get("thought") if isinstance(part.get("thought"), str) else None
            )
            return None, thought_text, signature
        if "thought" in part and isinstance(part["thought"], str) and "text" not in part:
            return None, part["thought"], signature
        if "text" in part:
            return part["text"], None, signature
        return None, None, signature

    def _sync_response(self, response: httpx.Response) -> ChatResponse:
        """Parses a synchronous (non-streaming) SSE response from the API."""
        full_text = ""
        full_thought = ""
        thought_signature = None
        tool_calls: list[ToolCall] = []
        citations: list[Citation] = []
        safety_ratings: list[SafetyRating] = []
        finish_reason = "stop"
        usage = UsageStats(0, 0, 0)
        model_version = None
        response_id = None
        grounding_metadata = None

        for line_str in response.iter_lines():
            line_str = line_str.strip()
            if line_str and line_str.startswith("data: "):
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunks = json.loads(data_str)
                    if not isinstance(chunks, list):
                        chunks = [chunks]
                    for chunk in chunks:
                        resp = chunk.get("response", {})
                        if "responseId" in resp:
                            response_id = resp["responseId"]
                        if "modelVersion" in resp:
                            model_version = resp["modelVersion"]

                        if "usageMetadata" in resp:
                            meta = resp["usageMetadata"]
                            thoughts_count = meta.get("thoughtsTokenCount", 0)
                            c_details_raw = meta.get("candidatesTokensDetails", [])
                            c_details = []
                            if isinstance(c_details_raw, list):
                                for d in c_details_raw:
                                    c_details.append(
                                        TokenDetail(
                                            modality=d.get("modality", "TEXT"),
                                            token_count=d.get("tokenCount", 0),
                                        )
                                    )
                                    if not thoughts_count and d.get("tokenCount"):
                                        thoughts_count = d.get("tokenCount", 0)
                            p_details_raw = meta.get("promptTokensDetails", [])
                            p_details = []
                            if isinstance(p_details_raw, list):
                                for d in p_details_raw:
                                    p_details.append(
                                        TokenDetail(
                                            modality=d.get("modality", "TEXT"),
                                            token_count=d.get("tokenCount", 0),
                                        )
                                    )
                            usage = UsageStats(
                                prompt_tokens=meta.get("promptTokenCount", 0),
                                completion_tokens=meta.get("candidatesTokenCount", 0),
                                total_tokens=meta.get("totalTokenCount", 0),
                                cached_tokens=meta.get("cachedContentTokenCount", 0),
                                thoughts_tokens=thoughts_count,
                                candidates_token_details=c_details,
                                prompt_token_details=p_details,
                            )

                        candidates = resp.get("candidates", [])
                        for candidate in candidates:
                            if "finishReason" in candidate:
                                finish_reason = candidate["finishReason"]

                            if "groundingMetadata" in candidate:
                                grounding_metadata = candidate["groundingMetadata"]

                            if "citationMetadata" in candidate:
                                for c in candidate["citationMetadata"].get("citations", []):
                                    citations.append(
                                        Citation(
                                            start_index=c.get("startIndex", 0),
                                            end_index=c.get("endIndex", 0),
                                            uri=c.get("uri", ""),
                                            title=c.get("title"),
                                            license=c.get("license"),
                                        )
                                    )

                            if "safetyRatings" in candidate:
                                for sr in candidate["safetyRatings"]:
                                    safety_ratings.append(
                                        SafetyRating(
                                            category=sr.get("category", ""),
                                            probability=sr.get("probability", ""),
                                            blocked=bool(sr.get("blocked", False)),
                                        )
                                    )

                            parts = candidate.get("content", {}).get("parts", [])
                            for part in parts:
                                text_val, thought_val, sig = self._parse_part(part)
                                if sig:
                                    thought_signature = sig
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
                                            thought_signature=part.get("thoughtSignature"),
                                        )
                                    )
                except json.JSONDecodeError:
                    logger.warning("Failed to decode chunk: %s", data_str)

        chat_resp = ChatResponse(
            text=full_text if full_text else None,
            thought=full_thought if full_thought else None,
            thought_signature=thought_signature,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls if tool_calls else None,
            safety_ratings=safety_ratings if safety_ratings else None,
            citations=citations if citations else None,
            model_version=model_version,
            response_id=response_id,
        )
        chat_resp._grounding_metadata = grounding_metadata
        return chat_resp

    async def _stream_response(self, response: httpx.Response) -> AsyncIterator[StreamChunk]:
        """Parses an async streaming SSE response from the API."""
        tool_calls: list[ToolCall] = []

        async for line_str in response.aiter_lines():
            line_str = line_str.strip()
            if line_str and line_str.startswith("data: "):
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunks = json.loads(data_str)
                    if not isinstance(chunks, list):
                        chunks = [chunks]
                    for chunk in chunks:
                        resp = chunk.get("response", {})
                        model_ver = resp.get("modelVersion")
                        resp_id = resp.get("responseId")
                        usage = None
                        if "usageMetadata" in resp:
                            meta = resp["usageMetadata"]
                            usage = UsageStats(
                                prompt_tokens=meta.get("promptTokenCount", 0),
                                completion_tokens=meta.get("candidatesTokenCount", 0),
                                total_tokens=meta.get("totalTokenCount", 0),
                                cached_tokens=meta.get("cachedContentTokenCount", 0),
                                thoughts_tokens=meta.get("thoughtsTokenCount", 0),
                            )

                        candidates = resp.get("candidates", [])
                        for candidate in candidates:
                            finish_reason = candidate.get("finishReason")
                            parts = candidate.get("content", {}).get("parts", [])
                            for part in parts:
                                text_val, thought_val, sig = self._parse_part(part)
                                if thought_val:
                                    yield StreamChunk(
                                        thought=thought_val,
                                        thought_signature=sig,
                                        usage=usage,
                                        model_version=model_ver,
                                        response_id=resp_id,
                                        finish_reason=finish_reason,
                                    )
                                if text_val:
                                    yield StreamChunk(
                                        text=text_val,
                                        thought_signature=sig,
                                        usage=usage,
                                        model_version=model_ver,
                                        response_id=resp_id,
                                        finish_reason=finish_reason,
                                    )
                                if "functionCall" in part:
                                    fc = part["functionCall"]
                                    tool_calls.append(
                                        ToolCall(
                                            id=fc.get("id", fc.get("name")),
                                            name=fc.get("name"),
                                            arguments=fc.get("args", {}),
                                            thought_signature=part.get("thoughtSignature"),
                                        )
                                    )
                except json.JSONDecodeError:
                    logger.warning("Failed to decode chunk: %s", data_str)

        if tool_calls:
            yield StreamChunk(tool_calls=tool_calls)

    @staticmethod
    def _raise_for_status(status_code: int, error_text: str) -> None:
        """Raises custom exceptions with parsed metadata details on error."""
        if status_code == 200:
            return
        if status_code == 401:
            raise AuthError(f"Unauthorized (401): {error_text}")
        if status_code == 404:
            raise ModelNotFoundError(f"Model or resource not found (404): {error_text}")
        if status_code == 429:
            delay = None
            ts = None
            try:
                err_json = json.loads(error_text)
                err_obj = err_json.get("error", {})
                delay = err_obj.get("quotaResetDelay")
                ts = err_obj.get("quotaResetTimeStamp")
                details = err_obj.get("details", [])
                for d in details:
                    if not delay:
                        delay = d.get("quotaResetDelay") or d.get("metadata", {}).get(
                            "quotaResetDelay"
                        )
                    if not ts:
                        ts = d.get("quotaResetTimeStamp") or d.get("metadata", {}).get(
                            "quotaResetTimeStamp"
                        )
            except Exception:
                pass
            reset_hint = ""
            if delay or ts:
                reset_hint = f" (Quota resets in {delay or ''} at {ts or ''})"
            raise RateLimitError(
                f"Rate limit exceeded (429): {error_text}{reset_hint}",
                quota_reset_delay=delay,
                quota_reset_timestamp=ts,
            )
        raise AgentAPIError(f"Error {status_code}: {error_text}")

    @property
    def _lock(self):
        """Returns the asyncio Lock used to prevent race conditions during token refresh."""
        if not hasattr(self, "_async_lock"):
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _save_account_info(self) -> None:
        """Saves account information to ~/.anti-api/accounts.json."""
        data_dir = os.environ.get("ANTI_API_DATA_DIR")
        if not data_dir:
            home = (
                os.environ.get("HOME") or os.environ.get("USERPROFILE") or os.path.expanduser("~")
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
                if response.status_code != 200:
                    data["client_id"] = OAUTH_CONFIG["fallback_client_id"]
                    data["client_secret"] = OAUTH_CONFIG["fallback_client_secret"]
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
        """Loads account information from local cache file."""
        data_dir = os.environ.get("ANTI_API_DATA_DIR")
        if not data_dir:
            home = (
                os.environ.get("HOME") or os.environ.get("USERPROFILE") or os.path.expanduser("~")
            )
            data_dir = os.path.join(home, ".anti-api")
        accounts_file = os.path.join(data_dir, "accounts.json")
        if os.path.exists(accounts_file):
            try:
                with open(accounts_file, encoding="utf-8") as f:
                    data = json.load(f)
                    accounts = data.get("accounts", [])
                    if accounts:
                        return accounts[0]
            except Exception:
                pass
        return {}
