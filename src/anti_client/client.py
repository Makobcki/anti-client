import json
import os
import secrets
import threading
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import AsyncIterator, Dict, List, Optional, Union

import httpx

from .exceptions import AgentAPIError, AuthError
from .types import (
    ChatResponse,
    Message,
    ModelInfo,
    QuotaInfo,
    Tool,
    ToolCall,
    UsageStats,
)

OAUTH_CONFIG = {
    "client_id": os.environ.get("ANTI_CLIENT_ID", "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"),
    "client_secret": os.environ.get("ANTI_CLIENT_SECRET", "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"),
    "callback_port": 51121,
    "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_url": "https://oauth2.googleapis.com/token",
    "project_url": "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist",
    "scopes": [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    ],
}

auth_code = None
auth_error = None
server_ready = threading.Event()


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handles the OAuth callback redirect locally."""
    def do_GET(self):
        """Processes the GET request for the OAuth callback."""
        global auth_code, auth_error
        parsed_path = urllib.parse.urlparse(self.path)

        if parsed_path.path == "/oauth-callback":
            query_params = urllib.parse.parse_qs(parsed_path.query)

            if "error" in query_params:
                auth_error = query_params["error"][0]
            elif "code" in query_params:
                auth_code = query_params["code"][0]

            self.send_response(302)
            self.send_header("Location", "https://antigravity.google/auth-success")
            self.end_headers()

            threading.Thread(target=self.server.shutdown).start()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        """Suppress standard log messages."""
        pass


def start_server() -> int:
    """Starts the local HTTP server to receive the OAuth callback.
    
    Returns:
        int: The port number the server is listening on.
        
    Raises:
        Exception: If no available port could be bound.
    """
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
        raise Exception("Could not bind to any port for OAuth callback")

    server_ready.set()
    httpd.serve_forever()
    return httpd.server_port


def authenticate():
    """Authenticates the user via Google OAuth 2.0.
    
    This function starts a local server, opens the user's browser, waits for the OAuth
    callback, exchanges the code for tokens, retrieves the project ID, and saves the
    authentication data to `~/.anti-api/accounts.json`.
    """
    global auth_code, auth_error
    auth_code = None
    auth_error = None
    server_ready.clear()

    state = secrets.token_urlsafe(16)

    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    server_ready.wait()
    redirect_uri = f"http://localhost:{OAUTH_CONFIG['callback_port']}/oauth-callback"

    params = {
        "client_id": OAUTH_CONFIG["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(OAUTH_CONFIG["scopes"]),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }

    auth_url = f"{OAUTH_CONFIG['auth_url']}?{urllib.parse.urlencode(params)}"

    print(
        f"Opening browser for authorization...\nIf the browser does not open automatically, please visit:\n{auth_url}\n"
    )
    webbrowser.open(auth_url)

    server_thread.join(timeout=300)

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
    }

    response = httpx.post(
        OAUTH_CONFIG["token_url"], data=token_data, timeout=60.0
    )
    if response.status_code != 200:
        print(f"Token exchange error: {response.status_code}\n{response.text}")
        return

    tokens = response.json()
    access_token = tokens.get("access_token")

    print("Retrieving Project ID...")
    project_response = httpx.post(
        OAUTH_CONFIG["project_url"],
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": "antigravity/2.0.10 macos/arm64",
        },
        json={"metadata": {"ideType": "ANTIGRAVITY"}},
        timeout=60.0,
    )

    project_id = "unknown"
    if project_response.status_code == 200:
        project_data = project_response.json()
        project_id = project_data.get("cloudaicompanionProject", "unknown")
    else:
        print(
            f"Failed to get Project ID: {project_response.status_code}\n{project_response.text}"
        )

    data_dir = os.environ.get("ANTI_API_DATA_DIR")
    if not data_dir:
        home = (
            os.environ.get("HOME")
            or os.environ.get("USERPROFILE")
            or os.path.expanduser("~")
        )
        data_dir = os.path.join(home, ".anti-api")
    os.makedirs(data_dir, exist_ok=True)

    accounts_file = os.path.join(data_dir, "accounts.json")
    with open(accounts_file, "w", encoding="utf-8") as f:
        json.dump(
            {"accounts": [{"accessToken": access_token, "projectId": project_id}]}, f
        )

    print("Authorization completed and saved successfully!")


class ModelsResource:
    """Provides methods for querying available models."""
    
    def __init__(self, client: "Client"):
        """Initializes the ModelsResource with a client.
        
        Args:
            client (Client): The main client instance.
        """
        self._client = client

    async def list(self) -> List[ModelInfo]:
        """Fetches a list of all available models asynchronously.
        
        Returns:
            List[ModelInfo]: A list of available models and their metadata.
            
        Raises:
            AgentAPIError: If the API request fails.
        """
        url = "https://cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels"

        headers = {
            "Authorization": f"Bearer {self._client.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "antigravity/2.0.10 macos/arm64",
        }

        data = {"project": self._client.project_id}

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            response = await http_client.post(url, json=data, headers=headers)
            
        if response.status_code != 200:
            raise AgentAPIError(
                f"Error fetching models {response.status_code}: {response.text}"
            )

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
        """Gets a mapping of model IDs to their quota info asynchronously.
        
        Returns:
            Dict[str, QuotaInfo]: A dictionary mapping model IDs to their quotas.
        """
        models = await self.list()
        return {m.id: m.quota_info for m in models if m.quota_info}


class Client:
    """The main async client for interacting with the Cloud Code (Gemini) API.
    
    Attributes:
        api_key (str): The OAuth access token.
        project_id (str): The Google Cloud project ID.
        models (ModelsResource): Resource for interacting with models.
    """
    
    def __init__(self, api_key: Optional[str] = None, project_id: Optional[str] = None):
        """Initializes the client.
        
        If `api_key` or `project_id` are not provided, it will attempt to load them from
        the local cache file or environment variables.
        
        Args:
            api_key (Optional[str], optional): The OAuth access token. Defaults to None.
            project_id (Optional[str], optional): The Google Cloud project ID. Defaults to None.
            
        Raises:
            AuthError: If authentication tokens are missing.
        """
        self.api_key = api_key or os.environ.get("MY_API_KEY")
        self.project_id = project_id

        if not self.api_key or not self.project_id:
            token, proj = self._load_account_info()
            if not self.api_key:
                self.api_key = token
            if not self.project_id:
                self.project_id = proj

        if not self.api_key or self.api_key == "YOUR_ACCESS_TOKEN":
            raise AuthError(
                "API key not provided. Run anti_client.authenticate()"
            )

        self.models = ModelsResource(self)

    def _load_account_info(self):
        """Loads account information from the local credentials file.
        
        Returns:
            tuple: A tuple containing (access_token, project_id) or (None, None).
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
            with open(accounts_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                accounts = data.get("accounts", [])
                if accounts:
                    return accounts[0].get("accessToken"), accounts[0].get(
                        "projectId", "unknown"
                    )

        return None, None

    async def generate(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.8,
        stream: bool = False,
        tools: Optional[List[Tool]] = None,
        **kwargs,
    ) -> Union[ChatResponse, AsyncIterator[Union[str, List[ToolCall]]]]:
        """Generates a response from the model asynchronously.
        
        Args:
            model (str): The model ID to use.
            messages (List[Message]): The conversation history.
            temperature (float, optional): The sampling temperature. Defaults to 0.8.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            tools (Optional[List[Tool]], optional): A list of tools available to the model. Defaults to None.
            **kwargs: Additional generation parameters.
            
        Returns:
            Union[ChatResponse, AsyncIterator[Union[str, List[ToolCall]]]]: The full ChatResponse, or an async iterator yielding text chunks and eventually a list of tool calls.
            
        Raises:
            AgentAPIError: If the generation request fails.
        """
        url = "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"

        contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = {"role": "user", "parts": [{"text": msg.content}]}
                continue

            role = "model" if msg.role == "assistant" else "user"

            parts = []
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
                except:
                    resp_data = {"result": msg.content}

                parts = [
                    {
                        "functionResponse": {
                            "name": msg.tool_call_id or "unknown_tool",
                            "response": resp_data,
                        }
                    }
                ]
                role = "user"

            contents.append({"role": role, "parts": parts})

        payload = {
            "model": model,
            "userAgent": "antigravity/2.0.10 macos/arm64",
            "requestType": "agent",
            "project": self.project_id,
            "requestId": f"agent-{uuid.uuid4()}",
            "request": {
                "contents": contents,
                "sessionId": "-1234567890",
                "generationConfig": {
                    "maxOutputTokens": 64000,
                    "stopSequences": ["\n\nHuman:", "[DONE]"],
                    "temperature": temperature,
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "OFF"},
                ],
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
                        "parameters": t.parameters,
                    }
                )
            payload["request"]["tools"] = [{"functionDeclarations": func_decls}]
            payload["request"]["toolConfig"] = {
                "functionCallingConfig": {"mode": "ANY"}
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "antigravity/2.0.10 macos/arm64",
            "Accept": "text/event-stream",
        }

        async def _do_stream() -> AsyncIterator[Union[str, List[ToolCall]]]:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                async with http_client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise AgentAPIError(f"Error {response.status_code}: {error_text.decode('utf-8', errors='replace')}")
                    
                    async for chunk in self._stream_response(response):
                        yield chunk

        async def _do_sync() -> ChatResponse:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                response = await http_client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise AgentAPIError(f"Error {response.status_code}: {response.text}")
                return self._sync_response(response)

        if stream:
            return _do_stream()
        else:
            return await _do_sync()

    def _sync_response(self, response: httpx.Response) -> ChatResponse:
        """Parses a synchronous (non-streaming) response from the API.
        
        Args:
            response (httpx.Response): The raw HTTP response.
            
        Returns:
            ChatResponse: The fully parsed response containing text, usage, and tool calls.
        """
        full_text = ""
        tool_calls = []
        finish_reason = "stop"
        usage = UsageStats(0, 0, 0)

        for line_str in response.iter_lines():
            if line_str:
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
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
                                    if "text" in part:
                                        full_text += part["text"]
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
                        pass

        return ChatResponse(
            text=full_text if full_text else None,
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls if tool_calls else None,
        )

    async def _stream_response(self, response: httpx.Response) -> AsyncIterator[Union[str, List[ToolCall]]]:
        """Parses an async streaming response from the API.
        
        Yields text chunks and optionally yields a list of tool calls at the end.
        
        Args:
            response (httpx.Response): The raw HTTP response.
            
        Yields:
            Union[str, List[ToolCall]]: Chunks of text generated by the model, or a list of tool calls.
        """
        tool_calls = []
        async for line_str in response.aiter_lines():
            if line_str:
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
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
                                    if "text" in part:
                                        yield part["text"]
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
                        pass
                        
        if tool_calls:
            yield tool_calls
