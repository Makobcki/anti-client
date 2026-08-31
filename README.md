# anti-client

A full-featured async Python client library for interacting with the internal Antigravity / Cloud Code API (Gemini, Claude, GPT-OSS).

## Features

- **OAuth 2.0 PKCE & Flexible Storage**: Built-in authentication flow with auto-discovery (`.anti-api/accounts.json`, `accounts.json`, `~/.anti-api/accounts.json`), project-level saving (`authenticate(save_to_project=True)`), and custom path support.
- **Manual Key & Token Passing**: Direct initialization with manual `api_key`, `refresh_token`, or `credentials` dictionary without requiring local files.
- **Multi-Account Support**: Manage multiple accounts in one file, select by email or index (`Client(account_email="...")`), and preserve emails during token refreshes.
- **Multi-Backend Resilience**: Primary routing to Daily backend with automatic fallback to Production upon rate limits (`429`) or network errors.
- **Family Quotas**: Granular tracking for Gemini and Claude/3P model families across 5-hour and weekly windows (`client.get_quota_summary()`).
- **Web Search Grounding**: Native Google Search integration with source citation extraction (`client.search()`).
- **Image Generation**: Native multimodal image generation with aspect ratio support and file saving (`client.generate_image()`).
- **Function Calling & Agents**: Async and sync tool execution, few-shot prompt chaining, and automatic error recovery.
- **Streaming & Typed Overloads**: Full `@overload` typing for IDE autocomplete in both streaming and unary modes.
- **MIME & Capability Validation**: Automatic validation of supported MIME types, token boundaries, and thinking budgets.
- **MCP Integration**: Protocol-based duck-typing (`MCPSessionProtocol`) for Model Context Protocol servers without heavy external dependencies.

## Installation

Install the package directly from source:

```bash
pip install .
```

## Quick Start

### 1. Basic Agent with Tools & Few-Shot Prompting

```python
import asyncio
from anti_client import Client, Agent, Tool, Message


def get_weather(city: str) -> str:
    """Retrieve weather information for a specific city."""
    return f"The weather in {city} is clear and sunny."


weather_tool = Tool(
    name="get_weather",
    description="Retrieve current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    func=get_weather,
)


async def main():
    async with Client() as client:
        agent = Agent(
            client=client,
            model="gemini-3.1-pro-low",
            system_prompt="You are a helpful assistant.",
            tools=[weather_tool],
        )

        # Few-shot initialization or simple string prompt
        response = await agent.run("What's the weather in Tokyo?")
        print("Assistant:", response.text)
        print("Tokens Used:", response.usage.total_tokens)


if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Web Search Grounding

```python
async with Client() as client:
    result = await client.search("Latest news about Python 3.14")
    print("Answer:", result.text)
    for src in result.sources:
        print(f"- {src.title}: {src.uri}")
```

### 3. Native Image Generation

```python
async with Client() as client:
    result = await client.generate_image("A futuristic city skyline at sunset", aspect_ratio="16:9")
    if result.image:
        result.image.save("city.png")
        print("Image saved to city.png")
```

### 4. Checking Family Quotas

```python
async with Client() as client:
    quota = await client.get_quota_summary()
    if quota.gemini:
        print(f"Gemini Weekly Remaining: {quota.gemini.weekly.remaining_fraction * 100:.1f}%")
        print(f"Gemini 5h Remaining: {quota.gemini.five_hour.remaining_fraction * 100:.1f}%")
    if quota.claude:
        print(f"Claude Weekly Remaining: {quota.claude.weekly.remaining_fraction * 100:.1f}%")
        print(f"Claude 5h Remaining: {quota.claude.five_hour.remaining_fraction * 100:.1f}%")
```

### 5. Authentication, Project Storage & Multi-Accounts

#### Save Credentials to Current Project
```python
from anti_client import authenticate

# Saves credentials to ./.anti-api/accounts.json in your current project
authenticate(save_to_project=True)

# Or save to a custom file
authenticate(credentials_path="./credentials.json")
```

#### Manual API Key & Token Passing
```python
from anti_client import Client

# Pass API key directly (no accounts.json needed)
async with Client(api_key="your_access_token") as client:
    resp = await client.generate("gemini-3.1-pro-low", "Hello world")
    print(resp.text)

# Pass full credentials with refresh token and email
async with Client(
    api_key="access_token",
    refresh_token="refresh_token",
    email="dev@company.com",
    project_id="my-project-id",
) as client:
    ...
```

#### Managing Multiple Accounts
```python
from anti_client import Client, list_accounts, remove_account, set_active_account

# List all configured accounts
accounts = list_accounts()
for acc in accounts:
    print(f"Account: {acc.get('email')} (Project: {acc.get('projectId')})")

# Initialize client with a specific account by email
async with Client(account_email="work@company.com") as client:
    ...

# Or initialize by index
async with Client(account_index=1) as client:
    ...
```

#### Logout & Token Revocation
```python
from anti_client import Client, logout

# Logout active or specific account (removes from file and revokes Google OAuth token)
logout(email="work@company.com")

# Logout all accounts
logout(all_accounts=True)

# Or logout directly via client instance
async with Client() as client:
    await client.logout(revoke=True)
```


