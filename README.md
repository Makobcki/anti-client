# anti-client

A full-featured Python client library for interacting with the internal Anrigravity API (Gemini).

## Features

- **OAuth 2.0 Authentication**: Built-in authentication flow with token caching (`~/.anti-api/accounts.json`).
- **Function Calling**: Complete support for tool execution by agents.
- **Streaming**: Native streaming capabilities, seamlessly integrated with function calling.
- **Resource Management**: Automated model discovery, quota checking, and precise usage statistics.

## Installation

Install the package directly from the source directory:

```bash
pip install .
```

## Usage Example

```python
import asyncio
from anti_client import Client, Agent, Tool

# 1. Define Tools
def get_weather(city: str) -> str:
    """Retrieve weather information for a specific city."""
    return f"The weather in {city} is clear and sunny."

weather_tool = Tool(
    name="get_weather",
    description="Retrieve the current weather for a specified city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The name of the city."}
        },
        "required": ["city"],
    },
    func=get_weather
)

async def main():
    # 2. Initialize Client using an async context manager
    # (Triggers OAuth flow if tokens are missing)
    async with Client() as client:

        # 3. Initialize Agent
        agent = Agent(
            client=client,
            model="gemini-3.1-pro-low",
            system_prompt="You are a helpful assistant.",
            tools=[weather_tool]
        )

        # 4. Generate Response
        response = await agent.run("Hello! What's the weather like in London?", stream=False)

        print("Response:", response.text)
        print("Tokens Used:", response.usage.total_tokens)

if __name__ == "__main__":
    asyncio.run(main())

```
