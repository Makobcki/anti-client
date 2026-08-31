import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from anti_client.agent import Agent
from anti_client.client import Client
from anti_client.exceptions import AgentMaxStepsError
from anti_client.types import (
    ChatResponse,
    CountTokensResult,
    GenerateOptions,
    Message,
    StreamChunk,
    Tool,
    ToolCall,
    UsageStats,
)


@pytest.mark.asyncio
async def test_agent_few_shot_messages():
    client = Client(api_key="test_token", project_id="aicode-consumers")

    agent = Agent(
        client=client,
        model="gemini-3.1-pro-low",
        system_prompt="You are a translation assistant.",
    )

    few_shot = [
        Message(role="user", content="Hello in French?"),
        Message(role="assistant", content="Bonjour"),
        Message(role="user", content="Good morning in French?"),
    ]

    mock_resp = ChatResponse(
        text="Bonjour / Bon matin",
        usage=UsageStats(prompt_tokens=25, completion_tokens=5, total_tokens=30),
    )

    with patch.object(client, "generate", new_callable=AsyncMock, return_value=mock_resp):
        res = await agent.run(few_shot)

        assert res.text == "Bonjour / Bon matin"
        # System prompt + 3 few shot messages + assistant answer
        assert len(agent.history) == 5
        assert agent.history[0].role == "system"
        assert agent.history[1].content == "Hello in French?"
        assert agent.history[2].content == "Bonjour"
        assert agent.history[3].content == "Good morning in French?"
        assert agent.history[4].content == "Bonjour / Bon matin"


@pytest.mark.asyncio
async def test_agent_async_and_sync_tool_execution():
    client = Client(api_key="test_token", project_id="aicode-consumers")

    def sync_tool(x: int) -> int:
        return x * 2

    async def async_tool(msg: str) -> str:
        return f"echo: {msg}"

    tools = [
        Tool(
            name="multiply",
            description="Multiplies by 2",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            func=sync_tool,
        ),
        Tool(
            name="echo",
            description="Echoes message",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
            func=async_tool,
        ),
    ]

    agent = Agent(client=client, model="gemini-3.1-pro-low", tools=tools)

    step1_resp = ChatResponse(
        tool_calls=[
            ToolCall(id="c1", name="multiply", arguments={"x": 21}),
            ToolCall(id="c2", name="echo", arguments={"msg": "hello"}),
        ]
    )
    step2_resp = ChatResponse(text="The result is 42 and echo: hello")

    with patch.object(
        client, "generate", new_callable=AsyncMock, side_effect=[step1_resp, step2_resp]
    ):
        res = await agent.run("Run both tools", options=GenerateOptions(max_steps=3))
        assert res.text == "The result is 42 and echo: hello"

        # Check tool responses in history
        tool_msgs = [m for m in agent.history if m.role == "tool"]
        assert len(tool_msgs) == 2
        assert "42" in tool_msgs[0].content
        assert "echo: hello" in tool_msgs[1].content


@pytest.mark.asyncio
async def test_agent_count_tokens():
    client = Client(api_key="test_token", project_id="aicode-consumers")

    def my_tool(x: int) -> int:
        return x

    tools = [Tool(name="test_tool", description="Test", parameters={}, func=my_tool)]

    agent = Agent(
        client=client,
        model="gemini-3.1-pro-low",
        system_prompt="You are a helpful assistant.",
        tools=tools,
    )
    agent.history.append(Message(role="user", content="Previous message"))
    agent.history.append(Message(role="assistant", content="Previous answer"))

    with patch.object(
        client,
        "count_tokens",
        new_callable=AsyncMock,
        return_value=CountTokensResult(total_tokens=55),
    ) as mock_count:
        res = await agent.count_tokens("Next user question")

        assert res == 55
        assert int(res) == 55
        mock_count.assert_called_once()
        call_kwargs = mock_count.call_args.kwargs
        messages = call_kwargs["messages"]
        # History (system + user + assistant) + next question = 4 messages
        assert len(messages) == 4
        assert messages[-1].content == "Next user question"
        assert call_kwargs["tools"] == tools


def test_safe_truncate_preserves_atomic_tool_blocks():
    system_msg = Message(role="system", content="System instruction")
    user1 = Message(role="user", content="Old question 1")
    asst1 = Message(role="assistant", content="Old answer 1")
    user2 = Message(role="user", content="Calculate sum")
    asst_tool = Message(
        role="assistant",
        tool_calls=[
            ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}),
            ToolCall(id="c2", name="sub", arguments={"a": 5, "b": 3}),
        ],
    )
    tool1 = Message(role="tool", name="add", tool_call_id="c1", content="3")
    tool2 = Message(role="tool", name="sub", tool_call_id="c2", content="2")
    asst2 = Message(role="assistant", content="Results are 3 and 2")

    history = [system_msg, user1, asst1, user2, asst_tool, tool1, tool2, asst2]

    # Truncate to max_length=5
    # Non-system messages total 7.
    # Unit 1: [user1] (1)
    # Unit 2: [asst1] (1)
    # Unit 3: [user2] (1)
    # Unit 4: [asst_tool, tool1, tool2] (3) - ATOMIC
    # Unit 5: [asst2] (1)
    # allowed non-system = 5 - 1 = 4.
    # From the end:
    # Unit 5 (1) -> count=1
    # Unit 4 (3) -> count=4 (1+3 <= 4)
    # Total kept = system_msg + Unit 4 + Unit 5 = 5 messages!
    truncated = Agent._safe_truncate(history, max_length=5)

    assert len(truncated) == 5
    assert truncated[0] == system_msg
    assert truncated[1] == asst_tool
    assert truncated[2] == tool1
    assert truncated[3] == tool2
    assert truncated[4] == asst2

    # Verify that tool pair is NEVER broken when max_length is 3
    # allowed non-system = 2. Unit 5 (1) fits, Unit 4 (3) does not fit (1+3 > 2).
    truncated_short = Agent._safe_truncate(history, max_length=3)
    assert len(truncated_short) == 2
    assert truncated_short[0] == system_msg
    assert truncated_short[1] == asst2


@pytest.mark.asyncio
async def test_tool_loop_detection_hint():
    client = Client(api_key="test_token", project_id="aicode-consumers")

    def failing_tool(x: int):
        raise ValueError("Invalid parameter value")

    tools = [
        Tool(
            name="bad_tool",
            description="Always fails",
            parameters={"type": "object"},
            func=failing_tool,
        )
    ]
    agent = Agent(client=client, model="gemini-3.1-pro-low", tools=tools)

    step1_resp = ChatResponse(tool_calls=[ToolCall(id="c1", name="bad_tool", arguments={"x": 10})])
    step2_resp = ChatResponse(tool_calls=[ToolCall(id="c2", name="bad_tool", arguments={"x": 10})])
    step3_resp = ChatResponse(text="Self-corrected without repeating failing tool")

    with patch.object(
        client, "generate", new_callable=AsyncMock, side_effect=[step1_resp, step2_resp, step3_resp]
    ):
        res = await agent.run("Run tool", options=GenerateOptions(max_steps=4))
        assert res.text == "Self-corrected without repeating failing tool"

        tool_msgs = [m for m in agent.history if m.role == "tool"]
        assert len(tool_msgs) == 2
        # First failure has no hint
        assert "hint" not in json.loads(tool_msgs[0].content)
        # Second identical failure has guidance hint
        second_err = json.loads(tool_msgs[1].content)
        assert "hint" in second_err
        assert "failed 2 times with identical arguments" in second_err["hint"]


@pytest.mark.asyncio
async def test_tool_custom_timeout():
    client = Client(api_key="test_token", project_id="aicode-consumers")

    async def slow_func(delay: float):
        await asyncio.sleep(delay)
        return "finished"

    tool = Tool(
        name="slow_tool",
        description="Takes long time",
        parameters={"type": "object"},
        func=slow_func,
        timeout=0.05,  # 50ms timeout
    )
    agent = Agent(client=client, model="gemini-3.1-pro-low", tools=[tool])

    step1_resp = ChatResponse(
        tool_calls=[ToolCall(id="c1", name="slow_tool", arguments={"delay": 1.0})]
    )
    step2_resp = ChatResponse(text="Recovered from timeout")

    with patch.object(
        client, "generate", new_callable=AsyncMock, side_effect=[step1_resp, step2_resp]
    ):
        res = await agent.run("Run slow tool")
        assert res.text == "Recovered from timeout"

        tool_msgs = [m for m in agent.history if m.role == "tool"]
        assert len(tool_msgs) == 1
        data = json.loads(tool_msgs[0].content)
        assert "timed out after 0.05 seconds" in data["error"]


@pytest.mark.asyncio
async def test_on_max_steps_handling():
    client = Client(api_key="test_token", project_id="aicode-consumers")

    def dummy_tool(x: int):
        return x

    tool = Tool(name="loop_tool", description="Dummy", parameters={}, func=dummy_tool)
    agent = Agent(client=client, model="gemini-3.1-pro-low", tools=[tool])

    repeating_resp = ChatResponse(
        tool_calls=[ToolCall(id="c1", name="loop_tool", arguments={"x": 1})]
    )

    # 1. Test on_max_steps="raise" (default)
    with patch.object(client, "generate", new_callable=AsyncMock, return_value=repeating_resp):
        with pytest.raises(AgentMaxStepsError) as exc_info:
            await agent.run("Infinite loop test", max_steps=2)
        assert exc_info.value.steps_taken == 2
        assert exc_info.value.last_response == repeating_resp

    # 2. Test on_max_steps="return_last"
    with patch.object(client, "generate", new_callable=AsyncMock, return_value=repeating_resp):
        res = await agent.run(
            "Infinite loop test", max_steps=2, options=GenerateOptions(on_max_steps="return_last")
        )
        assert res == repeating_resp


def test_clear_memory_with_reset_usage():
    client = Client(api_key="test_token", project_id="aicode-consumers")
    agent = Agent(client=client, model="gemini-3.1-pro-low", system_prompt="Sys")
    agent.total_usage = UsageStats(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    agent.last_usage = UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    agent.history.append(Message(role="user", content="Hi"))

    # Normal clear preserves total_usage
    agent.clear_memory(reset_usage=False)
    assert len(agent.history) == 1  # System prompt
    assert agent.total_usage.total_tokens == 150

    # Reset usage clear zeroes total_usage
    agent.clear_memory(reset_usage=True)
    assert agent.total_usage.total_tokens == 0
    assert agent.last_usage is None


@pytest.mark.asyncio
async def test_agent_streaming_usage_tracking():
    client = Client(api_key="test_token", project_id="aicode-consumers")
    agent = Agent(client=client, model="gemini-3.1-pro-low")

    async def mock_stream_gen(*args, **kwargs):
        yield StreamChunk(text="Hello ")
        yield StreamChunk(
            text="world!",
            usage=UsageStats(
                prompt_tokens=12,
                completion_tokens=4,
                total_tokens=16,
                cached_tokens=2,
                thoughts_tokens=0,
            ),
        )

    with patch.object(client, "generate", side_effect=mock_stream_gen):
        stream_iter = await agent.run("Hi", stream=True)
        chunks = [c async for c in stream_iter]

        assert len(chunks) == 2
        assert agent.last_usage is not None
        assert agent.last_usage.prompt_tokens == 12
        assert agent.last_usage.completion_tokens == 4
        assert agent.total_usage.total_tokens == 16
        assert agent.total_usage.cached_tokens == 2
