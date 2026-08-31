import json
from unittest.mock import AsyncMock, patch

import pytest

from anti_client.agent import Agent
from anti_client.client import Client
from anti_client.types import (
    ChatResponse,
    Message,
    Tool,
    ToolCall,
)


def test_safe_truncate_heavy_interleaved_tool_blocks():
    # Construct a complex conversation with 10 interleaved turns:
    # Sys, User1, Asst1, User2, [AsstTool1, T1, T2], Asst2, User3, [AsstTool2, T3], Asst3, User4, [AsstTool3, T4, T5, T6], Asst4
    sys = Message(role="system", content="Sys")
    u1 = Message(role="user", content="u1")
    a1 = Message(role="assistant", content="a1")
    u2 = Message(role="user", content="u2")
    at1 = Message(
        role="assistant",
        tool_calls=[
            ToolCall(id="1", name="t1", arguments={}),
            ToolCall(id="2", name="t2", arguments={}),
        ],
    )
    t1 = Message(role="tool", name="t1", tool_call_id="1", content="res1")
    t2 = Message(role="tool", name="t2", tool_call_id="2", content="res2")
    a2 = Message(role="assistant", content="a2")
    u3 = Message(role="user", content="u3")
    at2 = Message(role="assistant", tool_calls=[ToolCall(id="3", name="t3", arguments={})])
    t3 = Message(role="tool", name="t3", tool_call_id="3", content="res3")
    a3 = Message(role="assistant", content="a3")
    u4 = Message(role="user", content="u4")
    at3 = Message(
        role="assistant",
        tool_calls=[
            ToolCall(id="4", name="t4", arguments={}),
            ToolCall(id="5", name="t5", arguments={}),
            ToolCall(id="6", name="t6", arguments={}),
        ],
    )
    t4 = Message(role="tool", name="t4", tool_call_id="4", content="res4")
    t5 = Message(role="tool", name="t5", tool_call_id="5", content="res5")
    t6 = Message(role="tool", name="t6", tool_call_id="6", content="res6")
    a4 = Message(role="assistant", content="a4")

    history = [sys, u1, a1, u2, at1, t1, t2, a2, u3, at2, t3, a3, u4, at3, t4, t5, t6, a4]
    # Total messages = 18

    # Test truncating to lengths 1..18 and verify invariants
    for max_l in range(1, 20):
        trunc = Agent._safe_truncate(history, max_l)
        assert len(trunc) <= max_l
        if trunc:
            assert trunc[0] == sys

        for idx, m in enumerate(trunc):
            if m.role == "tool":
                assert idx > 0
                prev = trunc[idx - 1]
                assert prev.role in ("tool", "assistant")
                if prev.role == "assistant":
                    assert prev.tool_calls is not None
            if m.role == "assistant" and m.tool_calls and idx + 1 < len(trunc):
                assert trunc[idx + 1].role == "tool"


@pytest.mark.asyncio
async def test_agent_history_rollback_on_generation_failure():
    client = Client(api_key="test_tok", project_id="proj")
    agent = Agent(client=client, model="gemini-3.1-pro-low")
    agent.history.append(Message(role="user", content="Initial message"))
    initial_len = len(agent.history)

    # Generation throws unexpected connection error
    with patch.object(
        client, "generate", new_callable=AsyncMock, side_effect=RuntimeError("Fatal network crash")
    ):
        with pytest.raises(RuntimeError, match="Fatal network crash"):
            await agent.run("New user prompt that fails")

        # History MUST be restored back to initial length
        assert len(agent.history) == initial_len
        assert agent.history[-1].content == "Initial message"


@pytest.mark.asyncio
async def test_agent_tool_returning_non_serializable_and_none():
    client = Client(api_key="test_tok", project_id="proj")

    def return_none():
        return None

    def return_custom_object():
        class CustomObj:
            def __str__(self):
                return "CustomStringRepresentation"

        return CustomObj()

    tools = [
        Tool(name="none_tool", description="returns None", parameters={}, func=return_none),
        Tool(
            name="obj_tool",
            description="returns CustomObj",
            parameters={},
            func=return_custom_object,
        ),
    ]

    agent = Agent(client=client, model="gemini-3.1-pro-low", tools=tools)

    step1_resp = ChatResponse(
        tool_calls=[
            ToolCall(id="c1", name="none_tool", arguments={}),
            ToolCall(id="c2", name="obj_tool", arguments={}),
        ]
    )
    step2_resp = ChatResponse(text="Final answer")

    with patch.object(
        client, "generate", new_callable=AsyncMock, side_effect=[step1_resp, step2_resp]
    ):
        res = await agent.run("Execute tools")
        assert res.text == "Final answer"

        tool_msgs = [m for m in agent.history if m.role == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0].content == "None"
        assert "CustomStringRepresentation" in tool_msgs[1].content


@pytest.mark.asyncio
async def test_agent_loop_detection_multiple_steps():
    client = Client(api_key="test_tok", project_id="proj")

    def broken_func(arg: str):
        raise ValueError("Invalid arg")

    tools = [Tool(name="broken_tool", description="Fails", parameters={}, func=broken_func)]
    agent = Agent(client=client, model="gemini-3.1-pro-low", tools=tools)

    # 3 consecutive failures with same arg
    tc1 = ChatResponse(tool_calls=[ToolCall(id="1", name="broken_tool", arguments={"arg": "x"})])
    tc2 = ChatResponse(tool_calls=[ToolCall(id="2", name="broken_tool", arguments={"arg": "x"})])
    tc3 = ChatResponse(tool_calls=[ToolCall(id="3", name="broken_tool", arguments={"arg": "x"})])
    final_resp = ChatResponse(text="Self corrected")

    with patch.object(
        client, "generate", new_callable=AsyncMock, side_effect=[tc1, tc2, tc3, final_resp]
    ):
        res = await agent.run("Trigger loop", max_steps=5)
        assert res.text == "Self corrected"

        tool_msgs = [m for m in agent.history if m.role == "tool"]
        assert len(tool_msgs) == 3
        # Step 1: no hint
        assert "hint" not in json.loads(tool_msgs[0].content)
        # Step 2: hint (2 times)
        assert "failed 2 times" in json.loads(tool_msgs[1].content)["hint"]
        # Step 3: hint (3 times)
        assert "failed 3 times" in json.loads(tool_msgs[2].content)["hint"]
