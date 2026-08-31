"""Autonomous Agent implementation for multi-turn conversation and tool execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
)

from .client import Client
from .exceptions import AgentMaxStepsError
from .types import (
    ChatResponse,
    CountTokensResult,
    FileAttachment,
    GenerateOptions,
    Message,
    StreamChunk,
    Tool,
    ToolCall,
    UsageStats,
)

if TYPE_CHECKING:
    from .mcp import MCPSessionProtocol


class Agent:
    """An agent that interacts with the AI model asynchronously using history, few-shot contexts, and tools."""

    def __init__(
        self,
        client: Client,
        model: str,
        name: str = "Assistant",
        system_prompt: str = "You are a helpful assistant.",
        tools: list[Tool] | None = None,
        tool_timeout: float = 90.0,
        on_max_steps: Literal["raise", "return_last"] = "raise",
        max_history_length: int | None = None,
    ):
        """Initializes the Agent.

        Args:
            client (Client): An authenticated Client instance.
            model (str): The name of the model to use (e.g., 'gemini-3.1-pro-low').
            name (str, optional): The name of the agent. Defaults to "Assistant".
            system_prompt (str, optional): The system instruction prompt. Defaults to "You are a helpful assistant.".
            tools (Optional[List[Tool]], optional): A list of tools the agent can use. Defaults to None.
            tool_timeout (float, optional): Default timeout for tool execution in seconds. Defaults to 90.0.
            on_max_steps (Literal["raise", "return_last"], optional): Behavior when max_steps is reached. Defaults to "raise".
            max_history_length (Optional[int], optional): Maximum history length to maintain automatically.
        """
        self.client = client
        self.model = model
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self._tool_map = {t.name: t for t in self.tools}
        self.tool_timeout = tool_timeout
        self.on_max_steps = on_max_steps
        self.max_history_length = max_history_length
        self.history: list[Message] = []
        if self.system_prompt:
            self.history.append(Message(role="system", content=self.system_prompt))
        self.last_usage: UsageStats | None = None
        self.total_usage: UsageStats = UsageStats(0, 0, 0, 0, 0)

    @overload
    async def run(
        self,
        prompt: str | list[Message] | Message,
        temperature: float = 0.8,
        stream: Literal[False] = False,
        max_steps: int = 5,
        attachments: list[FileAttachment] | None = None,
        options: GenerateOptions | None = None,
        **kwargs,
    ) -> ChatResponse: ...

    @overload
    async def run(
        self,
        prompt: str | list[Message] | Message,
        temperature: float = 0.8,
        stream: Literal[True] = True,
        max_steps: int = 5,
        attachments: list[FileAttachment] | None = None,
        options: GenerateOptions | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]: ...

    async def run(
        self,
        prompt: str | list[Message] | Message,
        temperature: float = 0.8,
        stream: bool = False,
        max_steps: int = 5,
        attachments: list[FileAttachment] | None = None,
        options: GenerateOptions | None = None,
        **kwargs,
    ) -> ChatResponse | AsyncIterator[StreamChunk]:
        """Runs the agent asynchronously with a prompt, few-shot message sequence, or single message.

        Args:
            prompt (Union[str, List[Message], Message]): The user input text, a message object, or a list of messages.
            temperature (float, optional): The sampling temperature. Defaults to 0.8.
            stream (bool, optional): If True, streams the response as an async iterator. Defaults to False.
            max_steps (int, optional): The maximum number of tool execution steps allowed. Defaults to 5.
            attachments (Optional[List[FileAttachment]], optional): Optional file attachments for a text prompt.
            options (Optional[GenerateOptions], optional): Structured generation options.
            **kwargs: Additional keyword arguments passed to client.generate.

        Returns:
            Union[ChatResponse, AsyncIterator[StreamChunk]]: Complete ChatResponse or StreamChunk iterator.
        """
        if self.max_history_length:
            self.truncate_history(self.max_history_length)

        initial_history_len = len(self.history)

        if isinstance(prompt, str):
            self.history.append(Message(role="user", content=prompt, attachments=attachments))
        elif isinstance(prompt, list):
            for m in prompt:
                self.history.append(m if isinstance(m, Message) else Message(**m))
        elif isinstance(prompt, Message):
            self.history.append(prompt)

        if options and options.max_steps:
            max_steps = options.max_steps

        effective_on_max_steps = (
            options.on_max_steps if options and options.on_max_steps else None
        ) or self.on_max_steps
        effective_tool_timeout = (
            options.tool_timeout if options and options.tool_timeout else self.tool_timeout
        )

        if stream:
            return self._run_stream(
                temperature=temperature,
                max_steps=max_steps,
                options=options,
                tool_timeout=effective_tool_timeout,
                on_max_steps=effective_on_max_steps,
                **kwargs,
            )

        failed_calls: dict[str, int] = {}
        last_response: ChatResponse | None = None

        try:
            for _step in range(max_steps):
                response: ChatResponse = await self.client.generate(
                    model=self.model,
                    messages=self.history,
                    temperature=temperature,
                    stream=False,
                    tools=self.tools,
                    options=options,
                    **kwargs,
                )
                last_response = response

                if response.usage:
                    self.last_usage = response.usage
                    self.total_usage.prompt_tokens += response.usage.prompt_tokens
                    self.total_usage.completion_tokens += response.usage.completion_tokens
                    self.total_usage.total_tokens += response.usage.total_tokens
                    self.total_usage.cached_tokens += response.usage.cached_tokens
                    self.total_usage.thoughts_tokens += response.usage.thoughts_tokens

                if response.text or response.thought or response.tool_calls:
                    self.history.append(
                        Message(
                            role="assistant",
                            content=response.text,
                            thought=response.thought,
                            thought_signature=response.thought_signature,
                            tool_calls=response.tool_calls,
                        )
                    )

                if not response.tool_calls:
                    if not self.history or self.history[-1].role != "assistant":
                        self.history.append(
                            Message(
                                role="assistant",
                                content=response.text or "",
                                thought=response.thought,
                                thought_signature=response.thought_signature,
                            )
                        )
                    return response

                for tool_call in response.tool_calls:
                    result = await self._execute_tool(
                        tool_call=tool_call,
                        timeout=effective_tool_timeout,
                        failed_calls=failed_calls,
                    )
                    self.history.append(
                        Message(
                            role="tool",
                            content=result,
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                    )

            if effective_on_max_steps == "return_last" and last_response:
                return last_response

            raise AgentMaxStepsError(
                f"Agent exceeded the maximum number of steps ({max_steps}) while executing tools.",
                last_response=last_response,
                steps_taken=max_steps,
            )
        except Exception:
            self.history = self.history[:initial_history_len]
            raise

    async def _execute_tool(
        self,
        tool_call: ToolCall,
        timeout: float | None = None,
        failed_calls: dict[str, int] | None = None,
    ) -> str:
        """Executes a requested tool call asynchronously with robust error trapping and loop detection."""
        tool = self._tool_map.get(tool_call.name)
        if not tool:
            return json.dumps({"error": f"Tool '{tool_call.name}' not found."})
        if not tool.func:
            return json.dumps({"error": f"Tool '{tool_call.name}' has no executable function."})

        call_key = (
            f"{tool_call.name}:{json.dumps(tool_call.arguments, sort_keys=True, default=str)}"
        )
        exec_timeout = tool.timeout or timeout or self.tool_timeout

        try:
            if asyncio.iscoroutinefunction(tool.func):
                result = await asyncio.wait_for(
                    tool.func(**tool_call.arguments), timeout=exec_timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.func, **tool_call.arguments),
                    timeout=exec_timeout,
                )
            return (
                json.dumps(result, ensure_ascii=False)
                if isinstance(result, (dict, list))
                else str(result)
            )
        except asyncio.TimeoutError:
            err_dict: dict[str, Any] = {
                "error": f"Tool '{tool_call.name}' timed out after {exec_timeout} seconds."
            }
            if failed_calls is not None:
                failed_calls[call_key] = failed_calls.get(call_key, 0) + 1
                if failed_calls[call_key] >= 2:
                    err_dict["hint"] = (
                        f"Tool '{tool_call.name}' has failed {failed_calls[call_key]} times with identical arguments. "
                        "Please adjust your arguments or respond directly to the user."
                    )
            return json.dumps(err_dict)
        except Exception as e:
            err_dict = {"error": f"Execution failed: {e!s}"}
            if failed_calls is not None:
                failed_calls[call_key] = failed_calls.get(call_key, 0) + 1
                if failed_calls[call_key] >= 2:
                    err_dict["hint"] = (
                        f"Tool '{tool_call.name}' has failed {failed_calls[call_key]} times with identical arguments. "
                        "Please adjust your arguments or respond directly to the user."
                    )
            return json.dumps(err_dict)

    async def _run_stream(
        self,
        temperature: float,
        max_steps: int,
        options: GenerateOptions | None = None,
        tool_timeout: float | None = None,
        on_max_steps: Literal["raise", "return_last"] = "raise",
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Runs the agent in streaming mode asynchronously."""
        failed_calls: dict[str, int] = {}
        for _step in range(max_steps):
            full_text = ""
            full_thought = ""
            thought_signature = None
            tool_calls: list[ToolCall] = []
            stream_response = await self.client.generate(
                model=self.model,
                messages=self.history,
                temperature=temperature,
                stream=True,
                tools=self.tools,
                options=options,
                **kwargs,
            )
            async for chunk in stream_response:
                if isinstance(chunk, StreamChunk):
                    if chunk.usage:
                        self.last_usage = chunk.usage
                        self.total_usage.prompt_tokens += chunk.usage.prompt_tokens
                        self.total_usage.completion_tokens += chunk.usage.completion_tokens
                        self.total_usage.total_tokens += chunk.usage.total_tokens
                        self.total_usage.cached_tokens += chunk.usage.cached_tokens
                        self.total_usage.thoughts_tokens += chunk.usage.thoughts_tokens
                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls
                        break
                    if chunk.thought_signature:
                        thought_signature = chunk.thought_signature
                    if chunk.thought:
                        full_thought += chunk.thought
                    if chunk.text:
                        full_text += chunk.text
                    yield chunk
                elif isinstance(chunk, list):
                    tool_calls = chunk
                    break
                elif isinstance(chunk, str):
                    full_text += chunk
                    yield StreamChunk(text=chunk)

            if full_text or full_thought or tool_calls:
                self.history.append(
                    Message(
                        role="assistant",
                        content=full_text if full_text else None,
                        thought=full_thought if full_thought else None,
                        thought_signature=thought_signature,
                        tool_calls=tool_calls if tool_calls else None,
                    )
                )

            if not tool_calls:
                if not self.history or self.history[-1].role != "assistant":
                    self.history.append(
                        Message(
                            role="assistant",
                            content=full_text if full_text else "",
                            thought=full_thought if full_thought else None,
                            thought_signature=thought_signature,
                        )
                    )
                return

            for tool_call in tool_calls:
                result = await self._execute_tool(
                    tool_call=tool_call,
                    timeout=tool_timeout,
                    failed_calls=failed_calls,
                )
                self.history.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
                )

        if on_max_steps == "return_last":
            return

        raise AgentMaxStepsError(
            f"Agent exceeded the maximum number of steps ({max_steps}) while executing tools.",
            steps_taken=max_steps,
        )

    async def add_mcp_tools(self, session: MCPSessionProtocol) -> list[Tool]:
        """Loads and adds tools from an MCP session to the agent.

        Args:
            session (MCPSessionProtocol): An active MCP ClientSession.

        Returns:
            List[Tool]: The list of newly added Tool objects.
        """
        from .mcp import load_mcp_tools

        mcp_tools = await load_mcp_tools(session)
        for tool in mcp_tools:
            self.tools.append(tool)
            self._tool_map[tool.name] = tool
        return mcp_tools

    async def count_tokens(
        self,
        prompt: str | list[Message] | Message | None = None,
        attachments: list[FileAttachment] | None = None,
    ) -> CountTokensResult:
        """Calculates the exact token count for the agent's current history, tools, and optional next prompt.

        Args:
            prompt (Optional[Union[str, List[Message], Message]], optional): Next prompt to evaluate. Defaults to None.
            attachments (Optional[List[FileAttachment]], optional): File attachments to evaluate. Defaults to None.

        Returns:
            CountTokensResult: Token count result object supporting integer arithmetic and comparisons.
        """
        test_history = list(self.history)
        if prompt:
            if isinstance(prompt, str):
                test_history.append(Message(role="user", content=prompt, attachments=attachments))
            elif isinstance(prompt, list):
                for m in prompt:
                    test_history.append(m if isinstance(m, Message) else Message(**m))
            elif isinstance(prompt, Message):
                test_history.append(prompt)

        return await self.client.count_tokens(
            messages=test_history,
            model=self.model,
            tools=self.tools if self.tools else None,
            system_instruction=self.system_prompt
            if not any(m.role == "system" for m in test_history)
            else None,
        )

    def clear_memory(self, reset_usage: bool = False) -> None:
        """Clears the conversation history, retaining only the system prompt.

        Args:
            reset_usage (bool, optional): If True, resets the accumulated total_usage. Defaults to False.
        """
        self.history = []
        if self.system_prompt:
            self.history.append(Message(role="system", content=self.system_prompt))
        if reset_usage:
            self.total_usage = UsageStats(0, 0, 0, 0, 0)
            self.last_usage = None

    @staticmethod
    def _safe_truncate(messages: list[Message], max_length: int) -> list[Message]:
        """Truncates conversation history while preserving atomic tool transaction blocks.

        Prevents Google API 400 Bad Request errors by ensuring that (assistant with tool_calls)
        and all associated tool response messages are kept or discarded together as an indivisible unit.

        Args:
            messages (List[Message]): The list of messages to truncate.
            max_length (int): The maximum allowed number of messages.

        Returns:
            List[Message]: The safely truncated message list preserving system prompts and atomic tool pairs.
        """
        if len(messages) <= max_length:
            return messages

        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        # Group non_system messages into atomic conversational units
        units: list[list[Message]] = []
        i = 0
        n = len(non_system)
        while i < n:
            msg = non_system[i]
            if msg.role == "assistant" and msg.tool_calls:
                unit = [msg]
                i += 1
                while i < n and non_system[i].role == "tool":
                    unit.append(non_system[i])
                    i += 1
                units.append(unit)
            elif msg.role == "tool":
                # Orphaned tool response: attach to previous unit if exists, otherwise skip
                if units:
                    units[-1].append(msg)
                i += 1
            else:
                units.append([msg])
                i += 1

        allowed = max(0, max_length - len(system_msgs))
        kept_units: list[list[Message]] = []
        count = 0

        for unit in reversed(units):
            if count + len(unit) <= allowed:
                kept_units.append(unit)
                count += len(unit)
            else:
                break

        kept_non_system: list[Message] = []
        for unit in reversed(kept_units):
            kept_non_system.extend(unit)

        return system_msgs + kept_non_system

    def truncate_history(self, max_length: int = 100) -> None:
        """Truncates conversation history to a maximum length while preserving atomic tool pairs and system prompt.

        Args:
            max_length (int, optional): The target maximum history length. Defaults to 100.
        """
        self.history = self._safe_truncate(self.history, max_length)
