import asyncio
import json
from typing import Any, AsyncIterator, List, Literal, Optional, Union, overload

from .client import Client
from .exceptions import AgentAPIError
from .types import ChatResponse, Message, StreamChunk, Tool, ToolCall


class Agent:
    """An agent that interacts with the AI model asynchronously using history and tools."""

    def __init__(
        self,
        client: Client,
        model: str,
        name: str = "Assistant",
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[List[Tool]] = None,
    ):
        """Initializes the Agent.
        Args:
            client (Client): An authenticated Client instance.
            model (str): The name of the model to use (e.g., 'gemini-3.1-pro-low').
            name (str, optional): The name of the agent. Defaults to "Assistant".
            system_prompt (str, optional): The system instruction prompt. Defaults to "You are a helpful assistant.".
            tools (Optional[List[Tool]], optional): A list of tools the agent can use. Defaults to None.
        """
        self.client = client
        self.model = model
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self._tool_map = {t.name: t for t in self.tools}
        self.history: List[Message] = []
        if self.system_prompt:
            self.history.append(Message(role="system", content=self.system_prompt))

    @overload
    async def run(
        self,
        prompt: str,
        temperature: float = 0.8,
        stream: Literal[False] = False,
        max_steps: int = 5,
        **kwargs,
    ) -> ChatResponse: ...

    @overload
    async def run(
        self,
        prompt: str,
        temperature: float = 0.8,
        stream: Literal[True] = True,
        max_steps: int = 5,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]: ...

    async def run(
        self,
        prompt: str,
        temperature: float = 0.8,
        stream: bool = False,
        max_steps: int = 5,
        **kwargs,
    ) -> Union[ChatResponse, AsyncIterator[StreamChunk]]:
        """Runs the agent asynchronously with a new user prompt.
        Args:
            prompt (str): The user's input prompt.
            temperature (float, optional): The sampling temperature for generation. Defaults to 0.8.
            stream (bool, optional): If True, streams the response as an async iterator. Defaults to False.
            max_steps (int, optional): The maximum number of tool execution steps allowed. Defaults to 5.
            **kwargs: Additional keyword arguments passed to the client's generate method.
        Returns:
            Union[ChatResponse, AsyncIterator[StreamChunk]]: A ChatResponse object if stream is False,
                otherwise an async iterator yielding StreamChunks.
        Raises:
            AgentAPIError: If the maximum number of steps is exceeded while executing tools.
        """
        self.history.append(Message(role="user", content=prompt))

        if stream:
            return self._run_stream(temperature, max_steps, **kwargs)

        for step in range(max_steps):
            response: ChatResponse = await self.client.generate(
                model=self.model,
                messages=self.history,
                temperature=temperature,
                stream=False,
                tools=self.tools,
                **kwargs,
            )

            self.history.append(
                Message(
                    role="assistant",
                    content=response.text,
                    thought=response.thought,
                    tool_calls=response.tool_calls,
                )
            )

            if not response.tool_calls:
                return response

            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                self.history.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
                )

        raise AgentAPIError(
            f"Agent exceeded the maximum number of steps ({max_steps}) while executing tools."
        )

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Executes a requested tool call asynchronously.
        Args:
            tool_call (ToolCall): The tool call requested by the model.
        Returns:
            str: A JSON-encoded string containing the result or error.
        """
        tool = self._tool_map.get(tool_call.name)
        if not tool:
            return json.dumps({"error": f"Tool '{tool_call.name}' not found."})
        if not tool.func:
            return json.dumps(
                {"error": f"Tool '{tool_call.name}' has no executable function."}
            )
        try:
            if asyncio.iscoroutinefunction(tool.func):
                result = await tool.func(**tool_call.arguments)
            else:
                result = await asyncio.to_thread(tool.func, **tool_call.arguments)
            return (
                json.dumps(result, ensure_ascii=False)
                if isinstance(result, (dict, list))
                else str(result)
            )
        except Exception as e:
            return json.dumps({"error": f"Execution failed: {str(e)}"})

    async def _run_stream(
        self, temperature: float, max_steps: int, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Runs the agent in streaming mode asynchronously.
        Args:
            temperature (float): The sampling temperature for generation.
            max_steps (int): The maximum number of tool execution steps allowed.
            **kwargs: Additional keyword arguments passed to the client generation method.
        Yields:
            StreamChunk: Chunks of generated text, thoughts, or tool calls.
        Raises:
            AgentAPIError: If the maximum number of steps is exceeded while executing tools.
        """
        for step in range(max_steps):
            full_text = ""
            full_thought = ""
            tool_calls = []
            stream_response = await self.client.generate(
                model=self.model,
                messages=self.history,
                temperature=temperature,
                stream=True,
                tools=self.tools,
                **kwargs,
            )
            async for chunk in stream_response:
                if isinstance(chunk, StreamChunk):
                    if chunk.tool_calls:
                        tool_calls = chunk.tool_calls
                        break
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
            self.history.append(
                Message(
                    role="assistant",
                    content=full_text if full_text else None,
                    thought=full_thought if full_thought else None,
                    tool_calls=tool_calls if tool_calls else None,
                )
            )
            if not tool_calls:
                return
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                self.history.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
                )
        raise AgentAPIError(
            f"Agent exceeded the maximum number of steps ({max_steps}) while executing tools."
        )

    async def add_mcp_tools(self, session: Any) -> List[Tool]:
        """Loads and adds tools from an MCP (Model Context Protocol) session to the agent.
        Args:
            session (Any): An initialized MCP ClientSession instance.
        Returns:
            List[Tool]: The list of MCP tools that were added to the agent.
        """
        from .mcp import load_mcp_tools

        mcp_tools = await load_mcp_tools(session)
        for tool in mcp_tools:
            self.tools.append(tool)
            self._tool_map[tool.name] = tool
        return mcp_tools

    def clear_memory(self):
        """Clears the conversation history, retaining only the system prompt."""
        self.history = []
        if self.system_prompt:
            self.history.append(Message(role="system", content=self.system_prompt))
