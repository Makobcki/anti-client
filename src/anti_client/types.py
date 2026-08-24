from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolCall:
    """Represents a tool call requested by the model.
    Attributes:
        id (str): A unique identifier for the tool call.
        name (str): The name of the tool/function to call.
        arguments (Dict[str, Any]): The arguments to pass to the tool.
        thought_signature (Optional[str]): An optional signature representing the model's reasoning process.
    """

    id: str
    name: str
    arguments: Dict[str, Any]
    thought_signature: Optional[str] = None


@dataclass
class Tool:
    """Represents an executable tool that the agent can use.
    Attributes:
        name (str): The name of the tool.
        description (str): A description of what the tool does.
        parameters (Dict[str, Any]): A JSON schema describing the tool's parameters.
        func (Optional[Callable]): The actual Python function to execute.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Optional[Callable] = None

    @classmethod
    def from_mcp(cls, mcp_tool: Any, session: Any) -> "Tool":
        """Creates a Tool instance from an MCP (Model Context Protocol) tool.

        Args:
            mcp_tool (Any): An MCP Tool object (e.g., from session.list_tools()).
            session (Any): An active MCP ClientSession used to execute the tool.

        Returns:
            Tool: A Tool instance configured to invoke the MCP tool via the session.
        """
        name = getattr(mcp_tool, "name", str(mcp_tool))
        description = getattr(mcp_tool, "description", "") or ""
        parameters = (
            getattr(mcp_tool, "inputSchema", None)
            or getattr(mcp_tool, "parameters", None)
            or {"type": "object", "properties": {}}
        )

        async def mcp_func(**kwargs) -> Any:
            result = await session.call_tool(name, arguments=kwargs)
            if hasattr(result, "content"):
                contents = []
                for item in result.content:
                    if hasattr(item, "text"):
                        contents.append(item.text)
                    else:
                        contents.append(str(item))
                return "\n".join(contents)
            return str(result)

        return cls(
            name=name,
            description=description,
            parameters=parameters,
            func=mcp_func,
        )


@dataclass
class FileAttachment:
    """Represents a file attachment to be sent with a message.
    Attributes:
        mime_type (str): The MIME type of the file (e.g., 'image/png').
        data (str): The base64-encoded string of the file's content.
    """

    mime_type: str
    data: str


@dataclass
class Message:
    """Represents a message in the conversation history.
    Attributes:
        role (str): The role of the message sender (e.g., 'user', 'assistant', 'system', 'tool').
        content (Optional[str]): The text content of the message.
        tool_calls (Optional[List[ToolCall]]): A list of tool calls initiated in this message.
        tool_call_id (Optional[str]): The ID of the tool call this message is responding to (if role is 'tool').
        attachments (Optional[List[FileAttachment]]): A list of file attachments to include with the message.
    """

    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    thought: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    attachments: Optional[List[FileAttachment]] = None


@dataclass
class UsageStats:
    """Holds token usage statistics for a generation request.
    Attributes:
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens generated in the response.
        total_tokens (int): The total number of tokens used.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatResponse:
    """Represents the final response from a chat generation request.
    Attributes:
        text (Optional[str]): The generated text content.
        usage (UsageStats): The token usage statistics.
        finish_reason (str): The reason the generation finished (e.g., 'STOP', 'MAX_TOKENS').
        tool_calls (Optional[List[ToolCall]]): Any tool calls requested by the model.
    """

    text: Optional[str] = None
    thought: Optional[str] = None
    usage: Optional[UsageStats] = None
    finish_reason: str = "stop"
    tool_calls: Optional[List[ToolCall]] = None


@dataclass
class StreamChunk:
    """Represents a chunk emitted during streaming generation.
    Attributes:
        text (Optional[str]): The generated text content chunk.
        thought (Optional[str]): The generated thinking/reasoning chunk.
        tool_calls (Optional[List[ToolCall]]): Any tool calls in this chunk.
    """

    text: Optional[str] = None
    thought: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    @property
    def is_thought(self) -> bool:
        return self.thought is not None

    @property
    def is_text(self) -> bool:
        return self.text is not None

    @property
    def is_tool_call(self) -> bool:
        return self.tool_calls is not None

    def __str__(self) -> str:
        if self.text is not None:
            return self.text
        if self.thought is not None:
            return self.thought
        return ""


@dataclass
class QuotaInfo:
    """Contains quota information for a specific model.
    Attributes:
        remaining_fraction (float): The remaining fraction of the quota.
        reset_time (Optional[str]): The timestamp or description of when the quota resets.
    """

    remaining_fraction: float
    reset_time: Optional[str]


@dataclass
class ModelInfo:
    """Contains metadata and configuration for an available model.
    Attributes:
        id (str): The identifier of the model.
        internal_model_id (str): The internal identifier used by the API.
        display_name (str): The human-readable name of the model.
        model_provider (str): The provider of the model.
        api_provider (str): The API provider backend.
        max_tokens (int): The maximum number of total tokens supported.
        max_output_tokens (Optional[int]): The maximum number of tokens the model can generate.
        is_internal (bool): Whether the model is an internal-only model.
        supports_images (bool): Whether the model supports image inputs.
        supports_thinking (bool): Whether the model supports reasoning/thinking capabilities.
        quota_info (Optional[QuotaInfo]): The quota information for this model, if available.
    """

    id: str
    internal_model_id: str
    display_name: str
    model_provider: str
    api_provider: str
    max_tokens: int
    max_output_tokens: Optional[int]
    is_internal: bool
    supports_images: bool
    supports_thinking: bool
    quota_info: Optional[QuotaInfo]
