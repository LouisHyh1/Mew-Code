"""Protocol-agnostic LLM interface types and factory."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from novacode.config import ProviderConfig

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


@dataclass
class ToolCall:
    """协议无关地承载模型发起的一次工具调用（流式拼接完成后）。"""

    id: str
    name: str
    input: str  # raw JSON string


@dataclass
class ToolResult:
    """协议无关地承载一次工具执行结果。"""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ToolDefinition:
    """注册中心导出的协议无关工具定义。"""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Usage:
    """本轮输入/输出 token 数。"""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class StreamEvent:
    """文本增量 / 工具调用 / token 用量 / 结束 / 错误。

    usage 非空：本轮 token 用量，done 之前一次性发出。
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: "Usage | None" = None
    done: bool = False
    err: Exception | None = None


class Provider(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def model(self) -> str: ...
    def stream(
        self,
        msgs: list[Message],
        tools: list[ToolDefinition],
        system_suffix: str = "",
    ) -> AsyncIterator[StreamEvent]: ...


def new_provider(cfg: ProviderConfig) -> "Provider":
    if cfg.protocol == "anthropic":
        from novacode.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg)
    if cfg.protocol == "openai":
        from novacode.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    raise ValueError(f"Unknown protocol: {cfg.protocol}")
