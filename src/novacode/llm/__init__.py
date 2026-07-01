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
    is_policy_denial: bool = False  # 是否为策略级硬拒绝（如 Plan 模式拒绝写入）


@dataclass
class ToolDefinition:
    """注册中心导出的协议无关工具定义。"""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Usage:
    """本轮输入/输出 token 数，含缓存命中信息。

    cache_write: Anthropic cache_creation_input_tokens；OpenAI 恒 0（自动缓存无写计数）。
    cache_read:  Anthropic cache_read_input_tokens；OpenAI prompt_tokens_details.cached_tokens。
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write: int = 0
    cache_read: int = 0


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


@dataclass
class System:
    """系统提示——分为可缓存稳定块与不进缓存的环境块。"""

    stable: str = ""
    environment: str = ""


@dataclass
class Request:
    """一次 LLM 请求的全部入参。

    messages:  持久对话历史（不含本轮 reminder）。
    tools:     本轮工具集（普通=全量 / 规划=只读）。
    system:    系统提示（stable 可缓存 + environment 不缓存）。
    reminder:  本轮 system-reminder 内容（已含标签；空=不注入）。
    """

    messages: list[Message] = field(default_factory=list)
    tools: list[ToolDefinition] = field(default_factory=list)
    system: System = field(default_factory=System)
    reminder: str = ""


class Provider(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def model(self) -> str: ...
    def stream(self, req: Request) -> AsyncIterator[StreamEvent]: ...


def new_provider(cfg: ProviderConfig) -> "Provider":
    if cfg.protocol == "anthropic":
        from novacode.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg)
    if cfg.protocol == "openai":
        from novacode.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    raise ValueError(f"Unknown protocol: {cfg.protocol}")
