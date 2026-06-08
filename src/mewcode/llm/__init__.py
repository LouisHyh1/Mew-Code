"""Protocol-agnostic LLM interface types and factory."""

from dataclasses import dataclass
from typing import AsyncIterator, Literal, Protocol

from mewcode.config import ProviderConfig


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class StreamEvent:
    text: str = ""
    done: bool = False
    err: Exception | None = None


class Provider(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def model(self) -> str: ...
    def stream(self, msgs: list[Message]) -> AsyncIterator[StreamEvent]: ...


def new_provider(cfg: ProviderConfig) -> Provider:
    if cfg.protocol == "anthropic":
        from mewcode.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(cfg)
    if cfg.protocol == "openai":
        from mewcode.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(cfg)
    raise ValueError(f"Unknown protocol: {cfg.protocol}")
