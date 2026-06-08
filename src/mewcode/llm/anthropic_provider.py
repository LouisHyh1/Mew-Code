"""Anthropic Messages API adapter with streaming."""

import asyncio

from mewcode.config import ProviderConfig
from mewcode.llm import Message, StreamEvent
from mewcode.prompt import SYSTEM_PROMPT


class AnthropicProvider:
    def __init__(self, cfg: ProviderConfig) -> None:
        from anthropic import AsyncAnthropic

        self._name = cfg.name
        self._model = cfg.model
        self._thinking = cfg.thinking
        self._client = AsyncAnthropic(
            api_key=cfg.api_key,
            base_url=cfg.base_url or None,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def stream(self, msgs: list[Message]) -> "AsyncIterator[StreamEvent]":
        from anthropic import AsyncAnthropic

        messages = [{"role": m.role, "content": m.content} for m in msgs]
        params: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }
        if self._thinking:
            params["thinking"] = {"type": "enabled", "budget_tokens": 2048}

        try:
            async with self._client.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if getattr(event.delta, "type", None) == "text_delta":
                            yield StreamEvent(text=event.delta.text)
                    # thinking_delta events are received but discarded
            yield StreamEvent(done=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield StreamEvent(err=e)
