"""OpenAI Chat Completions API adapter with streaming."""

import asyncio

from mewcode.config import ProviderConfig
from mewcode.llm import Message, StreamEvent
from mewcode.prompt import SYSTEM_PROMPT


class OpenAIProvider:
    def __init__(self, cfg: ProviderConfig) -> None:
        from openai import AsyncOpenAI

        self._name = cfg.name
        self._model = cfg.model
        self._client = AsyncOpenAI(
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
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        messages.extend({"role": m.role, "content": m.content} for m in msgs)

        try:
            s = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
            )
            async for chunk in s:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield StreamEvent(text=delta.content)
            yield StreamEvent(done=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield StreamEvent(err=e)
