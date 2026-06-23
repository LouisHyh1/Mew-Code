"""Anthropic Messages API adapter with streaming and tool use."""

import asyncio
import json
from collections.abc import AsyncIterator

from novacode.config import ProviderConfig
from novacode.llm import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    Message,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    Usage,
)
from novacode.prompt import SYSTEM_PROMPT


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

    async def stream(
        self, msgs: list[Message], tools: list[ToolDefinition], system_suffix: str = ""
    ) -> "AsyncIterator[StreamEvent]":

        messages = self._to_anthropic_messages(msgs)
        params: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "system": self._effective_system(system_suffix),
            "messages": messages,
        }
        if tools:
            params["tools"] = self._to_anthropic_tools(tools)
        if self._thinking and not self._has_tool_history(msgs):
            params["thinking"] = {"type": "enabled", "budget_tokens": 2048}

        try:
            async with self._client.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if getattr(event.delta, "type", None) == "text_delta":
                            yield StreamEvent(text=event.delta.text)

                final_message = await stream.get_final_message()
                # 用量：流正常结束后、done 之前一次性上抛
                if final_message.usage is not None:
                    yield StreamEvent(
                        usage=Usage(
                            input_tokens=final_message.usage.input_tokens,
                            output_tokens=final_message.usage.output_tokens,
                        )
                    )
                if final_message.stop_reason == "tool_use":
                    calls = []
                    for block in final_message.content:
                        if block.type == "tool_use":
                            calls.append(
                                ToolCall(
                                    id=block.id,
                                    name=block.name,
                                    input=json.dumps(block.input),
                                )
                            )
                    if calls:
                        yield StreamEvent(tool_calls=calls)
                yield StreamEvent(done=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield StreamEvent(err=e)

    def _effective_system(self, suffix: str) -> str:
        """系统提示拼接：suffix 非空时拼到 SYSTEM_PROMPT 之后。"""
        if suffix == "":
            return SYSTEM_PROMPT
        return SYSTEM_PROMPT + "\n\n" + suffix

    def _to_anthropic_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def _has_tool_history(self, msgs: list[Message]) -> bool:
        for m in msgs:
            if m.tool_calls or m.tool_results:
                return True
        return False

    def _to_anthropic_messages(self, msgs: list[Message]) -> list[dict]:
        result = []
        for m in msgs:
            if m.role == ROLE_USER:
                result.append({"role": "user", "content": m.content})
            elif m.role == ROLE_ASSISTANT:
                if m.tool_calls:
                    content: list[dict] = [{"type": "text", "text": m.content}]
                    for c in m.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": c.id,
                                "name": c.name,
                                "input": json.loads(c.input),
                            }
                        )
                    result.append({"role": "assistant", "content": content})
                else:
                    result.append({"role": "assistant", "content": m.content})
            elif m.role == ROLE_TOOL:
                content = []
                for r in m.tool_results:
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": r.tool_call_id,
                            "content": r.content,
                            "is_error": r.is_error,
                        }
                    )
                result.append({"role": "user", "content": content})
        return result
