"""Anthropic Messages API adapter with streaming, cache control, and tool use."""

import asyncio
import json
from collections.abc import AsyncIterator

from novacode.config import ProviderConfig
from novacode.llm import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    Message,
    Request,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    Usage,
)


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

    async def stream(self, req: Request) -> "AsyncIterator[StreamEvent]":
        # ── 构造 system 文本块（stable 带 cache_control 断点，env 不带）──
        system: list[dict] = []
        if req.system.stable:
            system.append(
                {
                    "type": "text",
                    "text": req.system.stable,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        if req.system.environment:
            system.append(
                {
                    "type": "text",
                    "text": req.system.environment,
                }
            )

        # ── 转换消息并注入 reminder ─────────────────────────────
        messages = self._to_anthropic_messages(req.messages)
        if req.reminder:
            self._inject_reminder_anthropic(messages, req.reminder)

        # ── 构造请求参数 ───────────────────────────────────────
        params: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
        }
        if req.tools:
            params["tools"] = self._to_anthropic_tools(req.tools)
        if self._thinking and not self._has_tool_history(req.messages):
            params["thinking"] = {"type": "enabled", "budget_tokens": 2048}

        try:
            async with self._client.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if getattr(event.delta, "type", None) == "text_delta":
                            yield StreamEvent(text=event.delta.text)

                final_message = await stream.get_final_message()
                # 用量：含缓存写/读字段
                if final_message.usage is not None:
                    yield StreamEvent(
                        usage=Usage(
                            input_tokens=final_message.usage.input_tokens,
                            output_tokens=final_message.usage.output_tokens,
                            cache_write=getattr(
                                final_message.usage,
                                "cache_creation_input_tokens",
                                0,
                            )
                            or 0,
                            cache_read=getattr(
                                final_message.usage,
                                "cache_read_input_tokens",
                                0,
                            )
                            or 0,
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

    # ── reminder 注入 ──────────────────────────────────────────

    @staticmethod
    def _inject_reminder_anthropic(messages: list[dict], reminder: str) -> None:
        """把 reminder 文本块注入到最后一条消息的 content 中。

        Anthropic 要求严格角色交替：末条恒为 user（tool_result 映射为 user），
        追加文本块到 content 列表不破坏角色交替（N3）。

        极端情形（空消息 / 末条非 user）则新起一条 user 消息防御。
        """
        if not messages:
            messages.append({"role": "user", "content": reminder})
            return
        last = messages[-1]
        if last["role"] != "user":
            messages.append({"role": "user", "content": reminder})
            return
        # 确保 content 为 list 形态
        content = last["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        content.append({"type": "text", "text": reminder})
        last["content"] = content

    # ── 工具格式转换 ──────────────────────────────────────────

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

    # ── 消息格式转换 ──────────────────────────────────────────

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
