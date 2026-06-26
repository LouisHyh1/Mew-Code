"""OpenAI Chat Completions API adapter with streaming and tool calls."""

import asyncio
from collections.abc import AsyncIterator

from novacode.config import ProviderConfig
from novacode.llm import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    Request,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    Usage,
)


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

    async def stream(self, req: Request) -> "AsyncIterator[StreamEvent]":
        messages = self._to_openai_messages(req)
        params: dict = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if req.tools:
            params["tools"] = self._to_openai_tools(req.tools)

        try:
            s = await self._client.chat.completions.create(**params)
            tool_calls_buf: dict[int, dict[str, str]] = {}
            finish_reason = None
            async for chunk in s:
                # 末尾 usage chunk（choices 空，带 chunk.usage）
                if not chunk.choices:
                    if chunk.usage is not None:
                        # 解析缓存命中（OpenAI 自动前缀缓存，仅读取）
                        cache_read = (
                            getattr(
                                getattr(chunk.usage, "prompt_tokens_details", None),
                                "cached_tokens",
                                0,
                            )
                            or 0
                        )
                        yield StreamEvent(
                            usage=Usage(
                                input_tokens=chunk.usage.prompt_tokens,
                                output_tokens=chunk.usage.completion_tokens,
                                cache_write=0,
                                cache_read=cache_read,
                            )
                        )
                    continue
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason or finish_reason
                if delta.content:
                    yield StreamEvent(text=delta.content)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_buf:
                            tool_calls_buf[idx] = {"id": "", "name": "", "args": ""}
                        buf = tool_calls_buf[idx]
                        if tc_delta.id:
                            buf["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            buf["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            buf["args"] = buf["args"] + tc_delta.function.arguments
            if finish_reason == "tool_calls" or tool_calls_buf:
                calls = []
                for idx in sorted(tool_calls_buf):
                    v = tool_calls_buf[idx]
                    calls.append(
                        ToolCall(
                            id=v["id"],
                            name=v["name"],
                            input=v.get("args") or "{}",
                        )
                    )
                if calls:
                    yield StreamEvent(tool_calls=calls)
            yield StreamEvent(done=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield StreamEvent(err=e)

    def _to_openai_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    def _to_openai_messages(self, req: Request) -> list[dict]:
        """构造 OpenAI 消息列表。

        - 首条 system 消息 = stable + environment 拼接（stable 居前）
        - 随即映射对话历史
        - reminder 非空时追加尾部 user 消息
        """
        # 系统消息（stable 在前，兼容端点前缀缓存）
        system_text = req.system.stable
        if req.system.environment:
            if system_text:
                system_text = system_text + "\n\n" + req.system.environment
            else:
                system_text = req.system.environment
        result: list[dict] = [{"role": "system", "content": system_text}]

        # 对话历史
        for m in req.messages:
            if m.role == ROLE_USER:
                result.append({"role": "user", "content": m.content})
            elif m.role == ROLE_ASSISTANT:
                if m.tool_calls:
                    tc_list = []
                    for c in m.tool_calls:
                        tc_list.append(
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": c.input or "{}",
                                },
                            }
                        )
                    result.append(
                        {
                            "role": "assistant",
                            "content": m.content or None,
                            "tool_calls": tc_list,
                        }
                    )
                else:
                    result.append({"role": "assistant", "content": m.content})
            elif m.role == ROLE_TOOL:
                for r in m.tool_results:
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": r.tool_call_id,
                            "content": r.content,
                        }
                    )

        # 补充消息注入（尾部 user 消息，OpenAI 容忍连续 user）
        if req.reminder:
            result.append({"role": "user", "content": req.reminder})

        return result
