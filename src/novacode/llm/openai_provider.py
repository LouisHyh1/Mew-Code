"""OpenAI Chat Completions API adapter with streaming and tool calls."""

import asyncio
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

    async def stream(
        self, msgs: list[Message], tools: list[ToolDefinition], system_suffix: str = ""
    ) -> "AsyncIterator[StreamEvent]":
        messages = self._to_openai_messages(msgs, system_suffix)
        params: dict = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            params["tools"] = self._to_openai_tools(tools)

        try:
            s = await self._client.chat.completions.create(**params)
            tool_calls_buf: dict[int, dict[str, str]] = {}
            finish_reason = None
            async for chunk in s:
                # 末尾 usage chunk（choices 空，带 chunk.usage）
                if not chunk.choices:
                    if chunk.usage is not None:
                        yield StreamEvent(
                            usage=Usage(
                                input_tokens=chunk.usage.prompt_tokens,
                                output_tokens=chunk.usage.completion_tokens,
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

    def _to_openai_messages(self, msgs: list[Message], system_suffix: str = "") -> list[dict]:
        system_text = SYSTEM_PROMPT
        if system_suffix != "":
            system_text = SYSTEM_PROMPT + "\n\n" + system_suffix
        result: list[dict] = [{"role": "system", "content": system_text}]
        for m in msgs:
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
        return result
