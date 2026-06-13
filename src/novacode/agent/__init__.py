"""Single-turn agent — up to 2 rounds of tool calls, then final answer."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

from novacode.conversation import Conversation
from novacode.llm import Provider, ToolCall, ToolResult
from novacode.tool import DEFAULT_TIMEOUT, Registry

MAX_TOOL_ROUNDS = 2


class Phase(Enum):
    START = "start"
    END = "end"


@dataclass
class ToolEvent:
    """一次工具调用的开始/结束（供 TUI 渲染工具行与结果摘要）。"""
    name: str
    args: str = ""
    phase: Phase = Phase.START
    result: str = ""
    is_error: bool = False


@dataclass
class Event:
    """单轮闭环对外事件流元素，TUI 据非 None 字段分派渲染。"""
    text: str = ""
    tool: ToolEvent | None = None
    done: bool = False
    err: Exception | None = None


class Agent:
    """持有 provider 与注册中心，执行最多两轮工具调用后给出最终答复。"""

    def __init__(self, provider: Provider, registry: Registry) -> None:
        self._provider = provider
        self._registry = registry

    async def run(self, conv: Conversation) -> AsyncIterator[Event]:
        defs = self._registry.definitions()

        for round_num in range(1, MAX_TOOL_ROUNDS + 1):
            # ── 请求 LLM（带工具定义） ──
            text = ""
            calls: list[ToolCall] = []
            try:
                async for ev in self._provider.stream(conv.messages(), defs):
                    if ev.err is not None:
                        yield Event(err=ev.err)
                        return
                    if ev.text:
                        text += ev.text
                        yield Event(text=ev.text)
                    if ev.tool_calls:
                        calls = ev.tool_calls
                    if ev.done:
                        break
            except Exception as e:
                yield Event(err=e)
                return

            # 无工具调用 → 纯文本回答，直接结束
            if not calls:
                conv.add_assistant(text)
                yield Event(done=True)
                return

            # 有工具调用 → 追加 assistant tool_use 回合并执行
            conv.add_assistant_with_tool_calls(text, calls)

            results: list[ToolResult] = []
            for call in calls:
                args_preview = (
                    call.input[:80] + "…" if len(call.input) > 80 else call.input
                )
                yield Event(tool=ToolEvent(
                    name=call.name, args=args_preview, phase=Phase.START
                ))
                r = await self._registry.execute(
                    call.name, call.input, timeout=DEFAULT_TIMEOUT
                )
                yield Event(tool=ToolEvent(
                    name=call.name,
                    args=args_preview,
                    phase=Phase.END,
                    result=r.content,
                    is_error=r.is_error,
                ))
                results.append(ToolResult(
                    tool_call_id=call.id,
                    content=r.content,
                    is_error=r.is_error,
                ))

            # 结果回灌到对话历史
            conv.add_tool_results(results)

        # ── 最后一轮：纯文本答复（忽略工具调用） ──
        final = ""
        try:
            async for ev in self._provider.stream(conv.messages(), defs):
                if ev.err is not None:
                    yield Event(err=ev.err)
                    return
                if ev.text:
                    final += ev.text
                    yield Event(text=ev.text)
                if ev.done:
                    break
        except Exception as e:
            yield Event(err=e)
            return

        if not final.strip():
            final = (
                "（工具已执行完毕。如果你需要更多分析，"
                "请继续提问，我会基于已有结果给出详细回答。）"
            )
            yield Event(text=final)

        conv.add_assistant(final)
        yield Event(done=True)
