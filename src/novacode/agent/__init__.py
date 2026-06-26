"""ReAct 循环编排——模型自主多轮：想 → 调工具 → 看结果 → 边做边调整，直到任务完成。"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum, IntEnum

from novacode import prompt
from novacode.conversation import Conversation
from novacode.llm import Provider, Request, System, ToolCall, ToolResult
from novacode.tool import DEFAULT_TIMEOUT, Registry

MAX_ITERATIONS: int = 25
MAX_UNKNOWN_RUN: int = 3

# 规划模式提醒：首轮完整，之后每 PLAN_REMINDER_INTERVAL 轮重复完整，其余轮精简。
PLAN_REMINDER_INTERVAL: int = 4

NOTICE_MAX_ITER = "（已达最大迭代轮数 25，自动停止；可继续发消息推进。）"
NOTICE_UNKNOWN_TOOLS = "（连续多轮只请求到未注册的工具，自动停止。）"
NOTICE_STREAM_ERR = "（请求出错，本轮已中断。）"
NOTICE_CANCELLED = "（已取消。）"


class Mode(IntEnum):
    NORMAL = 0
    PLAN = 1


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
class Usage:
    """一轮请求的 token 用量（透传 llm.Usage 语义，含缓存命中）。"""

    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0


@dataclass
class Event:
    """Agent Loop 对外事件流元素，TUI 据非默认字段分派渲染。"""

    text: str = ""
    tool: ToolEvent | None = None
    usage: Usage | None = None
    iter: int = 0
    notice: str = ""
    done: bool = False
    err: Exception | None = None


def _args_preview(args: str) -> str:
    """工具参数截断预览（最多 80 字符）。"""
    return args[:80] + "…" if len(args) > 80 else args


class Agent:
    """持有 provider 与注册中心，执行 ReAct 循环。"""

    def __init__(self, provider: Provider, registry: Registry, version: str = "") -> None:
        self._provider = provider
        self._registry = registry
        self._version = version

    async def run(
        self,
        conv: Conversation,
        mode: Mode,
        cancel: asyncio.Event,
    ) -> AsyncIterator[Event]:
        # ── 环境采集 + 系统提示装配（run 起始一次） ──────────
        env = prompt.gather_environment(self._version, self._provider.model)
        sys = prompt.build_system_prompt()
        env_text = env.render()

        # 按 mode 取工具集
        if mode == Mode.PLAN:
            defs = self._registry.read_only_definitions()
        else:
            defs = self._registry.definitions()

        unknown_run = 0

        for it in range(1, MAX_ITERATIONS + 1):
            yield Event(iter=it)
            if cancel.is_set():
                self._finish_cancelled(conv)
                return

            # ── 本轮 reminder（规划模式按轮次详略） ─────────────
            reminder = ""
            if mode == Mode.PLAN:
                full = it == 1 or (it - 1) % PLAN_REMINDER_INTERVAL == 0
                reminder = prompt.plan_reminder(full)

            stream_events, text, calls, usage, ok = await self._stream_once(
                conv, defs, sys, env_text, reminder, cancel
            )
            for ev in stream_events:
                yield ev

            if not ok:
                if cancel.is_set():
                    self._finish_cancelled(conv)
                    return
                self._ensure_assistant_tail(conv, NOTICE_STREAM_ERR)
                return

            if usage is not None:
                yield Event(
                    usage=Usage(
                        input=usage.input_tokens,
                        output=usage.output_tokens,
                        cache_write=usage.cache_write,
                        cache_read=usage.cache_read,
                    )
                )

            # 无工具调用 → 自然完成
            if not calls:
                conv.add_assistant(self._ensure_final(text))
                yield Event(done=True)
                return

            # 有工具调用
            conv.add_assistant_with_tool_calls(text, calls)

            # 统计连续未知工具
            unknown_run = unknown_run + 1 if self._all_unknown(calls) else 0

            batch_events, results, completed = await self._execute_batched(calls, cancel)
            for ev in batch_events:
                yield ev
            conv.add_tool_results(results)

            # 执行中被取消——最高优先级终止
            if not completed:
                self._ensure_assistant_tail(conv, NOTICE_CANCELLED)
                return

            # 连续未知工具停止
            if unknown_run >= MAX_UNKNOWN_RUN:
                yield Event(notice=NOTICE_UNKNOWN_TOOLS)
                self._ensure_assistant_tail(conv, NOTICE_UNKNOWN_TOOLS)
                yield Event(done=True)
                return

        # 迭代上限
        yield Event(notice=NOTICE_MAX_ITER)
        self._ensure_assistant_tail(conv, NOTICE_MAX_ITER)
        yield Event(done=True)

    async def _stream_once(
        self,
        conv: Conversation,
        defs: list,
        sys: str,
        env_text: str,
        reminder: str,
        cancel: asyncio.Event,
    ):
        """单轮 LLM 请求。返回 (events, text, calls, usage, ok)。"""
        from novacode.llm import Usage as LLMUsage

        events: list[Event] = []
        text = ""
        calls: list[ToolCall] = []
        usage: LLMUsage | None = None

        req = Request(
            messages=conv.messages(),
            tools=defs,
            system=System(stable=sys, environment=env_text),
            reminder=reminder,
        )

        async for ev in self._provider.stream(req):
            if cancel.is_set():
                return events, "", [], None, False

            if ev.err is not None:
                events.append(Event(err=ev.err))
                return events, "", [], None, False

            if ev.usage is not None:
                usage = ev.usage

            if ev.tool_calls:
                calls = ev.tool_calls

            if ev.text:
                text += ev.text
                events.append(Event(text=ev.text))

        if cancel.is_set():
            return events, "", [], None, False

        return events, text, calls, usage, True

    async def _execute_batched(
        self, calls: list[ToolCall], cancel: asyncio.Event
    ) -> tuple[list[Event], list[ToolResult], bool]:
        """保序分批并发执行工具。返回 (events, results, completed)。"""
        events: list[Event] = []
        results: list[ToolResult | None] = [None] * len(calls)
        i = 0
        while i < len(calls):
            if cancel.is_set():
                self._fill_cancelled(results, i)
                return events, self._finalize_results(results, calls), False

            if self._registry.is_read_only(calls[i].name):
                # 吃最长连续只读区间 [i, j)
                j = i + 1
                while j < len(calls) and self._registry.is_read_only(calls[j].name):
                    j += 1
                # 先按序发所有 PHASE_START
                for k in range(i, j):
                    events.append(
                        Event(
                            tool=ToolEvent(
                                name=calls[k].name,
                                args=_args_preview(calls[k].input),
                                phase=Phase.START,
                            )
                        )
                    )
                # 并发执行
                tasks = [self._run_one(k, calls[k], results, cancel) for k in range(i, j)]
                await asyncio.gather(*tasks)
                # 按序发所有 PHASE_END
                for k in range(i, j):
                    if results[k] is not None:
                        r = results[k]
                        events.append(
                            Event(
                                tool=ToolEvent(
                                    name=calls[k].name,
                                    args=_args_preview(calls[k].input),
                                    phase=Phase.END,
                                    result=r.content,
                                    is_error=r.is_error,
                                )
                            )
                        )
                i = j
            else:
                # 有副作用：单个串行执行
                events.append(
                    Event(
                        tool=ToolEvent(
                            name=calls[i].name,
                            args=_args_preview(calls[i].input),
                            phase=Phase.START,
                        )
                    )
                )
                await self._run_one(i, calls[i], results, cancel)
                if results[i] is not None:
                    r = results[i]
                    events.append(
                        Event(
                            tool=ToolEvent(
                                name=calls[i].name,
                                args=_args_preview(calls[i].input),
                                phase=Phase.END,
                                result=r.content,
                                is_error=r.is_error,
                            )
                        )
                    )
                i += 1

        return events, self._finalize_results(results, calls), True

    async def _run_one(
        self,
        idx: int,
        call: ToolCall,
        results: list[ToolResult | None],
        cancel: asyncio.Event,
    ) -> None:
        """执行单个工具调用，结果写入 results[idx]。支持 cancel 中断。"""
        if cancel.is_set():
            results[idx] = ToolResult(tool_call_id=call.id, content=NOTICE_CANCELLED, is_error=True)
            return

        from novacode.tool import Result as ToolExecResult

        try:
            exec_task = asyncio.create_task(
                self._registry.execute(call.name, call.input, timeout=DEFAULT_TIMEOUT)
            )
            cancel_task = asyncio.create_task(cancel.wait())
            done, pending = await asyncio.wait(
                [exec_task, cancel_task],
                timeout=DEFAULT_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # 清理未完成的任务
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            if cancel_task in done:
                results[idx] = ToolResult(
                    tool_call_id=call.id, content=NOTICE_CANCELLED, is_error=True
                )
                return

            r: ToolExecResult = exec_task.result()
            results[idx] = ToolResult(
                tool_call_id=call.id,
                content=r.content,
                is_error=r.is_error,
            )
        except TimeoutError:
            results[idx] = ToolResult(
                tool_call_id=call.id,
                content=f"工具 {call.name} 执行超时（{DEFAULT_TIMEOUT}s）",
                is_error=True,
            )
        except Exception as e:
            results[idx] = ToolResult(
                tool_call_id=call.id,
                content=f"工具 {call.name} 异常: {e}",
                is_error=True,
            )

    def _fill_cancelled(self, results: list[ToolResult | None], start: int) -> None:
        """给 start 及之后的所有 None 槽位填「已取消」结果。"""
        for k in range(start, len(results)):
            if results[k] is None:
                results[k] = ToolResult(
                    tool_call_id="",
                    content=NOTICE_CANCELLED,
                    is_error=True,
                )

    def _finalize_results(
        self, results: list[ToolResult | None], calls: list[ToolCall]
    ) -> list[ToolResult]:
        """确保所有槽位都有 ToolResult（防御性兜底）。"""
        finalized: list[ToolResult] = []
        for k, r in enumerate(results):
            if r is not None:
                finalized.append(r)
            else:
                finalized.append(
                    ToolResult(
                        tool_call_id=calls[k].id if k < len(calls) else "",
                        content="（未执行）",
                        is_error=True,
                    )
                )
        return finalized

    # ── 辅助函数 ─────────────────────────────────────────────

    def _all_unknown(self, calls: list[ToolCall]) -> bool:
        """所有 call 的 name 在注册中心都不存在才返回 True；混入已知工具视为有进展。"""
        for c in calls:
            if self._registry.get(c.name) is not None:
                return False
        return True

    def _ensure_final(self, text: str) -> str:
        """text 非空原样返回；为空则返回占位文本（避免空 assistant 回合）。"""
        if text.strip():
            return text
        fallback = (
            "（工具已执行完毕。如果你需要更多分析，请继续提问，我会基于已有结果给出详细回答。）"
        )
        return fallback

    def _ensure_assistant_tail(self, conv: Conversation, fallback: str) -> None:
        """若 conv 末尾不是 assistant 角色，写入兜底文案保证角色交替合法。"""
        if conv.last_role() != "assistant":
            conv.add_assistant(fallback)

    def _finish_cancelled(self, conv: Conversation) -> None:
        """取消路径统一收尾——保证 assistant 文本尾巴后 generator 自然结束。"""
        self._ensure_assistant_tail(conv, NOTICE_CANCELLED)
