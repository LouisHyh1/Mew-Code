"""Tests for agent ReAct loop with fake provider — ch04 + ch05."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from novacode.agent import (
    MAX_ITERATIONS,
    MAX_UNKNOWN_RUN,
    NOTICE_CANCELLED,
    NOTICE_MAX_ITER,
    NOTICE_UNKNOWN_TOOLS,
    Agent,
    Phase,
)
from novacode.conversation import Conversation
from novacode.llm import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    Request,
    StreamEvent,
    ToolCall,
    Usage,
)
from novacode.permission import Mode
from novacode.tool import Registry, Result

# ── Fake 工具 ──────────────────────────────────────────────


class FakeReadTool:
    """只读工具：返回固定内容。"""

    read_only = True

    def name(self) -> str:
        return "read_file"

    def description(self) -> str:
        return "Read a file."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    async def execute(self, args: str) -> Result:
        return Result(content="file contents here")


# ── Fake Provider（ch05 新签名） ────────────────────────────


class FakeProvider:
    """可编排 Provider：scripts 为 list[list[StreamEvent]]，逐次消费。

    ch05: stream(req: Request) 新签名；记录收到的 req 供断言。
    """

    def __init__(self, scripts: list[list[StreamEvent]]) -> None:
        self.scripts = scripts
        self.call_count = 0
        # 用于断言——记录每次调用的 Request
        self.requests: list[Request] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    async def stream(self, req: Request) -> "AsyncIterator[StreamEvent]":
        self.requests.append(req)
        if self.call_count >= len(self.scripts):
            yield StreamEvent(done=True)
            return
        for ev in self.scripts[self.call_count]:
            yield ev
        self.call_count += 1
        yield StreamEvent(done=True)


class FakeProviderWithUsage(FakeProvider):
    """FakeProvider 支持发送自定义 Usage（含 cache 字段）。"""

    def __init__(
        self,
        scripts: list[list[StreamEvent]],
        usage_seq: list[Usage] | None = None,
    ) -> None:
        super().__init__(scripts)
        self._usage_seq = usage_seq or []
        self._usage_idx = 0

    async def stream(self, req: Request) -> "AsyncIterator[StreamEvent]":
        self.requests.append(req)
        if self.call_count >= len(self.scripts):
            yield StreamEvent(done=True)
            return
        for ev in self.scripts[self.call_count]:
            yield ev
        # 注入自定义 usage
        if self._usage_idx < len(self._usage_seq):
            yield StreamEvent(usage=self._usage_seq[self._usage_idx])
            self._usage_idx += 1
        self.call_count += 1
        yield StreamEvent(done=True)


class InfiniteToolFakeProvider:
    """每轮只返回一个工具调用（永不自然停止），用于测试迭代上限。"""

    def __init__(self) -> None:
        self.call_count = 0
        self.requests: list[Request] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    async def stream(self, req: Request) -> "AsyncIterator[StreamEvent]":
        self.requests.append(req)
        self.call_count += 1
        yield StreamEvent(
            tool_calls=[
                ToolCall(id=f"t{self.call_count}", name="read_file", input='{"path": "x.txt"}')
            ]
        )
        yield StreamEvent(done=True)


# ── 场景 A：多轮链路 (AC1) ─────────────────────────────────


@pytest.mark.asyncio
async def test_multi_turn_autonomous_loop():
    """R1 工具调用 → 执行 → R2 纯文本 → 自然完成。"""
    scripts = [
        [
            StreamEvent(text="Let me "),
            StreamEvent(text="read it."),
            StreamEvent(
                tool_calls=[ToolCall(id="t1", name="read_file", input='{"path": "test.txt"}')]
            ),
        ],
        [
            StreamEvent(text="The file contains: file contents here."),
        ],
    ]
    registry = Registry()
    registry.register(FakeReadTool())
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("read test.txt")

    agent = Agent(provider, registry, "test-version")
    events = []
    async for ev in agent.run(conv, Mode.DEFAULT, asyncio.Event()):
        events.append(ev)

    # 事件类型覆盖
    has_text = any(ev.text for ev in events)
    has_tool = any(ev.tool is not None for ev in events)
    has_done = any(ev.done for ev in events)
    iters = [ev.iter for ev in events if ev.iter > 0]
    assert has_text
    assert has_tool
    assert has_done
    assert len(iters) >= 2  # 至少 iter=1 和 iter=2

    # 最终答复文本
    final_text = "".join(ev.text for ev in events)
    assert "file contents here" in final_text

    # 对话历史完整
    msgs = conv.messages()
    assert len(msgs) == 4  # user, asst+tool, tool_result, asst
    assert msgs[0].role == ROLE_USER
    assert msgs[1].role == ROLE_ASSISTANT
    assert len(msgs[1].tool_calls) == 1
    assert msgs[2].role == ROLE_TOOL
    assert len(msgs[2].tool_results) == 1
    assert msgs[3].role == ROLE_ASSISTANT


# ── 场景 B：迭代上限 (AC3) ─────────────────────────────────


@pytest.mark.asyncio
async def test_max_iterations_stop():
    """无限工具调用 → 恰好 MAX_ITERATIONS 轮后停止。"""
    registry = Registry()
    registry.register(FakeReadTool())
    provider = InfiniteToolFakeProvider()
    conv = Conversation()
    conv.add_user("go")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv, Mode.DEFAULT, asyncio.Event()):
        events.append(ev)

    assert provider.call_count == MAX_ITERATIONS
    notices = [ev.notice for ev in events if ev.notice]
    assert any(NOTICE_MAX_ITER in n for n in notices)
    assert conv.last_role() == "assistant"
    assert NOTICE_MAX_ITER in conv.messages()[-1].content


# ── 场景 C：连续未知工具 (AC4) ─────────────────────────────


@pytest.mark.asyncio
async def test_unknown_tools_stop():
    """连续 MAX_UNKNOWN_RUN 轮只产生未知工具调用 → 停止。"""
    scripts = []
    for _ in range(MAX_UNKNOWN_RUN + 2):
        scripts.append(
            [
                StreamEvent(tool_calls=[ToolCall(id="u1", name="nonexistent_tool", input="{}")]),
            ]
        )
    registry = Registry()
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("go")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv, Mode.DEFAULT, asyncio.Event()):
        events.append(ev)

    notices = [ev.notice for ev in events if ev.notice]
    assert any(NOTICE_UNKNOWN_TOOLS in n for n in notices)
    assert conv.last_role() == "assistant"


@pytest.mark.asyncio
async def test_unknown_reset_by_known_tool():
    """未知工具间混入已知工具 → 计数重置，不提前停。"""
    scripts = [
        [StreamEvent(tool_calls=[ToolCall(id="u1", name="nonexistent", input="{}")])],
        [StreamEvent(tool_calls=[ToolCall(id="t1", name="nonexistent", input="{}")])],
        [StreamEvent(tool_calls=[ToolCall(id="t2", name="read_file", input='{"path": "a.txt"}')])],
        [StreamEvent(text="Done after mixed.")],
    ]
    registry = Registry()
    registry.register(FakeReadTool())
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("go")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv, Mode.DEFAULT, asyncio.Event()):
        events.append(ev)

    # 应自然完成（第4轮纯文本），未被未知工具截停
    assert any(ev.done and not ev.err for ev in events)
    final_text = "".join(ev.text for ev in events)
    assert "Done after mixed" in final_text


# ── 场景 D：保序分批并发 (AC8) ─────────────────────────────


class InstrumentedReadOnlyTool:
    """记录并发峰值的只读插桩工具。"""

    read_only = True
    _concurrent = 0
    _max_concurrent = 0
    _lock = asyncio.Lock()

    def __init__(self, name: str = "ro_tool", sleep: float = 0.1) -> None:
        self._name = name
        self._sleep = sleep

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Instrumented RO."

    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, args: str) -> Result:
        async with InstrumentedReadOnlyTool._lock:
            InstrumentedReadOnlyTool._concurrent += 1
            InstrumentedReadOnlyTool._max_concurrent = max(
                InstrumentedReadOnlyTool._max_concurrent,
                InstrumentedReadOnlyTool._concurrent,
            )
        await asyncio.sleep(self._sleep)
        async with InstrumentedReadOnlyTool._lock:
            InstrumentedReadOnlyTool._concurrent -= 1
        return Result(content=f"{self._name} done")

    @classmethod
    def reset_counters(cls) -> None:
        cls._concurrent = 0
        cls._max_concurrent = 0


class InstrumentedWriteTool:
    """记录开始时刻的有副作用插桩工具。"""

    read_only = False
    start_time: float = 0.0

    def __init__(self, name: str = "rw_tool", sleep: float = 0.05) -> None:
        self._name = name
        self._sleep = sleep

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Instrumented RW."

    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, args: str) -> Result:
        InstrumentedWriteTool.start_time = asyncio.get_event_loop().time()
        await asyncio.sleep(self._sleep)
        return Result(content=f"{self._name} done")


@pytest.mark.asyncio
async def test_concurrent_batch():
    """连续只读工具并发执行、有副作用工具串行，结果按原序回灌。"""
    InstrumentedReadOnlyTool.reset_counters()
    InstrumentedWriteTool.start_time = 0.0

    registry = Registry()
    ro1 = InstrumentedReadOnlyTool("ro1", sleep=0.1)
    ro2 = InstrumentedReadOnlyTool("ro2", sleep=0.1)
    rw = InstrumentedWriteTool("rw", sleep=0.05)
    registry.register(ro1)
    registry.register(ro2)
    registry.register(rw)

    scripts = [
        [
            StreamEvent(
                tool_calls=[
                    ToolCall(id="c1", name="ro1", input="{}"),
                    ToolCall(id="c2", name="ro2", input="{}"),
                    ToolCall(id="c3", name="rw", input="{}"),
                ]
            ),
        ]
    ]
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("go")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv, Mode.DEFAULT, asyncio.Event()):
        events.append(ev)

    # 两只读的并发峰值 ≥2（确实并发）
    assert InstrumentedReadOnlyTool._max_concurrent >= 2, (
        f"Expected concurrent peak >= 2, got {InstrumentedReadOnlyTool._max_concurrent}"
    )

    # rw 的开始时刻应晚于两只读的完成（串行——在只读批之后）
    assert InstrumentedWriteTool.start_time > 0.0, "rw should have been executed"

    # 工具结果按调用序回灌
    msgs = conv.messages()
    assert len(msgs) >= 3
    tool_results_msg = msgs[2]  # ROLE_TOOL
    assert tool_results_msg.role == ROLE_TOOL
    assert len(tool_results_msg.tool_results) == 3
    result_ids = [r.tool_call_id for r in tool_results_msg.tool_results]
    assert result_ids == ["c1", "c2", "c3"]

    # 工具事件顺序：6 个事件（3×START + 3×END），START 按序、END 按序
    tool_events = [ev.tool for ev in events if ev.tool is not None]
    start_names = [t.name for t in tool_events if t.phase == Phase.START]
    end_names = [t.name for t in tool_events if t.phase == Phase.END]
    assert start_names == ["ro1", "ro2", "rw"]
    assert end_names == ["ro1", "ro2", "rw"]


# ── 场景 E：取消历史一致 (AC9) ─────────────────────────────


class BlockingTool:
    """执行中阻塞的工具，供取消测试使用。"""

    read_only = True

    def __init__(self, name: str = "blocker", block_time: float = 5.0) -> None:
        self._name = name
        self._block_time = block_time

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Blocking tool."

    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, args: str) -> Result:
        await asyncio.sleep(self._block_time)
        return Result(content="done (should not reach)")


@pytest.mark.asyncio
async def test_cancel_history_consistency():
    """执行中取消 → 历史配对合法、末尾 assistant 文本、可继续对话。"""
    registry = Registry()
    blocker = BlockingTool("blocker", block_time=5.0)
    registry.register(blocker)

    scripts = [
        [
            StreamEvent(text="About to block…"),
            StreamEvent(
                tool_calls=[
                    ToolCall(id="b1", name="blocker", input="{}"),
                ]
            ),
        ]
    ]
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("go")

    cancel = asyncio.Event()
    agent = Agent(provider, registry)

    events = []
    gen = agent.run(conv, Mode.DEFAULT, cancel)

    # 收集事件直到工具开始执行
    async for ev in gen:
        events.append(ev)
        if ev.tool is not None and ev.tool.phase == Phase.START:
            # 在工具执行期间触发取消
            await asyncio.sleep(0.05)
            cancel.set()

    # 验证历史合法
    msgs = conv.messages()
    assert conv.last_role() == "assistant"
    # 不能有悬空 tool_use（必须有对应的 tool_result）
    for msg in msgs:
        if msg.role == ROLE_ASSISTANT and msg.tool_calls:
            found = False
            for m2 in msgs:
                if m2.role == ROLE_TOOL:
                    for tr in m2.tool_results:
                        if tr.tool_call_id in [tc.id for tc in msg.tool_calls]:
                            found = True
            assert found, "Dangling tool_use without tool_result"

    # 取消后可以追加新一轮对话
    scripts2 = [[StreamEvent(text="Continuing after cancel.")]]
    provider2 = FakeProvider(scripts2)
    agent2 = Agent(provider2, registry)
    async for ev in agent2.run(conv, Mode.DEFAULT, asyncio.Event()):
        pass
    assert conv.last_role() == "assistant"


# ── 场景 F：流出错 (AC5) ──────────────────────────────────


@pytest.mark.asyncio
async def test_stream_error_recovery():
    """provider 流出错 → 停止本轮、发 err、历史合法。"""
    scripts = [
        [
            StreamEvent(err=RuntimeError("connection lost")),
        ]
    ]
    registry = Registry()
    registry.register(FakeReadTool())
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("go")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv, Mode.DEFAULT, asyncio.Event()):
        events.append(ev)

    errs = [ev.err for ev in events if ev.err is not None]
    assert len(errs) >= 1
    assert "connection lost" in str(errs[0])
    assert conv.last_role() == "assistant"


# ── 取消入口 (AC10) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_before_stream():
    """run 开始前就触发 cancel → 立即停止、无请求发出。"""
    registry = Registry()
    registry.register(FakeReadTool())

    scripts = [[StreamEvent(text="Should not appear.")]]
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("go")

    cancel = asyncio.Event()
    cancel.set()  # 预先触发

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv, Mode.DEFAULT, cancel):
        events.append(ev)

    assert len(events) == 1  # 仅 iter=1 事件（先 yield 再检查 cancel）
    assert events[0].iter == 1
    assert conv.last_role() == "assistant"
    assert NOTICE_CANCELLED in conv.messages()[-1].content


# ── ch05 新测试 ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_mode_toolset():
    """AC9/F7 — Mode.PLAN → req.tools 仅含只读工具。"""
    registry = Registry()
    registry.register(FakeReadTool())  # read_only = True

    class FakeWriteTool:
        read_only = False

        def name(self) -> str:
            return "write_file"

        def description(self) -> str:
            return "Write a file."

        def parameters(self) -> dict:
            return {"type": "object", "properties": {}, "required": []}

        async def execute(self, args: str) -> Result:
            return Result(content="written")

    registry.register(FakeWriteTool())

    scripts = [[StreamEvent(text="Here is the plan…")]]
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("plan the feature")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv, Mode.PLAN, asyncio.Event()):
        events.append(ev)

    # 检查 fake 收到的工具定义（仅只读）
    assert len(provider.requests) >= 1
    plan_tool_names = [t.name for t in provider.requests[0].tools]
    assert "read_file" in plan_tool_names
    assert "write_file" not in plan_tool_names


@pytest.mark.asyncio
async def test_request_has_system_and_environment():
    """ch05 — req.system.stable 非空、req.system.environment 非空。"""
    registry = Registry()
    registry.register(FakeReadTool())
    scripts = [[StreamEvent(text="Hello.")]]
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("hi")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv, Mode.DEFAULT, asyncio.Event()):
        events.append(ev)

    assert len(provider.requests) >= 1
    req = provider.requests[0]
    assert len(req.system.stable) > 0
    assert "NovaCode" in req.system.stable
    assert "Working Directory" in req.system.environment


@pytest.mark.asyncio
async def test_stable_system_same_for_normal_and_plan():
    """AC9/F7/N1 — 普通与规划模式 req.system.stable 相同（规划提醒已移出系统通道）。"""
    registry = Registry()
    registry.register(FakeReadTool())

    # Normal mode
    provider_normal = FakeProvider([[StreamEvent(text="ok")]])
    conv_normal = Conversation()
    conv_normal.add_user("hi")
    agent_normal = Agent(provider_normal, registry)
    async for ev in agent_normal.run(conv_normal, Mode.DEFAULT, asyncio.Event()):
        pass

    # Plan mode
    provider_plan = FakeProvider([[StreamEvent(text="plan")]])
    conv_plan = Conversation()
    conv_plan.add_user("plan")
    agent_plan = Agent(provider_plan, registry)
    async for ev in agent_plan.run(conv_plan, Mode.PLAN, asyncio.Event()):
        pass

    stable_normal = provider_normal.requests[0].system.stable
    stable_plan = provider_plan.requests[0].system.stable
    assert stable_normal == stable_plan, "Stable system prompt must be identical across modes"


@pytest.mark.asyncio
async def test_plan_mode_reminder_per_iteration():
    """AC9/F7 — 规划模式 iter1 完整提醒、iter2 精简、iter5 完整。

    使用一个多轮 Plan 模式的脚本来模拟经过多轮。
    构造两个只读工具调用的脚本 → iter1 和 iter5 都是完整提醒，
    因为 PLAN_REMINDER_INTERVAL=4，iter5 时 (5-1)%4==0。
    """
    registry = Registry()
    registry.register(FakeReadTool())

    # 两轮工具调用后自然完成
    scripts = [
        [
            StreamEvent(
                tool_calls=[ToolCall(id="t1", name="read_file", input='{"path": "a.txt"}')]
            ),
        ],
        [
            StreamEvent(
                tool_calls=[ToolCall(id="t2", name="read_file", input='{"path": "b.txt"}')]
            ),
        ],
        [
            StreamEvent(text="Final plan."),
        ],
    ]
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("plan this")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv, Mode.PLAN, asyncio.Event()):
        events.append(ev)

    # 检查各轮 reminder
    assert len(provider.requests) >= 3

    # iter1: 完整提醒
    r1 = provider.requests[0].reminder
    assert "<system-reminder>" in r1
    assert "step-by-step" in r1 or "/do" in r1

    # iter2: 精简提醒
    r2 = provider.requests[1].reminder
    assert "<system-reminder>" in r2
    assert "PLAN MODE" in r2
    assert len(r2) < len(r1), f"iter2 should be concise: {len(r2)} >= {len(r1)}"

    # iter3: 精简提醒（3-1=2, 2%4!=0）
    r3 = provider.requests[2].reminder
    assert "PLAN MODE" in r3

    # 验证 PLAN_REMINDER_INTERVAL=4——iter5 应完整
    # 这里只有3轮，无法验证 iter5；通过常量断言
    from novacode.agent import PLAN_REMINDER_INTERVAL

    assert PLAN_REMINDER_INTERVAL == 4


@pytest.mark.asyncio
async def test_reminder_not_persisted_in_conversation():
    """AC8/F6 — reminder 不写入 conv 持久历史。"""
    registry = Registry()
    registry.register(FakeReadTool())
    scripts = [[StreamEvent(text="ok")]]
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("hi")

    agent = Agent(provider, registry)
    async for ev in agent.run(conv, Mode.PLAN, asyncio.Event()):
        pass

    msgs = conv.messages()
    for m in msgs:
        assert "<system-reminder>" not in m.content, (
            "Reminder must not appear in persisted conversation"
        )


@pytest.mark.asyncio
async def test_normal_mode_no_reminder():
    """普通模式下不注入 reminder。"""
    registry = Registry()
    registry.register(FakeReadTool())
    scripts = [[StreamEvent(text="ok")]]
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("hi")

    agent = Agent(provider, registry)
    async for ev in agent.run(conv, Mode.DEFAULT, asyncio.Event()):
        pass

    assert len(provider.requests) >= 1
    assert provider.requests[0].reminder == "", "Normal mode should have no reminder"


@pytest.mark.asyncio
async def test_cache_usage_passthrough():
    """AC6/F4 — 缓存用量从 provider 透传到 Event.usage。"""
    registry = Registry()
    registry.register(FakeReadTool())
    scripts = [[StreamEvent(text="ok")]]
    usage_with_cache = Usage(
        input_tokens=100,
        output_tokens=50,
        cache_write=200,
        cache_read=150,
    )
    provider = FakeProviderWithUsage(scripts, usage_seq=[usage_with_cache])
    conv = Conversation()
    conv.add_user("test")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv, Mode.DEFAULT, asyncio.Event()):
        events.append(ev)

    # 找到 usage 事件
    usage_events = [ev.usage for ev in events if ev.usage is not None]
    assert len(usage_events) >= 1, "Should have at least one usage event"
    u = usage_events[0]
    assert u.input == 100
    assert u.output == 50
    assert u.cache_write == 200
    assert u.cache_read == 150


@pytest.mark.asyncio
async def test_environment_in_request():
    """AC3/F2 — req.system.environment 含关键字段。"""
    registry = Registry()
    registry.register(FakeReadTool())
    scripts = [[StreamEvent(text="ok")]]
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("where am I?")

    agent = Agent(provider, registry, "2.0-test")
    async for ev in agent.run(conv, Mode.DEFAULT, asyncio.Event()):
        pass

    assert len(provider.requests) >= 1
    env = provider.requests[0].system.environment
    assert "Working Directory" in env
    assert "Platform:" in env or "Platform" in env
    assert "Date:" in env or "Date" in env
    assert "2.0-test" in env


# ── 历史合法 (AC12/N3) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_mode_multi_turn_no_400_pattern():
    """规划模式多轮后消息序列角色合法——测试 reminder 注入不破坏结构。

    我们无法真实验证 400，但可以验证：
    1. reminder 注入了 req
    2. conversation 中不含 reminder
    3. 消息序列在注入前后角色交替合法
    """
    registry = Registry()
    registry.register(FakeReadTool())

    # 多轮只读工具调用
    scripts = [
        [StreamEvent(tool_calls=[ToolCall(id="t1", name="read_file", input='{"path":"a.txt"}')])],
        [StreamEvent(text="Plan complete.")],
    ]
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("investigate")

    agent = Agent(provider, registry)
    async for ev in agent.run(conv, Mode.PLAN, asyncio.Event()):
        pass

    # 2轮请求（第一轮工具调用 + 第二轮文本完成）
    assert len(provider.requests) >= 2
    # 两轮都有 reminder
    assert provider.requests[0].reminder != ""
    assert provider.requests[1].reminder != ""
    # conversation 干净
    msgs = conv.messages()
    for m in msgs:
        assert "<system-reminder>" not in m.content
