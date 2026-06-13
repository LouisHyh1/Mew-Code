"""Tests for agent up-to-2-round tool loop with fake provider."""

from collections.abc import AsyncIterator

import pytest

from novacode.agent import Agent, Phase
from novacode.conversation import Conversation
from novacode.llm import (
    ROLE_ASSISTANT,
    ROLE_TOOL,
    ROLE_USER,
    Message,
    StreamEvent,
    ToolCall,
    ToolDefinition,
)
from novacode.tool import Registry, Result


class FakeReadTool:
    """Fake read_file tool that returns canned content."""

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


class FakeProvider:
    """可编排 Provider，done 由 FakeProvider 在脚本末自动追加并递增 call_count。"""

    def __init__(self, scripts: list[list[StreamEvent]]) -> None:
        self.scripts = scripts
        self.call_count = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    async def stream(
        self, msgs: list[Message], tools: list[ToolDefinition]
    ) -> "AsyncIterator[StreamEvent]":
        if self.call_count >= len(self.scripts):
            yield StreamEvent(done=True)
            return
        for ev in self.scripts[self.call_count]:
            yield ev
        self.call_count += 1
        yield StreamEvent(done=True)


@pytest.mark.asyncio
async def test_agent_full_loop_ac8():
    """端到端：R1 tool call → 执行 → R2 纯文本 → 最终答复。"""
    scripts = [
        [  # R1：preamble + tool call
            StreamEvent(text="Let me "),
            StreamEvent(text="read it."),
            StreamEvent(tool_calls=[
                ToolCall(id="t1", name="read_file", input='{"path": "test.txt"}')
            ]),
        ],
        [  # R2：纯文本答复
            StreamEvent(text="The file says:"),
            StreamEvent(text=" file contents here."),
        ],
    ]
    registry = Registry()
    registry.register(FakeReadTool())
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("read test.txt")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv):
        events.append(ev)

    phases_seen = []
    final_texts = []
    for ev in events:
        if ev.tool is not None:
            phases_seen.append(ev.tool.phase)
        if ev.text:
            final_texts.append(ev.text)

    assert Phase.START in phases_seen
    assert Phase.END in phases_seen
    assert any("file contents here" in t for t in final_texts)

    msgs = conv.messages()
    assert len(msgs) == 4  # user, assistant+tool, tool_result, assistant
    assert msgs[0].role == ROLE_USER
    assert msgs[1].role == ROLE_ASSISTANT
    assert len(msgs[1].tool_calls) == 1
    assert msgs[2].role == ROLE_TOOL
    assert len(msgs[2].tool_results) == 1
    assert msgs[3].role == ROLE_ASSISTANT


@pytest.mark.asyncio
async def test_agent_two_round_tools():
    """R1 tool → R2 tool → 最终文本（验证最多两轮工具执行）。"""
    scripts = [
        [  # R1：tool call
            StreamEvent(tool_calls=[
                ToolCall(id="t1", name="read_file", input='{"path": "a.txt"}')
            ]),
        ],
        [  # R2：又一个 tool call → 第二轮也会执行
            StreamEvent(text="Looking more…"),
            StreamEvent(tool_calls=[
                ToolCall(id="t2", name="read_file", input='{"path": "b.txt"}')
            ]),
        ],
        [  # 最终轮：纯文本
            StreamEvent(text="All done."),
        ],
    ]
    registry = Registry()
    registry.register(FakeReadTool())
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("go")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv):
        events.append(ev)

    tool_starts = [ev for ev in events if ev.tool and ev.tool.phase == Phase.START]
    tool_ends = [ev for ev in events if ev.tool and ev.tool.phase == Phase.END]
    # 两轮工具各一次 START/END
    assert len(tool_starts) == 2
    assert len(tool_ends) == 2

    msgs = conv.messages()
    assert len(msgs) == 6  # user, asst+tool, toolres, asst+tool, toolres, asst
    assert msgs[-1].role == ROLE_ASSISTANT
    assert "All done." in msgs[-1].content


@pytest.mark.asyncio
async def test_agent_final_round_ignores_tools():
    """最终答复轮的工具调用被忽略（等价原 AC9）。"""
    scripts = [
        [  # R1
            StreamEvent(tool_calls=[
                ToolCall(id="t1", name="read_file", input='{"path": "a.txt"}')
            ]),
        ],
        [  # R2
            StreamEvent(tool_calls=[
                ToolCall(id="t2", name="read_file", input='{"path": "b.txt"}')
            ]),
        ],
        [  # 最终轮：模型还想调工具 → 应被忽略
            StreamEvent(text="Should be final."),
            StreamEvent(tool_calls=[
                ToolCall(id="t3", name="read_file", input='{"path": "c.txt"}')
            ]),
        ],
    ]
    registry = Registry()
    registry.register(FakeReadTool())
    provider = FakeProvider(scripts)
    conv = Conversation()
    conv.add_user("go")

    agent = Agent(provider, registry)
    events = []
    async for ev in agent.run(conv):
        events.append(ev)

    # 只有前两轮的工具被执行（2 次 START/END）
    tool_starts = [ev for ev in events if ev.tool and ev.tool.phase == Phase.START]
    assert len(tool_starts) == 2

    # conv 末尾是 assistant 不含 tool_calls
    msgs = conv.messages()
    assert msgs[-1].role == ROLE_ASSISTANT
    assert msgs[-1].tool_calls == []
    assert "Should be final." in msgs[-1].content
