"""Tests for conversation module."""

from novacode.conversation import Conversation
from novacode.llm import ROLE_ASSISTANT, ROLE_TOOL, ROLE_USER, ToolCall, ToolResult


def test_add_and_retrieve() -> None:
    conv = Conversation()
    conv.add_user("hello")
    conv.add_assistant("hi there")
    msgs = conv.messages()
    assert len(msgs) == 2
    assert msgs[0].role == ROLE_USER
    assert msgs[0].content == "hello"
    assert msgs[1].role == ROLE_ASSISTANT
    assert msgs[1].content == "hi there"


def test_messages_is_copy() -> None:
    conv = Conversation()
    conv.add_user("hello")
    msgs = conv.messages()
    msgs.clear()
    assert len(conv.messages()) == 1


def test_tool_call_roundtrip() -> None:
    """依次 add_user → add_assistant_with_tool_calls → add_tool_results → add_assistant，
    检查 messages() 长度=4、role 序列正确、tool_calls/tool_results 内容正确。"""
    conv = Conversation()
    conv.add_user("read test.txt")
    calls = [ToolCall(id="toolu_001", name="read_file", input='{"path": "test.txt"}')]
    conv.add_assistant_with_tool_calls("Let me read the file.", calls)
    results = [ToolResult(tool_call_id="toolu_001", content="hello", is_error=False)]
    conv.add_tool_results(results)
    conv.add_assistant("The file contains 'hello'.")

    msgs = conv.messages()
    assert len(msgs) == 4
    assert msgs[0].role == ROLE_USER
    assert msgs[1].role == ROLE_ASSISTANT
    assert len(msgs[1].tool_calls) == 1
    assert msgs[1].tool_calls[0].name == "read_file"
    assert msgs[2].role == ROLE_TOOL
    assert len(msgs[2].tool_results) == 1
    assert msgs[2].tool_results[0].content == "hello"
    assert msgs[3].role == ROLE_ASSISTANT
    assert "hello" in msgs[3].content


def test_last_role_empty() -> None:
    """空会话 last_role() 返回空字符串。"""
    conv = Conversation()
    assert conv.last_role() == ""


def test_last_role_user() -> None:
    """add_user 后 last_role() == 'user'。"""
    conv = Conversation()
    conv.add_user("hello")
    assert conv.last_role() == ROLE_USER


def test_last_role_assistant() -> None:
    """add_assistant 后 last_role() == 'assistant'。"""
    conv = Conversation()
    conv.add_assistant("hi")
    assert conv.last_role() == ROLE_ASSISTANT


def test_last_role_tool() -> None:
    """add_tool_results 后 last_role() == 'tool'。"""
    conv = Conversation()
    conv.add_tool_results([ToolResult(tool_call_id="t1", content="ok")])
    assert conv.last_role() == ROLE_TOOL
