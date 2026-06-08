"""Tests for conversation module."""

from mewcode.conversation import Conversation


def test_add_and_retrieve() -> None:
    conv = Conversation()
    conv.add_user("hello")
    conv.add_assistant("hi there")
    msgs = conv.messages()
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[0].content == "hello"
    assert msgs[1].role == "assistant"
    assert msgs[1].content == "hi there"


def test_messages_is_copy() -> None:
    conv = Conversation()
    conv.add_user("hello")
    msgs = conv.messages()
    msgs.clear()
    assert len(conv.messages()) == 1  # original unaffected
