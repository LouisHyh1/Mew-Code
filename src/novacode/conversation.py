"""Single-session conversation history."""

from novacode.llm import ROLE_ASSISTANT, ROLE_TOOL, ROLE_USER, Message, ToolCall, ToolResult


class Conversation:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add_user(self, text: str) -> None:
        self._messages.append(Message(role=ROLE_USER, content=text))

    def add_assistant(self, text: str) -> None:
        self._messages.append(Message(role=ROLE_ASSISTANT, content=text))

    def add_assistant_with_tool_calls(self, text: str, calls: list[ToolCall]) -> None:
        """assistant 工具调用回合。"""
        self._messages.append(Message(
            role=ROLE_ASSISTANT, content=text, tool_calls=list(calls)
        ))

    def add_tool_results(self, results: list[ToolResult]) -> None:
        """ROLE_TOOL 结果回合。"""
        self._messages.append(Message(
            role=ROLE_TOOL, tool_results=list(results)
        ))

    def messages(self) -> list[Message]:
        return list(self._messages)
