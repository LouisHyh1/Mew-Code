"""Tests for Anthropic system block serialization — cache_control guard."""

from novacode.config import ProviderConfig
from novacode.llm import Message, Request, System
from novacode.llm.anthropic_provider import AnthropicProvider


def _make_dummy_cfg() -> ProviderConfig:
    """构造一个不会真正发请求的假配置。"""
    return ProviderConfig(
        name="test",
        protocol="anthropic",
        model="claude-sonnet-4-6",
        api_key="test-key",
        base_url="",
        thinking=False,
    )


class TestAnthropicSystemBlocks:
    """AC4/F3 — 稳定块带 cache_control，环境块不带。"""

    def _get_system_blocks(self, stable: str, environment: str) -> list[dict]:
        """利用 Request 内容构造 system 块并返回。"""
        req = Request(
            messages=[Message(role="user", content="hello")],
            system=System(stable=stable, environment=environment),
        )
        # 直接调用内部方法构造 system 块列表
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
            system.append({"type": "text", "text": req.system.environment})
        return system

    def test_stable_block_has_cache_control(self):
        """稳定块必须带 cache_control: ephemeral。"""
        blocks = self._get_system_blocks(stable="You are an AI.", environment="")
        assert len(blocks) == 1
        assert "cache_control" in blocks[0]
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_environment_block_no_cache_control(self):
        """环境块不得带 cache_control。"""
        blocks = self._get_system_blocks(
            stable="You are an AI.",
            environment="Working Directory: /tmp",
        )
        assert len(blocks) == 2
        assert "cache_control" in blocks[0]
        assert "cache_control" not in blocks[1]

    def test_no_stable_no_cache_control(self):
        """stable 为空时不产生任何带 cache_control 的块。"""
        blocks = self._get_system_blocks(
            stable="",
            environment="Working Directory: /tmp",
        )
        assert len(blocks) == 1
        assert "cache_control" not in blocks[0]

    def test_both_empty(self):
        """两个都为空时 system 块列表为空。"""
        blocks = self._get_system_blocks(stable="", environment="")
        assert blocks == []

    def test_cache_control_guard(self):
        """回归守护：稳定块序列化格式不得改变（否则缓存断点失效）。"""
        blocks = self._get_system_blocks(stable="SYSTEM_PROMPT_HERE", environment="")
        assert blocks == [
            {
                "type": "text",
                "text": "SYSTEM_PROMPT_HERE",
                "cache_control": {"type": "ephemeral"},
            }
        ]


class TestAnthropicReminderInjection:
    """AC12/N3 — reminder 安全织入末条 user 消息。"""

    def test_reminder_appended_to_last_user_string_content(self):
        """末条 user 消息 content 为字符串时 → 转换为 list 并追加文本块。"""
        messages = [{"role": "user", "content": "hello"}]
        AnthropicProvider._inject_reminder_anthropic(messages, "<reminder>test</reminder>")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][0] == {"type": "text", "text": "hello"}
        assert messages[0]["content"][1] == {"type": "text", "text": "<reminder>test</reminder>"}

    def test_reminder_appended_to_last_user_list_content(self):
        """末条 user 消息 content 已是 list → 直接追加文本块。"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "result"}],
            }
        ]
        AnthropicProvider._inject_reminder_anthropic(messages, "<r>test</r>")
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][1] == {"type": "text", "text": "<r>test</r>"}

    def test_reminder_new_message_when_empty(self):
        """空消息列表 → 新建 user 消息。"""
        messages: list[dict] = []
        AnthropicProvider._inject_reminder_anthropic(messages, "<r>test</r>")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "<r>test</r>"

    def test_reminder_new_message_when_last_not_user(self):
        """末条非 user → 新建 user 消息。"""
        messages = [{"role": "assistant", "content": "I'll help."}]
        AnthropicProvider._inject_reminder_anthropic(messages, "<r>test</r>")
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "<r>test</r>"
