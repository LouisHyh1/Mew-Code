"""Tests for TUI approval interaction — Ch06 permission UI."""

import asyncio
from unittest.mock import MagicMock

import pytest

from novacode.agent import ApprovalRequest
from novacode.config import ProviderConfig
from novacode.permission import Mode, Outcome
from novacode.permission.engine import Engine
from novacode.permission.rule import RuleSet
from novacode.tool import Registry
from novacode.tui.app import (
    NovaCodeApp,
    SessionState,
    _outcome_for_index,
)


# ── helpers ──────────────────────────────────────────────────


def _make_engine() -> Engine:
    return Engine(
        root=".",
        blacklist=[],
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path="",
        _start_mode=Mode.BYPASS,
    )


def _make_app() -> NovaCodeApp:
    """Create a minimal NovaCodeApp suitable for testing."""
    cfg = ProviderConfig(
        name="test",
        protocol="openai",
        api_key="sk-test",
        base_url="http://localhost:8000",
        model="gpt-4",
    )
    return NovaCodeApp(
        providers=[cfg],
        registry=Registry(),
        version="test",
        engine=_make_engine(),
    )


def _request() -> ApprovalRequest:
    loop = asyncio.get_running_loop()
    return ApprovalRequest(
        name="Bash",
        args='echo "hello"',
        reason="default 模式下命令执行需确认",
        respond=loop.create_future(),
    )


# ── unit: outcome index ───────────────────────────────────────


class TestOutcomeIndex:
    def test_0_allow_once(self):
        assert _outcome_for_index(0) == Outcome.ALLOW_ONCE

    def test_1_allow_forever(self):
        assert _outcome_for_index(1) == Outcome.ALLOW_FOREVER

    def test_2_deny_once(self):
        assert _outcome_for_index(2) == Outcome.DENY_ONCE


# ── unit: _update_approving key dispatch ──────────────────────


class TestUpdateApproving:
    """Test the pure key→outcome dispatch logic without TUI rendering."""

    @pytest.mark.asyncio
    async def test_enter_commits_default_cursor_allow_once(self):
        """enter at cursor 0 → ALLOW_ONCE."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 0
        app.state = SessionState.APPROVING

        handled = app._update_approving("enter")
        assert handled is True
        assert app.pending is None
        assert app.approve_cursor == 0
        assert req.respond.done()
        assert req.respond.result() == Outcome.ALLOW_ONCE
        assert app.state == SessionState.STREAMING

    @pytest.mark.asyncio
    async def test_down_then_enter_commits_allow_forever(self):
        """down moves to 1, enter → ALLOW_FOREVER."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 0
        app.state = SessionState.APPROVING

        app._update_approving("down")  # cursor → 1
        assert app.approve_cursor == 1

        handled = app._update_approving("enter")
        assert handled is True
        assert req.respond.result() == Outcome.ALLOW_FOREVER

    @pytest.mark.asyncio
    async def test_key_3_deny_once(self):
        """3 → DENY_ONCE directly."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 0
        app.state = SessionState.APPROVING

        handled = app._update_approving("3")
        assert handled is True
        assert req.respond.result() == Outcome.DENY_ONCE

    @pytest.mark.asyncio
    async def test_key_y_allow_once(self):
        """y → ALLOW_ONCE."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 0
        app.state = SessionState.APPROVING

        handled = app._update_approving("y")
        assert handled is True
        assert req.respond.result() == Outcome.ALLOW_ONCE

    @pytest.mark.asyncio
    async def test_key_n_deny_once(self):
        """n → DENY_ONCE."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 0
        app.state = SessionState.APPROVING

        handled = app._update_approving("n")
        assert handled is True
        assert req.respond.result() == Outcome.DENY_ONCE

    @pytest.mark.asyncio
    async def test_key_d_deny_once(self):
        """d → DENY_ONCE."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 0
        app.state = SessionState.APPROVING

        handled = app._update_approving("d")
        assert handled is True
        assert req.respond.result() == Outcome.DENY_ONCE

    @pytest.mark.asyncio
    async def test_key_1_allow_once(self):
        """1 → ALLOW_ONCE."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 0
        app.state = SessionState.APPROVING

        handled = app._update_approving("1")
        assert handled is True
        assert req.respond.result() == Outcome.ALLOW_ONCE

    @pytest.mark.asyncio
    async def test_key_2_allow_forever(self):
        """2 → ALLOW_FOREVER."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 0
        app.state = SessionState.APPROVING

        handled = app._update_approving("2")
        assert handled is True
        assert req.respond.result() == Outcome.ALLOW_FOREVER

    @pytest.mark.asyncio
    async def test_space_commits_current_cursor(self):
        """space → commits cursor item."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 2
        app.state = SessionState.APPROVING

        handled = app._update_approving("space")
        assert handled is True
        assert req.respond.result() == Outcome.DENY_ONCE

    @pytest.mark.asyncio
    async def test_up_wraps_to_bottom(self):
        """up at 0 wraps to 2."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 0
        app.state = SessionState.APPROVING

        app._update_approving("up")
        assert app.approve_cursor == 2

    @pytest.mark.asyncio
    async def test_down_wraps_to_top(self):
        """down at 2 wraps to 0."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 2
        app.state = SessionState.APPROVING

        app._update_approving("down")
        assert app.approve_cursor == 0

    @pytest.mark.asyncio
    async def test_j_and_k_aliases(self):
        """j = down, k = up (vim keys)."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 0
        app.state = SessionState.APPROVING

        app._update_approving("j")
        assert app.approve_cursor == 1
        app._update_approving("k")
        assert app.approve_cursor == 0

    @pytest.mark.asyncio
    async def test_unhandled_key_returns_false(self):
        """Random key → False, state unchanged."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.approve_cursor = 0
        app.state = SessionState.APPROVING

        handled = app._update_approving("x")
        assert handled is False
        assert app.pending is not None  # still pending
        assert app.state == SessionState.APPROVING

    @pytest.mark.asyncio
    async def test_commit_clears_approval_widget_ref(self):
        """After commit, _approval_widget is None."""
        app = _make_app()
        req = _request()
        app.pending = req
        app.state = SessionState.APPROVING
        app._approval_widget = MagicMock()

        app._commit_approval(Outcome.ALLOW_ONCE)
        assert app._approval_widget is None

    @pytest.mark.asyncio
    async def test_enter_without_pending_noop(self):
        """No pending → enter is no-op."""
        app = _make_app()
        app.state = SessionState.APPROVING
        handled = app._update_approving("enter")
        assert handled is True
        # No crash, no side effect


# ── note: ChatInput._on_key delegation requires Textual App mount ──
# ── tested via end-to-end tmux session (see plan verification)  ──
# ── unit: _commit_approval state transitions ─────────────────


class TestCommitApproval:
    @pytest.mark.asyncio
    async def test_resets_state_to_streaming(self):
        app = _make_app()
        req = _request()
        app.pending = req
        app.state = SessionState.APPROVING

        app._commit_approval(Outcome.ALLOW_ONCE)
        assert app.state == SessionState.STREAMING
        assert app.pending is None
        assert app.approve_cursor == 0

    @pytest.mark.asyncio
    async def test_noop_when_pending_is_none(self):
        app = _make_app()
        app.pending = None
        app._commit_approval(Outcome.ALLOW_ONCE)
        # No crash.

    @pytest.mark.asyncio
    async def test_no_double_resolve(self):
        app = _make_app()
        req = _request()
        app.pending = req

        app._commit_approval(Outcome.ALLOW_ONCE)
        # Second call is noop
        app._commit_approval(Outcome.DENY_ONCE)

        assert req.respond.result() == Outcome.ALLOW_ONCE  # first value sticks
