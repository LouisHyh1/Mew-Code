"""Textual TUI application — NovaCodeApp."""

import asyncio
import os
import time
from dataclasses import dataclass
from enum import Enum

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as TMessage
from textual.widgets import Markdown, OptionList, Static, TextArea

from novacode import __version__
from novacode.agent import Agent, ApprovalRequest, Phase
from novacode.config import ProviderConfig
from novacode.conversation import Conversation
from novacode.llm import Provider as LLMProvider
from novacode.llm import new_provider
from novacode.permission import Mode, Outcome
from novacode.permission.engine import Engine
from novacode.prompt import EXECUTE_DIRECTIVE
from novacode.tool import Registry
from novacode.tui.view import approval_block, tool_line, tool_result_summary

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class ToolDisplay:
    name: str
    args: str


class SessionState(Enum):
    SELECTING = "selecting"
    IDLE = "idle"
    STREAMING = "streaming"
    APPROVING = "approving"


class ChatInput(TextArea):
    """TextArea: Enter submits, Shift+Enter inserts newline."""

    class Submitted(TMessage):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def _on_key(self, event: "events.Key") -> None:
        # 审批态：按键委托给 App._update_approving，防止被 TextArea 吞掉
        if self.app.state == SessionState.APPROVING:
            self.app._update_approving(event.key)
            event.stop()
            event.prevent_default()
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            if "shift" in getattr(event, "modifiers", ""):
                self.insert("\n")
            else:
                if self.text.strip():
                    self.post_message(self.Submitted(self.text))
                self.clear()


def _next_mode(m: Mode) -> Mode:
    """循环切换四档模式：DEFAULT → ACCEPT_EDITS → PLAN → BYPASS → DEFAULT。"""
    return Mode((int(m) + 1) % 4)


def _outcome_for_index(idx: int) -> Outcome:
    """菜单索引 → Outcome。0=ALLOW_ONCE, 1=ALLOW_FOREVER, 2=DENY_ONCE。"""
    return [Outcome.ALLOW_ONCE, Outcome.ALLOW_FOREVER, Outcome.DENY_ONCE][idx]


class NovaCodeApp(App):
    CSS_PATH = "styles.tcss"
    TITLE = "NovaCode"

    BINDINGS = [
        Binding("ctrl+c", "handle_ctrl_c", "Ctrl+C", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+o", "toggle_tool_blocks", "Toggle tools", priority=True),
    ]

    def __init__(
        self,
        providers: list[ProviderConfig],
        registry: Registry,
        version: str | None = None,
        driver_class: type | None = None,
        engine: Engine | None = None,
    ) -> None:
        super().__init__(driver_class=driver_class)
        self._version = version or __version__
        self.providers = providers
        self.provider: LLMProvider | None = None
        self.conv = Conversation()
        self._tool_registry = registry
        self.engine = engine
        self.state = SessionState.SELECTING if len(providers) > 1 else SessionState.IDLE
        self.turn_start = 0.0
        self._agent_task: asyncio.Task[None] | None = None
        self.mode: Mode = engine.start_mode if engine else Mode.DEFAULT
        self.iter: int = 0
        self.usage_in: int = 0
        self.usage_out: int = 0
        self.cur_tools: list[ToolDisplay] = []
        self.turn_cancel: asyncio.Event | None = None
        # 人在回路状态
        self.pending: ApprovalRequest | None = None
        self.approve_cursor: int = 0
        self._approval_widget: Static | None = None
        # 流式渲染状态
        self._current_ai_row: Vertical | None = None
        self._streaming_label: Static | None = None
        self._accumulated_text: str = ""
        self._spinner_label: Static | None = None
        self._spinner_idx: int = 0
        self._spinner_timer = None
        self._last_ai_text: str = ""

    # ── compose ─────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Static(self._make_banner(), id="title-bar")
        if len(self.providers) > 1:
            with Vertical(id="provider-select"):
                yield Static("Select a Provider", id="select-label")
                yield OptionList(
                    *[f"{p.name}  [{p.model}]" for p in self.providers],
                    id="provider-list",
                )
        yield VerticalScroll(id="chat-area")
        with Vertical(id="input-area"):
            yield ChatInput(id="chat-input")
            with Horizontal(id="status-bar"):
                yield Static("", id="mode-label")
                yield Static("", id="model-label")

    @staticmethod
    def _make_banner(model: str = "", work_dir: str = "") -> Text:
        t = Text()
        t.append(" /\\_/\\    ", style="bold #875FFF")
        t.append(f"NovaCode v{__version__}\n", style="#c9d1d9")
        t.append("( o.o )   ", style="bold #875FFF")
        t.append(f"{model}\n" if model else "\n", style="#c9d1d9")
        t.append(" > ^ <    ", style="bold #875FFF")
        t.append(work_dir, style="#c9d1d9")
        return t

    def on_mount(self) -> None:
        if len(self.providers) == 1:
            self._select_provider(self.providers[0])
        else:
            self.query_one("#chat-area").display = False
            self.query_one("#input-area").display = False

    def _select_provider(self, provider_cfg: ProviderConfig) -> None:
        self.provider = new_provider(provider_cfg)
        self._update_mode_label()
        work_dir = os.getcwd()
        self.query_one("#title-bar", Static).update(self._make_banner(provider_cfg.model, work_dir))
        self.query_one("#model-label", Static).update(provider_cfg.model)

        select = self.query("#provider-select")
        if select:
            select.first().display = False
        self.query_one("#chat-area").display = True
        self.query_one("#input-area").display = True
        self.query_one("#chat-input", ChatInput).focus()
        self.state = SessionState.IDLE

    # ── right-click copy ───────────────────────────────────────

    def on_click(self, event: events.Click) -> None:
        """右键 → 智能复制：输入框内复制选中文本，否则复制最后 AI 回复。"""
        if event.button != 3:
            return
        try:
            inp = self.query_one("#chat-input", ChatInput)
            if inp.selected_text:
                self.copy_to_clipboard(inp.selected_text)
                self._flash_copy_feedback()
                return
        except Exception:
            pass
        if self._last_ai_text:
            try:
                self.copy_to_clipboard(self._last_ai_text)
                self._flash_copy_feedback()
            except Exception:
                pass

    def _flash_copy_feedback(self) -> None:
        try:
            label = self.query_one("#mode-label", Static)
            label.update(Text("  Copied!", style="bold #875FFF"))
            self.set_timer(1.5, self._update_mode_label)
        except Exception:
            pass

    # ── mode label ─────────────────────────────────────────────

    def _update_mode_label(self) -> None:
        try:
            label = self.query_one("#mode-label", Static)
        except Exception:
            return
        label.update(Text(f"  {self.mode.label()}", style=self._mode_style()))

    def _mode_style(self) -> str:
        styles = {
            Mode.DEFAULT: "dim",
            Mode.ACCEPT_EDITS: "bold #58a6ff",
            Mode.PLAN: "bold #ffa500",
            Mode.BYPASS: "bold #f85149",
        }
        return styles.get(self.mode, "dim")

    # ── provider selector ──────────────────────────────────────

    @on(OptionList.OptionSelected)
    async def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "provider-list":
            return
        index = event.option_index
        assert index is not None
        self._select_provider(self.providers[index])

    # ── keys ────────────────────────────────────────────────────

    def _on_key(self, event: "events.Key") -> None:
        """全局按键分派——处理 Shift+Tab 和 approving 态按键。"""
        key = event.key

        # Shift+Tab 循环切换权限模式（仅 idle 态）
        if key == "shift+tab" and self.state == SessionState.IDLE:
            event.stop()
            event.prevent_default()
            self.mode = _next_mode(self.mode)
            self._update_mode_label()
            self._show_system(f"已切换到 {self.mode.label()} 模式")
            return

        # Approving 态按键分派
        if self.state == SessionState.APPROVING:
            handled = self._update_approving(key)
            if handled:
                event.stop()
                event.prevent_default()
                return

        # 默认处理链
        super()._on_key(event)

    async def action_handle_ctrl_c(self) -> None:
        # 输入框中有选中文本 → 优先复制，不触发取消/退出
        try:
            inp = self.query_one("#chat-input", ChatInput)
            if inp.selected_text:
                self.copy_to_clipboard(inp.selected_text)
                self._flash_copy_feedback()
                return
        except Exception:
            pass

        # STREAMING 或 APPROVING 态 → 取消本轮
        if self.state in (SessionState.STREAMING, SessionState.APPROVING):
            if self.state == SessionState.APPROVING and self.pending is not None:
                if not self.pending.respond.done():
                    self.pending.respond.set_result(Outcome.DENY_ONCE)
            if self._agent_task and not self._agent_task.done():
                self._agent_task.cancel()
            self._show_system("(response interrupted)")
            self._finish_streaming()
            return
        self.exit()

    def action_cancel(self) -> None:
        if self.state == SessionState.STREAMING and self.turn_cancel is not None:
            self.turn_cancel.set()
        elif self.state == SessionState.APPROVING and self.pending is not None:
            if not self.pending.respond.done():
                self.pending.respond.set_result(Outcome.DENY_ONCE)

    def action_toggle_tool_blocks(self) -> None:
        """Ctrl+O 切换所有工具块展开/折叠（预留）。"""
        pass

    # ── approving 交互 ────────────────────────────────────────

    def _update_approving(self, key: str) -> bool:
        """处理 approving 态按键。返回 True 表示已处理。"""
        if key in ("up", "k"):
            self.approve_cursor = (self.approve_cursor - 1) % 3
            self._refresh_approval_view()
            return True
        if key in ("down", "j"):
            self.approve_cursor = (self.approve_cursor + 1) % 3
            self._refresh_approval_view()
            return True
        if key in ("enter", "space"):
            self._commit_approval(_outcome_for_index(self.approve_cursor))
            return True
        if key == "1":
            self._commit_approval(Outcome.ALLOW_ONCE)
            return True
        if key == "2":
            self._commit_approval(Outcome.ALLOW_FOREVER)
            return True
        if key == "3":
            self._commit_approval(Outcome.DENY_ONCE)
            return True
        if key == "y":
            self._commit_approval(Outcome.ALLOW_ONCE)
            return True
        if key in ("n", "d"):
            self._commit_approval(Outcome.DENY_ONCE)
            return True
        return False

    def _commit_approval(self, outcome: Outcome) -> None:
        """提交人在回路决策并恢复 STREAMING。"""
        if self.pending is None:
            return
        pending = self.pending
        self.pending = None
        self.approve_cursor = 0
        self._approval_widget = None
        self.state = SessionState.STREAMING
        if not pending.respond.done():
            pending.respond.set_result(outcome)

    def _refresh_approval_view(self) -> None:
        """刷新待批准块——更新已有 widget 而非重复 mount。"""
        if self.pending is None or self._current_ai_row is None:
            return
        block = approval_block(self.pending, self.approve_cursor)
        if self._approval_widget is not None:
            self._approval_widget.update(block)
        else:
            # 兜底：widget 被意外清空时重新 mount
            self._approval_widget = Static(block, classes="approval-block")
            asyncio.ensure_future(self._current_ai_row.mount(self._approval_widget))
        self._scroll_chat()

    # ── submit ─────────────────────────────────────────────────

    @on(ChatInput.Submitted)
    async def _on_submit(self, event: ChatInput.Submitted) -> None:
        if self.state != SessionState.IDLE or self.provider is None:
            return
        text = event.text.strip() if event.text else ""
        if not text:
            return

        if text == "/exit":
            self.exit()
            return

        if text == "/plan":
            self.mode = Mode.PLAN
            self._update_mode_label()
            self._show_system("已进入计划模式（只读工具）。输入需求，我会先调研再给出分步计划。")
            return

        if text == "/do":
            self.mode = Mode.DEFAULT
            self._update_mode_label()
            self.conv.add_user(EXECUTE_DIRECTIVE)
            await self._start_stream()
            return

        await self._dispatch(text)

    async def _dispatch(self, text: str) -> None:
        self.conv.add_user(text)
        chat = self.query_one("#chat-area", VerticalScroll)
        # 用户消息
        user_row = Vertical(classes="user-row")
        await chat.mount(user_row)
        rich_text = Text()
        rich_text.append("❯ ", style="bold #58a6ff")
        rich_text.append(text, style="bold #c9d1d9")
        user_bubble = Static(rich_text, classes="message user-message")
        await user_row.mount(user_bubble)
        self._scroll_chat()
        await self._start_stream()

    async def _start_stream(self) -> None:
        self.state = SessionState.STREAMING
        self.turn_start = time.monotonic()
        self.iter = 0
        self.cur_tools = []
        self.turn_cancel = asyncio.Event()
        self._accumulated_text = ""
        self._current_ai_row = None
        self._streaming_label = None
        self._approval_widget = None
        self.pending = None
        self.approve_cursor = 0

        chat = self.query_one("#chat-area", VerticalScroll)
        self._spinner_idx = 0
        self._spinner_label = Static(f"  {SPINNER_FRAMES[0]} Thinking…", id="spinner-live")
        await chat.mount(self._spinner_label)
        self._scroll_chat()
        self._start_spinner()

        agent = Agent(self.provider, self._tool_registry, self._version, self.engine)
        self._agent_task = asyncio.create_task(
            self._consume_events(agent.run(self.conv, self.mode, self.turn_cancel))
        )

    # ── spinner ────────────────────────────────────────────────

    def _start_spinner(self) -> None:
        if self._spinner_timer is not None:
            return
        self._spinner_timer = self.set_interval(0.08, self._tick_spinner)

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
        if self._spinner_label is not None:
            self._spinner_label.remove()
            self._spinner_label = None

    def _tick_spinner(self) -> None:
        self._spinner_idx += 1
        frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
        elapsed = time.monotonic() - self.turn_start
        if self._spinner_label is not None:
            self._spinner_label.update(f"  {frame} Thinking…  ({elapsed:.0f}s)")
            if self._spinner_idx % 5 == 0:
                self._scroll_chat()

    # ── consume agent events ───────────────────────────────────

    async def _consume_events(self, agent_gen) -> None:
        try:
            async for ev in agent_gen:
                if ev.err is not None:
                    self._finish_with_error(ev.err)
                    return

                if ev.approval is not None:
                    # 人在回路——切到 APPROVING 态
                    if self._accumulated_text.strip():
                        self._flush_preamble()
                    self.pending = ev.approval
                    self.approve_cursor = 0
                    self.state = SessionState.APPROVING
                    self._mount_approval_block()
                    # 不要继续读事件——agent 正 await respond
                    continue

                if ev.tool is not None and ev.tool.phase == Phase.START:
                    if self._accumulated_text.strip():
                        self._flush_preamble()
                    self.cur_tools.append(ToolDisplay(name=ev.tool.name, args=ev.tool.args))
                elif ev.tool is not None and ev.tool.phase == Phase.END:
                    td = (
                        self.cur_tools.pop(0)
                        if self.cur_tools
                        else ToolDisplay(name=ev.tool.name, args=ev.tool.args)
                    )
                    self._mount_tool_block(ev.tool.name, td.args, ev.tool.result, ev.tool.is_error)

                if ev.usage is not None:
                    self.usage_in += ev.usage.input
                    self.usage_out += ev.usage.output

                if ev.notice:
                    self._show_system(ev.notice)

                if ev.iter > 0:
                    self.iter = ev.iter

                if ev.done:
                    self._finish_with_assistant(self._accumulated_text)
                    return

                if ev.text:
                    self._accumulated_text += ev.text
                    self._update_streaming_label()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._finish_with_error(e)

    def _flush_preamble(self) -> None:
        text = self._accumulated_text
        self._accumulated_text = ""
        self._ensure_ai_row()
        if self._streaming_label is not None:
            self._streaming_label.remove()
            self._streaming_label = None
        md = Markdown(text, classes="message ai-message")
        if self._current_ai_row is not None:
            asyncio.ensure_future(self._current_ai_row.mount(md))

    def _ensure_ai_row(self) -> None:
        if self._current_ai_row is None:
            chat = self.query_one("#chat-area", VerticalScroll)
            self._current_ai_row = Vertical(classes="ai-row")
            asyncio.ensure_future(chat.mount(self._current_ai_row))
            if self._streaming_label is None:
                self._streaming_label = Static("", classes="message ai-message")
                asyncio.ensure_future(self._current_ai_row.mount(self._streaming_label))

    def _update_streaming_label(self) -> None:
        self._ensure_ai_row()
        if self._streaming_label is not None:
            t = Text()
            t.append("● ", style="bold #875FFF")
            t.append(self._accumulated_text)
            self._streaming_label.update(t)
        self._scroll_chat()

    def _mount_tool_block(self, name: str, args: str, result: str, is_error: bool) -> None:
        self._ensure_ai_row()
        line = tool_line(name, args)
        summary = tool_result_summary(result, is_error)
        if self._current_ai_row is not None:
            asyncio.ensure_future(self._current_ai_row.mount(Static(line, classes="tool-block")))
            asyncio.ensure_future(
                self._current_ai_row.mount(Static(summary, classes="tool-detail"))
            )
        self._scroll_chat()

    def _mount_approval_block(self) -> None:
        """挂载人在回路待批准块——首次创建并保存 widget 引用。"""
        self._ensure_ai_row()
        if self.pending is None:
            return
        block = approval_block(self.pending, self.approve_cursor)
        self._approval_widget = Static(block, classes="approval-block")
        if self._current_ai_row is not None:
            asyncio.ensure_future(
                self._current_ai_row.mount(self._approval_widget)
            )
        self._scroll_chat()

    def _finish_with_assistant(self, reply: str) -> None:
        self._stop_spinner()
        elapsed = time.monotonic() - self.turn_start
        self._last_ai_text = reply.strip()

        if reply.strip():
            self._ensure_ai_row()
            if self._streaming_label is not None:
                self._streaming_label.remove()
                self._streaming_label = None
            md = Markdown(reply, classes="message ai-message")
            if self._current_ai_row is not None:
                asyncio.ensure_future(self._current_ai_row.mount(md))

        verb = "Thinking"
        done_text = Text(f"✻ {verb}d for {elapsed:.1f}s", style="dim italic")
        done_label = Static(done_text, classes="message thinking-done")
        if self._current_ai_row is not None:
            asyncio.ensure_future(self._current_ai_row.mount(done_label))

        self._scroll_chat()
        self._finish_streaming()

    def _finish_with_error(self, err: Exception) -> None:
        self._stop_spinner()
        name = type(err).__name__
        self._show_system(f"✖ {name}: {err}")
        self._finish_streaming()

    def _finish_streaming(self) -> None:
        self._stop_spinner()
        self._agent_task = None
        self.state = SessionState.IDLE
        self.cur_tools = []
        self.iter = 0
        self.turn_cancel = None
        self._current_ai_row = None
        self._streaming_label = None
        self._accumulated_text = ""
        self._approval_widget = None
        self.pending = None
        self.approve_cursor = 0
        try:
            inp = self.query_one("#chat-input", ChatInput)
            inp.focus()
        except Exception:
            pass

    # ── helpers ────────────────────────────────────────────────

    def _scroll_chat(self) -> None:
        try:
            chat = self.query_one("#chat-area", VerticalScroll)
            self.call_after_refresh(chat.scroll_end, animate=False)
        except Exception:
            pass

    def _show_system(self, text: str) -> None:
        chat = self.query_one("#chat-area", VerticalScroll)
        widget = Static(Text(f"  {text}", style="dim"), classes="message system-message")
        chat.mount(widget)
        self._scroll_chat()
