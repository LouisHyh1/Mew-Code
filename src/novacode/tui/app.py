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
from novacode.agent import Agent, Mode, Phase
from novacode.config import ProviderConfig
from novacode.conversation import Conversation
from novacode.llm import Provider as LLMProvider
from novacode.llm import new_provider
from novacode.prompt import EXECUTE_DIRECTIVE
from novacode.tool import Registry
from novacode.tui.view import tool_line, tool_result_summary

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class ToolDisplay:
    name: str
    args: str


class SessionState(Enum):
    SELECTING = "selecting"
    IDLE = "idle"
    STREAMING = "streaming"


class ChatInput(TextArea):
    """TextArea: Enter submits, Shift+Enter inserts newline."""

    class Submitted(TMessage):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def _on_key(self, event: "events.Key") -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            if "shift" in getattr(event, "modifiers", ""):
                self.insert("\n")
            else:
                if self.text.strip():
                    self.post_message(self.Submitted(self.text))
                self.clear()


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
    ) -> None:
        super().__init__(driver_class=driver_class)
        self._version = version or __version__
        self.providers = providers
        self.provider: LLMProvider | None = None
        self.conv = Conversation()
        self._tool_registry = registry
        self.state = SessionState.SELECTING if len(providers) > 1 else SessionState.IDLE
        self.turn_start = 0.0
        self._agent_task: asyncio.Task[None] | None = None
        self.mode: Mode = Mode.NORMAL
        self.iter: int = 0
        self.usage_in: int = 0
        self.usage_out: int = 0
        self.cur_tools: list[ToolDisplay] = []
        self.turn_cancel: asyncio.Event | None = None
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
        # 优先复制输入框中的选中文本
        try:
            inp = self.query_one("#chat-input", ChatInput)
            if inp.selected_text:
                self.copy_to_clipboard(inp.selected_text)
                self._flash_copy_feedback()
                return
        except Exception:
            pass
        # 否则复制最后 AI 回复
        if self._last_ai_text:
            try:
                self.copy_to_clipboard(self._last_ai_text)
                self._flash_copy_feedback()
            except Exception:
                pass

    def _flash_copy_feedback(self) -> None:
        """复制反馈——短暂高亮模式标签。"""
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
        if self.mode == Mode.PLAN:
            label.update(Text("  plan", style="bold #ffa500"))
        else:
            label.update(Text("  default", style="dim"))

    # ── provider selector ──────────────────────────────────────

    @on(OptionList.OptionSelected)
    async def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "provider-list":
            return
        index = event.option_index
        assert index is not None
        self._select_provider(self.providers[index])

    # ── keys ────────────────────────────────────────────────────

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

        if self.state == SessionState.STREAMING:
            if self._agent_task and not self._agent_task.done():
                self._agent_task.cancel()
            self._show_system("(response interrupted)")
            self._finish_streaming()
            return
        self.exit()

    def action_cancel(self) -> None:
        if self.state == SessionState.STREAMING and self.turn_cancel is not None:
            self.turn_cancel.set()

    def action_toggle_tool_blocks(self) -> None:
        """Ctrl+O 切换所有工具块展开/折叠（预留）。"""
        pass

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
            self.mode = Mode.NORMAL
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

        # 在聊天区挂载 spinner
        chat = self.query_one("#chat-area", VerticalScroll)
        self._spinner_idx = 0
        self._spinner_label = Static(f"  {SPINNER_FRAMES[0]} Thinking…", id="spinner-live")
        await chat.mount(self._spinner_label)
        self._scroll_chat()
        self._start_spinner()

        self._agent_task = asyncio.create_task(
            self._consume_events(
                Agent(self.provider, self._tool_registry).run(
                    self.conv, self.mode, self.turn_cancel
                )
            )
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

                if ev.tool is not None and ev.tool.phase == Phase.START:
                    # 提交 preamble
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
        """将工具调用前的文本作为 Markdown 提交。"""
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

        # done 标签
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
