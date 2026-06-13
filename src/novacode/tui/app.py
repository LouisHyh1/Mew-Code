"""Textual TUI application — NovaCodeApp with state machine and tool support."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum

from rich.markdown import Markdown
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message as TMessage
from textual.widgets import OptionList, RichLog, Static, TextArea

from novacode import __version__
from novacode.agent import Agent, Phase
from novacode.config import ProviderConfig
from novacode.conversation import Conversation
from novacode.llm import Provider as LLMProvider
from novacode.llm import new_provider
from novacode.prompt import render_banner
from novacode.tool import Registry
from novacode.tui.view import tool_line, tool_result_summary


@dataclass
class ToolDisplay:
    name: str
    args: str


class SessionState(Enum):
    SELECTING = "selecting"
    IDLE = "idle"
    STREAMING = "streaming"


class InputArea(TextArea):
    """TextArea: Enter submits, Shift+Enter inserts newline."""

    class SubmitMessage(TMessage):
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
                    self.post_message(self.SubmitMessage(self.text))
                self.clear()


class NovaCodeApp(App):
    CSS = """
    Screen {
        background: #0d1117;
    }

    #dialog {
        height: 1fr;
        border: none;
    }

    #streaming {
        height: auto;
        min-height: 0;
        color: #c9d1d9;
        padding: 0 1;
    }

    #input {
        height: auto;
        min-height: 3;
        max-height: 12;
        border: solid #30363d;
        margin: 0 1;
    }

    #statusbar {
        height: 1;
        background: #161b22;
        color: #8b949e;
        padding: 0 1;
    }

    #timer {
        height: 1;
        color: #58a6ff;
        padding: 0 1;
    }

    .user {
        color: #58a6ff;
    }

    .assistant {
        color: #c9d1d9;
    }

    .error {
        color: #f85149;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(
        self,
        providers: list[ProviderConfig],
        registry: Registry,
        version: str | None = None,
    ) -> None:
        super().__init__()
        self._version = version or __version__
        self.providers = providers
        self.provider: LLMProvider | None = None
        self.conv = Conversation()
        self._tool_registry = registry
        self.state = SessionState.SELECTING if len(providers) > 1 else SessionState.IDLE
        self.cur_reply = ""
        self.turn_start = 0.0
        self._stream_task: asyncio.Task[None] | None = None
        self._cur_tool: ToolDisplay | None = None

    def compose(self) -> ComposeResult:
        yield RichLog(id="dialog", wrap=True, markup=True, highlight=True)
        yield Static("", id="timer")
        yield InputArea(id="input")
        yield Static("", id="statusbar")

    def on_mount(self) -> None:
        banner = render_banner(self._version)
        log = self.query_one("#dialog", RichLog)
        log.write(banner)

        if len(self.providers) == 1:
            self.provider = new_provider(self.providers[0])
            self._update_statusbar()
        else:
            self._show_selector()

    # ── status bar ─────────────────────────────────────────────

    def _update_statusbar(self) -> None:
        if self.provider is None:
            return
        bar = self.query_one("#statusbar", Static)
        bar.update(f" {self.provider.name}  │  {self.provider.model} ")

    # ── provider selector ──────────────────────────────────────

    def _show_selector(self) -> None:
        self.state = SessionState.SELECTING
        options = [f"{p.name}  ({p.model})" for p in self.providers]
        self._selector = OptionList(*options)
        self._selector.id = "selector"
        self.mount(self._selector)
        self._selector.focus()

    @on(OptionList.OptionSelected)
    async def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        index = event.option_index
        assert index is not None
        self.provider = new_provider(self.providers[index])
        self._update_statusbar()
        await self._selector.remove()
        self.query_one("#input", InputArea).focus()
        self.state = SessionState.IDLE

    # ── submit ─────────────────────────────────────────────────

    @on(InputArea.SubmitMessage)
    async def _on_submit(self, event: InputArea.SubmitMessage) -> None:
        if self.state != SessionState.IDLE or self.provider is None:
            return
        text = event.text.strip() if event.text else ""
        if not text:
            return
        if text == "/exit":
            await self.action_quit()
            return
        await self._submit(text)

    async def _submit(self, text: str) -> None:
        self.state = SessionState.STREAMING
        self.conv.add_user(text)
        log = self.query_one("#dialog", RichLog)
        log.write(Text(f"❯ {text}", style="bold blue"))

        self.cur_reply = ""
        self.turn_start = time.monotonic()
        self._stream_task = asyncio.create_task(self._consume_agent_events())

    # ── consume agent events ───────────────────────────────────

    async def _consume_agent_events(self) -> None:
        assert self.provider is not None
        streaming = self.query_one("#timer", Static)

        try:
            agent = Agent(self.provider, self._tool_registry)
            async for ev in agent.run(self.conv):
                if ev.err is not None:
                    self._finish_with_error(ev.err)
                    return
                if ev.text:
                    self.cur_reply += ev.text
                    elapsed = time.monotonic() - self.turn_start
                    if self._cur_tool is not None:
                        streaming.update(
                            f"● {self._cur_tool.name}({self._cur_tool.args}) "
                            f"Running… ({elapsed:.0f}s)"
                        )
                    else:
                        streaming.update(f"● NovaCode  Imagining… ({elapsed:.0f}s)")
                if ev.tool is not None and ev.tool.phase == Phase.START:
                    # 先提交 preamble 到 scrollback，然后开始工具行
                    if self.cur_reply.strip():
                        log = self.query_one("#dialog", RichLog)
                        log.write(Markdown(self.cur_reply))
                        self.cur_reply = ""
                    self._cur_tool = ToolDisplay(
                        name=ev.tool.name, args=ev.tool.args
                    )
                if ev.tool is not None and ev.tool.phase == Phase.END:
                    log = self.query_one("#dialog", RichLog)
                    args_display = self._cur_tool.args if self._cur_tool else ev.tool.args
                    log.write(tool_line(ev.tool.name, args_display))
                    log.write(tool_result_summary(ev.tool.result, ev.tool.is_error))
                    self._cur_tool = None
                if ev.done:
                    self._finish_with_assistant(self.cur_reply)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._finish_with_error(e)

    def _finish_with_assistant(self, reply: str) -> None:
        elapsed = time.monotonic() - self.turn_start
        log = self.query_one("#dialog", RichLog)
        streaming = self.query_one("#timer", Static)

        if reply.strip():
            md = Markdown(reply)
            log.write(md)
        log.write(Text(f"  ({elapsed:.1f}s)", style="dim"))

        self.conv.add_assistant(reply)
        streaming.update("")
        self._stream_task = None
        self.state = SessionState.IDLE
        self._cur_tool = None
        self.query_one("#input", InputArea).focus()

    def _finish_with_error(self, err: Exception) -> None:
        log = self.query_one("#dialog", RichLog)
        streaming = self.query_one("#timer", Static)
        name = type(err).__name__
        log.write(Text(f"✖ {name}: {err}", style="bold red"))
        streaming.update("")
        self._stream_task = None
        self.state = SessionState.IDLE
        self._cur_tool = None
        self.query_one("#input", InputArea).focus()

    # ── quit ───────────────────────────────────────────────────

    async def action_quit(self) -> None:
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        self.exit()
