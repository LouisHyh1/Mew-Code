"""TUI rendering helpers for tool calls and status bar."""

from rich.padding import Padding
from rich.text import Text

from novacode.agent import Mode


def _compact_tok(n: int) -> str:
    """紧凑 token 数字格式：1.2k / 340 / 0。"""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def status_bar(
    name: str,
    model: str,
    mode: Mode,
    usage_in: int,
    usage_out: int,
) -> Text:
    """渲染单行状态栏：provider │ [PLAN] │ model │ ↑in ↓out tok。"""
    t = Text()
    t.append(f" {name} ", style="bold")
    if mode == Mode.PLAN:
        t.append("│ ", style="dim")
        t.append("PLAN", style="bold #ffa500")
        t.append(" ", style="dim")
    t.append(f"│ {model} ", style="dim")
    if usage_in > 0 or usage_out > 0:
        in_s = _compact_tok(usage_in)
        out_s = _compact_tok(usage_out)
        t.append("│ ", style="dim")
        t.append(f"↑{in_s}", style="#58a6ff")
        t.append(f" ↓{out_s}", style="#f85149")
        t.append(" tok", style="dim")
    return t


def tool_line(name: str, args: str) -> Text:
    """渲染工具行：● name(args)。"""
    return Text("● ", style="bold #875FFF") + Text(f"{name}({args})", style="bold")


def tool_result_summary(result: str, is_error: bool = False) -> Padding:
    """渲染工具结果摘要，缩进展示，过长截断（~8 行）。"""
    lines = result.splitlines()
    if len(lines) > 8:
        lines = lines[:8]
        lines.append("…")
    text_content = "\n".join(f"  ⎿  {line}" for line in lines)
    style = "bold red" if is_error else "dim"
    return Padding(Text(text_content, style=style), (0, 0, 0, 2))
