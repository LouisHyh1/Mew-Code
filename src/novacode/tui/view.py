"""TUI rendering helpers for tool calls."""

from rich.padding import Padding
from rich.text import Text


def tool_line(name: str, args: str) -> Text:
    """渲染 Claude Code 风格工具行：● name(args)。"""
    return Text("● ", style="bold cyan") + Text(f"{name}({args})", style="bold")


def tool_result_summary(result: str, is_error: bool = False) -> Padding:
    """渲染工具结果摘要，缩进展示，过长截断（~8 行）。"""
    lines = result.splitlines()
    if len(lines) > 8:
        lines = lines[:8]
        lines.append("…")
    text = "\n".join(f"  ⎿  {line}" for line in lines)
    style = "bold red" if is_error else "dim"
    return Padding(Text(text, style=style), (0, 0, 0, 2))
