"""TUI rendering helpers for tool calls, status bar, and approval block."""

from rich.padding import Padding
from rich.text import Text

from novacode.agent import ApprovalRequest
from novacode.permission import Mode


def _compact_tok(n: int) -> str:
    """紧凑 token 数字格式：1.2k / 340 / 0。"""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def status_bar(
    mode: Mode,
    model: str,
    usage_in: int,
    usage_out: int,
) -> Text:
    """渲染单行状态栏：左侧常驻权限模式（取代 provider 名）│ model │ ↑in ↓out tok。"""
    t = Text()
    mode_styles = {
        Mode.DEFAULT: "dim",
        Mode.ACCEPT_EDITS: "bold #58a6ff",
        Mode.PLAN: "bold #ffa500",
        Mode.BYPASS: "bold #f85149",
    }
    style = mode_styles.get(mode, "dim")
    t.append(f" {mode.label()} ", style=style)
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


def approval_block(req: ApprovalRequest, cursor: int) -> Text:
    """渲染人在回路待批准块——多行菜单 + 光标高亮。

    cursor: 0=允许本次, 1=永久允许, 2=拒绝本次
    """
    t = Text()
    # 工具名 + 参数
    t.append("● ", style="bold #875FFF")
    t.append(f"{req.name}", style="bold #c9d1d9")
    if req.args:
        t.append(f"\n   参数: {req.args}", style="dim")
    # 触发原因
    t.append(f"\n   {req.reason}", style="dim")
    t.append("\n")
    t.append("\n   是否继续?", style="bold")
    t.append("\n")

    # 菜单项
    items = [
        ("1. 允许本次", "本次执行，不记忆"),
        ("2. 永久允许（写入本地配置）", "本次执行 + 写入规则，下次自动放行"),
        ("3. 拒绝本次", "将拒绝原因回灌给模型"),
    ]

    for idx, (label, _desc) in enumerate(items):
        prefix = " > " if idx == cursor else "   "
        label_style = "bold #875FFF" if idx == cursor else "dim"
        desc_style = "dim" if idx == cursor else "dim"
        t.append(f"{prefix}{label}", style=label_style)
        t.append(f"  {_desc}\n", style=desc_style)

    t.append("\n", style="")
    t.append("↑↓ 选择 · 回车确认 · Esc 取消", style="dim italic")
    return t
