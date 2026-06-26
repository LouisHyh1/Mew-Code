"""系统提示工程化——模块化装配、环境采集、补充消息注入。

ch05 把全局指令拆成按职责划分、按优先级拼装的模块；区分稳定与变化内容；
提供 system-reminder 补充消息注入机制；规划模式提醒按轮次注入。
"""

import os

from novacode import __version__
from novacode.prompt.environment import Environment, gather_environment
from novacode.prompt.modules import Module, fixed_modules, optional_modules
from novacode.prompt.reminder import (
    EXECUTE_DIRECTIVE,
    plan_reminder,
    system_reminder,
)

# ── ASCII banner（保留自 ch04）────────────────────────────────

CAT = r"""
  /\_/\
 ( o.o )
  > ^ <
"""


def render_banner(version: str | None = None, cwd: str | None = None) -> str:
    """渲染启动 banner（含版本号、工作目录、ASCII 猫）。"""
    v = version or __version__
    d = cwd or os.getcwd()
    return f"""{CAT}
  NovaCode v{v}
  {d}

Ready — type a message or /exit to quit.
"""


# ── 核心装配逻辑 ──────────────────────────────────────────────


def assemble_system(mods: list[Module]) -> str:
    """按 priority 升序排列模块，跳过空 content，以双换行连接。

    排序使用稳定排序（Python sorted 已稳定），priority 相同时保持
    传入顺序。跳过 content == "" 的可选空槽，不留多余空行。

    此函数只操作常量内容 → 跨轮输出逐字节一致（N1 缓存确定性）。
    """
    sorted_mods = sorted(mods, key=lambda m: m.priority)
    parts = [m.content for m in sorted_mods if m.content]
    return "\n\n".join(parts)


def build_system_prompt() -> str:
    """装配完整稳定系统提示（固定模块 + 可选空槽）。

    可选空槽 content 为空 → 自动跳过，不产生多余空行。
    此函数输出逐字节确定，不含环境 / 时间相关内容（N1）。
    """
    return assemble_system(fixed_modules() + optional_modules())


__all__ = [
    # 模块化
    "Module",
    "fixed_modules",
    "optional_modules",
    "assemble_system",
    "build_system_prompt",
    # 环境
    "Environment",
    "gather_environment",
    # 补充消息
    "system_reminder",
    "plan_reminder",
    "EXECUTE_DIRECTIVE",
    # banner
    "CAT",
    "render_banner",
]
