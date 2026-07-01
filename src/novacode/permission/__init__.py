"""权限系统——五层防御：黑名单 → 沙箱 → 规则引擎 → 模式兜底 → 人在回路。"""

from enum import IntEnum


class Mode(IntEnum):
    """四档权限模式。"""

    DEFAULT = 0  # 只读 Allow / 文件写 Ask / 命令执行 Ask
    ACCEPT_EDITS = 1  # 文件写 Allow / 命令执行 Ask
    PLAN = 2  # 仅只读工具可见（沿用 ch04）；矩阵同 default 作防御兜底
    BYPASS = 3  # 全 Allow（黑名单/沙箱仍拦）

    def __str__(self) -> str:
        return _MODE_NAMES[self]

    def label(self) -> str:
        return _MODE_LABELS[self]


_MODE_NAMES = {
    Mode.DEFAULT: "default",
    Mode.ACCEPT_EDITS: "acceptEdits",
    Mode.PLAN: "plan",
    Mode.BYPASS: "bypassPermissions",
}
_MODE_LABELS = {
    Mode.DEFAULT: "DEFAULT",
    Mode.ACCEPT_EDITS: "ACCEPT EDITS",
    Mode.PLAN: "PLAN",
    Mode.BYPASS: "BYPASS",
}


def parse_mode(s: str) -> tuple[Mode, bool]:
    """大小写不敏感识别四档名；未知返回 (Mode.DEFAULT, False)。"""
    if not s:
        return Mode.DEFAULT, False
    lower = s.strip().lower()
    for m, name in _MODE_NAMES.items():
        if name.lower() == lower:
            return m, True
    return Mode.DEFAULT, False


class Decision(IntEnum):
    ALLOW = 0
    DENY = 1
    ASK = 2


class Category(IntEnum):
    READ = 0
    WRITE = 1
    EXEC = 2


class Outcome(IntEnum):
    """人在回路三选一结果。"""

    DENY_ONCE = 0  # 拒绝本次
    ALLOW_ONCE = 1  # 允许本次（不留规则）
    ALLOW_FOREVER = 2  # 永久允许（+写本地层文件，精确匹配）


class ApprovalError(Exception):
    """人在回路被取消或异常。"""
