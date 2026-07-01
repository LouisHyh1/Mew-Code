"""危险命令黑名单——启发式、非完备、不可配置放开（N1）。

参考 mewcode-python permissions/dangerous.py 设计：
1. 安全命令白名单快速放行（is_safe_command）
2. 危险模式正则匹配（detect）
3. 命中即 Deny，不可绕过（含 bypassPermissions）

用内置正则匹配命令串，命中即 Deny，作为最高优先级层。
不受任何规则、模式、配置影响——bypassPermissions 模式也拦得住。
"""

from __future__ import annotations

import re

# ═══════════════════════════════════════════════════════════════════
# 危险命令正则模式（启发式、非完备）
# ═══════════════════════════════════════════════════════════════════

_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # rm -rf /（递归强制删除根目录，含 -rf 和 -fr 变体）
    (re.compile(r"rm\s+-[a-z]*(?:r[a-z]*f|f[a-z]*r)[a-z]*\s+/\s*$"), "递归强制删除根目录"),
    # rm -rf ~（递归强制删除家目录）
    (re.compile(r"rm\s+-[a-z]*(?:r[a-z]*f|f[a-z]*r)[a-z]*\s+~\s*$"), "递归强制删除家目录"),
    # mkfs 格式化
    (re.compile(r"mkfs\."), "格式化磁盘"),
    # dd 写磁盘设备
    (re.compile(r"dd\s+if=.*of=/dev/"), "直接写磁盘设备"),
    # chmod -R 777 /（递归修改根目录权限）
    (re.compile(r"chmod\s+-R\s+777\s+/"), "递归修改根目录权限"),
    # fork bomb
    (re.compile(r":\(\)\{\s*:\|:&\s*\};:"), "fork bomb"),
    # curl 管道执行远程脚本
    (re.compile(r"curl\s+.*\|\s*(ba)?sh"), "管道执行远程脚本"),
    # wget 管道执行远程脚本
    (re.compile(r"wget\s+.*\|\s*(ba)?sh"), "管道执行远程脚本"),
    # 重定向覆盖磁盘设备
    (re.compile(r">\s*/dev/sd"), "覆盖磁盘设备"),
]

# ═══════════════════════════════════════════════════════════════════
# 安全命令白名单（明确无害的只读/查询命令）
# ═══════════════════════════════════════════════════════════════════

_SAFE_COMMANDS = frozenset(
    {
        "ls",
        "dir",
        "pwd",
        "echo",
        "cat",
        "head",
        "tail",
        "wc",
        "find",
        "which",
        "whereis",
        "whoami",
        "hostname",
        "uname",
        "date",
        "cal",
        "uptime",
        "df",
        "du",
        "free",
        "env",
        "printenv",
        "file",
        "stat",
        "readlink",
        "realpath",
        "basename",
        "dirname",
        "sort",
        "uniq",
        "tr",
        "cut",
        "awk",
        "sed",
        "grep",
        "egrep",
        "fgrep",
        "diff",
        "comm",
        "tee",
        "xargs",
        "true",
        "false",
        "test",
        "git status",
        "git log",
        "git diff",
        "git show",
        "git branch",
        "git tag",
        "git remote",
        "git rev-parse",
        "git ls-files",
        "git blame",
        "git stash list",
        "go version",
        "go env",
        "node -v",
        "npm -v",
        "npx",
        "python --version",
        "pip list",
        "cargo --version",
        "rustc --version",
        "java -version",
        "java --version",
        "docker ps",
        "docker images",
        "docker info",
        "docker version",
        "kubectl get",
        "kubectl describe",
        "kubectl logs",
        "kubectl version",
    }
)


def is_safe_command(command: str) -> bool:
    """安全命令白名单：明确无害的只读/查询命令直接放行。

    任何含管道、分号、重定向等 shell 元字符的命令都不是"安全"的。
    """
    trimmed = command.strip()
    if not trimmed:
        return False
    # 含 shell 元字符 → 不安全
    for ch in ("|", ";", "&&", ">", "$(", "`"):
        if ch in trimmed:
            return False
    for safe in _SAFE_COMMANDS:
        if trimmed == safe or trimmed.startswith(safe + " "):
            return True
    return False


def detect(command: str) -> tuple[bool, str]:
    """检测命令是否命中危险模式。返回 (危险?, 原因描述)。"""
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return True, reason
    return False, ""


def hits_blacklist(command: str) -> bool:
    """命令串命中任一黑名单正则即返回 True（兼容旧接口）。"""
    hit, _ = detect(command)
    return hit
