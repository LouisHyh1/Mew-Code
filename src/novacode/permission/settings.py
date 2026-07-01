"""配置加载与映射——Settings YAML、friendly_name、categorize、extract_target。"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from novacode.llm import ToolCall
from novacode.permission import Category
from novacode.permission.rule import RuleSet, parse_rule


class SettingsError(Exception):
    """配置文件解析/加载错误（调用方降级，不向上抛致命异常）。"""


# ── 友好名映射 ──────────────────────────────────────────────────

_FRIENDLY_MAP: dict[str, str] = {
    "bash": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "glob": "Glob",
    "grep": "Grep",
}


def friendly_name(internal: str) -> str:
    """内部名 → 友好名；未知原样返回。"""
    return _FRIENDLY_MAP.get(internal, internal)


# ── 工具分类 ────────────────────────────────────────────────────

_WRITE_TOOLS = {"write_file", "edit_file"}


def categorize(internal: str, read_only: bool) -> Category:
    """工具分类判定：read_only 优先 → READ；否则 write/edit → WRITE；其余 → EXEC。

    N7 最严：未知工具（read_only=False）归 EXEC，触发模式 Ask。
    """
    if read_only:
        return Category.READ
    if internal in _WRITE_TOOLS:
        return Category.WRITE
    return Category.EXEC


# ── 参数提取 ────────────────────────────────────────────────────

_FILE_TOOLS = {"read_file", "write_file", "edit_file"}
_SEARCH_TOOLS = {"glob", "grep"}


def extract_target(call: ToolCall) -> tuple[str, bool, bool]:
    """从工具调用提取目标 (target, is_file, ok)。

    target: 提取的字符串（路径/命令/搜索模式）
      - read_file/write_file/edit_file → 文件路径（走沙箱）
      - glob → glob 模式（如 **/*.py），用于规则匹配
      - grep → 正则搜索模式，用于规则匹配
      - bash → 命令串（走黑名单）
    is_file: True=文件类（走沙箱），False=命令执行类（走黑名单）
    ok: False=解析失败或缺必填字段（N7 安全默认）

    注：glob/grep 返回 pattern 而非 path，对齐 mewcode-python 的
    extract_content 设计。沙箱对 pattern 的检查是尽力而为（pattern 解析为
    项目相对路径，始终通过沙箱——参见 spec 中「glob/grep 沙箱盲区」）。
    """
    args = _parse_json(call.input)
    name = call.name

    if name in _FILE_TOOLS:
        if not isinstance(args, dict) or "path" not in args:
            return "", True, False
        path = args.get("path", "")
        if not isinstance(path, str):
            return "", True, False
        return path, True, True

    if name == "glob":
        if not isinstance(args, dict) or "pattern" not in args:
            return "", True, False
        pattern = args.get("pattern", "")
        if not isinstance(pattern, str):
            return "", True, False
        return pattern, True, True

    if name == "grep":
        if not isinstance(args, dict) or "pattern" not in args:
            return "", True, False
        pattern = args.get("pattern", "")
        if not isinstance(pattern, str):
            return "", True, False
        return pattern, True, True

    if name == "bash":
        if not isinstance(args, dict) or "command" not in args:
            return "", False, False  # 缺 command→空，落规则→模式兜底 Ask
        cmd = args.get("command", "")
        if not isinstance(cmd, str):
            return "", False, False
        return cmd, False, True

    # 未知工具
    return "", False, False


def extract_file_selectors(call: ToolCall) -> list[str]:
    """Extract every path/glob selector that may touch files.

    This is separate from extract_target because search tools have more than
    one file-facing argument: a root path plus an optional filename glob.
    """
    args = _parse_json(call.input)
    if not isinstance(args, dict):
        return []

    name = call.name
    selectors: list[str] = []

    if name in _FILE_TOOLS:
        path = args.get("path", "")
        if isinstance(path, str):
            selectors.append(path)
        return _non_empty(selectors)

    if name == "glob":
        path = args.get("path", "")
        pattern = args.get("pattern", "")
        if isinstance(pattern, str):
            selectors.append(pattern)
        if isinstance(path, str) and path and path != ".":
            selectors.append(path)
            if isinstance(pattern, str) and pattern:
                selectors.append(_join_selector(path, pattern))
        return _non_empty(selectors)

    if name == "grep":
        path = args.get("path", "")
        glob_filter = args.get("glob", "")
        if isinstance(glob_filter, str) and glob_filter:
            selectors.append(glob_filter)
            if isinstance(path, str) and path and path != ".":
                selectors.append(_join_selector(path, glob_filter))
        elif isinstance(path, str) and path and path != ".":
            selectors.append(path)
        return _non_empty(selectors)

    return []

def _join_selector(root: str, pattern: str) -> str:
    return root.rstrip("/\\") + "/" + pattern.lstrip("/\\")


def _non_empty(values: list[str]) -> list[str]:
    return [value for value in values if value]



def _parse_json(input_val: str) -> dict | str:
    """尝试将 input 解析为 dict；失败则返回原始字符串。"""
    try:
        parsed = json.loads(input_val)
        if isinstance(parsed, dict):
            return parsed
        return input_val
    except (json.JSONDecodeError, TypeError):
        return input_val


# ── Settings YAML ───────────────────────────────────────────────


@dataclass
class PermissionsBlock:
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)


@dataclass
class Settings:
    """单个 YAML 文件结构。"""

    default_mode: str = ""  # 可选：default/acceptEdits/plan/bypassPermissions
    permissions: PermissionsBlock = field(default_factory=PermissionsBlock)


def load_settings(path: str) -> Settings:
    """加载权限配置文件。文件不存在→空 Settings、不抛。

    YAML 解析失败→抛 SettingsError（调用方降级跳过）。
    """
    p = Path(path)
    if not p.exists():
        return Settings()

    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SettingsError(f"YAML parse error in {path}: {e}") from e

    if raw is None:
        return Settings()

    if not isinstance(raw, dict):
        raise SettingsError(f"Settings must be a mapping: {path}")

    settings = Settings()

    # default_mode
    dm = raw.get("default_mode", "")
    if isinstance(dm, str):
        settings.default_mode = dm

    # permissions block
    perms_raw = raw.get("permissions")
    if isinstance(perms_raw, dict):
        allow = perms_raw.get("allow", [])
        deny = perms_raw.get("deny", [])
        if isinstance(allow, list):
            settings.permissions.allow = [str(x) for x in allow if isinstance(x, str)]
        if isinstance(deny, list):
            settings.permissions.deny = [str(x) for x in deny if isinstance(x, str)]

    return settings


def to_rule_set(s: Settings) -> RuleSet:
    """将 Settings 转为 RuleSet：allow/deny 各条 parse_rule，非法条目跳过。"""
    ruleset = RuleSet()
    for item in s.permissions.allow:
        rule, ok = parse_rule(item)
        if ok:
            rule.allow = True
            ruleset.allow.append(rule)
    for item in s.permissions.deny:
        rule, ok = parse_rule(item)
        if ok:
            rule.allow = False
            ruleset.deny.append(rule)
    return ruleset
