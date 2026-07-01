"""权限引擎——前四层判定流水线 + 配置加载与合并。

check 流水线：① 黑名单 → ② 沙箱 → ③ 规则引擎 → ④ 模式兜底
任一层给出 Allow/Deny 即短路返回；Ask 作为第五层信号。
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from re import Pattern as RePattern

from novacode.llm import ToolCall
from novacode.permission import Category, Decision, Mode, parse_mode
from novacode.permission.blacklist import _DANGEROUS_PATTERNS, detect, is_safe_command
from novacode.permission.rule import RuleSet
from novacode.permission.sandbox import resolve_root, sandbox_ok
from novacode.permission.sensitive import detect_sensitive_tool_call
from novacode.permission.settings import (
    SettingsError,
    categorize,
    extract_target,
    friendly_name,
    load_settings,
    to_rule_set,
)

logger = logging.getLogger(__name__)


@dataclass
class Engine:
    """权限引擎——持有黑名单、沙箱根、三级规则集、启动模式。"""

    root: str  # 项目根（绝对、已解析符号链接）
    blacklist: list[RePattern]  # 内置危险命令正则
    user: RuleSet  # 用户级
    project: RuleSet  # 项目级
    local: RuleSet  # 本地级
    local_path: str  # 永久放行写入目标（本地层文件）
    _start_mode: Mode = Mode.DEFAULT  # 启动默认模式

    def check(self, mode: Mode, call: ToolCall, read_only: bool) -> tuple[Decision, str]:
        """前四层判定流水线（agent 每次执行工具前调用）。

        read_only 由调用方按批类型给定（等价 registry.is_read_only）。
        返回 (裁决, 原因)；原因文案统一供 Deny 回灌与 Ask 展示。
        """
        cat = categorize(call.name, read_only)
        friendly = friendly_name(call.name)
        target, is_file, ok = extract_target(call)

        hit, reason = detect_sensitive_tool_call(call)
        if hit:
            return Decision.DENY, reason

        # ① 黑名单（仅命令执行类）：安全命令白名单 → Allow；危险模式 → Deny
        if cat == Category.EXEC and target:
            # 1a: 安全命令白名单（明确无害的只读/查询命令直接放行）
            if is_safe_command(target):
                return Decision.ALLOW, ""
            # 1b: 危险命令黑名单（不可绕过，含 bypassPermissions）
            hit, reason = detect(target)
            if hit:
                return Decision.DENY, f"命中危险命令黑名单：{reason}（{_preview(target)}）"

        # ② 沙箱（仅文件类）
        if is_file:
            if not ok:
                return Decision.DENY, "无法解析文件路径参数，安全拒绝"
            if not sandbox_ok(self.root, target):
                return Decision.DENY, f"路径在项目目录之外：{target}"

        # ③ 规则引擎：local → project → user，就近命中即返回
        for layer_name, rule_set in [
            ("local", self.local),
            ("project", self.project),
            ("user", self.user),
        ]:
            d, hit = rule_set.match(friendly, target)
            if hit:
                if d == Decision.DENY:
                    return Decision.DENY, f"匹配 deny 规则：{friendly}({target})"
                return Decision.ALLOW, ""  # allow 规则直接放行（""=无需展示）

        # ④ 模式兜底：只产 Allow 或 Ask
        d = _mode_fallback(mode, cat)
        if d == Decision.ALLOW:
            return Decision.ALLOW, ""
        cat_name = {Category.READ: "只读", Category.WRITE: "文件写", Category.EXEC: "命令执行"}[cat]
        return Decision.ASK, f"{mode} 模式下 {cat_name} 类操作需确认"

    @property
    def start_mode(self) -> Mode:
        return self._start_mode


def new_engine(root: str) -> tuple[Engine, Exception | None]:
    """构造权限引擎：解析项目根、加载三层配置、编译黑名单、确定启动模式。

    致命错（仅 resolve_root 失败）也返回非 None 空规则安全引擎 + err。
    配置文件格式错误绝不致错，只降级该文件为空（N5）。
    """
    err: Exception | None = None

    # 解析项目根
    try:
        resolved_root = resolve_root(root)
    except Exception as e:
        resolved_root = root  # 退化为传入值
        err = e

    # 加载三层配置
    home = str(Path.home())
    user_path = f"{home}/.novacode/settings.yaml"
    project_path = f"{resolved_root}/.novacode/settings.yaml"
    local_path = f"{resolved_root}/.novacode/settings.local.yaml"

    user_settings = _load_or_empty(user_path)
    project_settings = _load_or_empty(project_path)
    local_settings = _load_or_empty(local_path)

    user_rules = to_rule_set(user_settings)
    project_rules = to_rule_set(project_settings)
    local_rules = to_rule_set(local_settings)

    # 启动默认模式：local > project > user
    start_mode = Mode.DEFAULT
    for settings in [local_settings, project_settings, user_settings]:
        if settings.default_mode:
            m, ok = parse_mode(settings.default_mode)
            if ok:
                start_mode = m
                break

    engine = Engine(
        root=resolved_root,
        blacklist=[p for p, _ in _DANGEROUS_PATTERNS],
        user=user_rules,
        project=project_rules,
        local=local_rules,
        local_path=local_path,
        _start_mode=start_mode,
    )
    return engine, err


def _load_or_empty(path: str):
    """加载配置，任何失败返回空 Settings（N5 降级）。"""
    from novacode.permission.settings import Settings

    try:
        return load_settings(path)
    except (SettingsError, Exception):
        return Settings()


def _mode_fallback(mode: Mode, cat: Category) -> Decision:
    """F5 矩阵——规则未命中时的兜底裁决，只产 Allow/Ask。

    | 模式              | 只读  | 文件写 | 命令执行 |
    | default           | Allow | Ask    | Ask      |
    | acceptEdits       | Allow | Allow  | Ask      |
    | plan              | Allow | Ask    | Ask      |
    | bypassPermissions | Allow | Allow  | Allow    |
    """
    if cat == Category.READ:
        return Decision.ALLOW
    if mode == Mode.BYPASS:
        return Decision.ALLOW
    if mode == Mode.ACCEPT_EDITS and cat == Category.WRITE:
        return Decision.ALLOW
    return Decision.ASK


def _preview(s: str, max_len: int = 60) -> str:
    """截断预览。"""
    return s[:max_len] + "…" if len(s) > max_len else s
