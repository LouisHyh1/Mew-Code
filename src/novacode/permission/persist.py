"""规则持久化——人在回路「永久」写本地层文件。"""

import logging
from pathlib import Path

import yaml

from novacode.llm import ToolCall
from novacode.permission.rule import Rule, parse_rule
from novacode.permission.settings import extract_target, friendly_name, load_settings

logger = logging.getLogger(__name__)


def rule_for(call: ToolCall, root: str) -> tuple[Rule, str, bool]:
    """据工具调用生成精确规则（无通配）。

    返回 (Rule, YAML串, ok)。ok=False 表示无法生成（解析失败/未知工具）。
    bash 命令串中的 glob 元字符被转义，防止规则被泛化。
    """
    target, is_file, ok = extract_target(call)
    if not ok or not target:
        return Rule("", "", False), "", False

    fn = friendly_name(call.name)
    if call.name == "bash":
        # 转义 glob 元字符
        escaped = _escape_glob(target)
        rule_str = f"{fn}({escaped})"
        rule, _ = parse_rule(rule_str)
        if rule.tool:
            rule.allow = True
            return rule, rule_str, True
        return Rule("", "", False), "", False

    # 文件类：用项目相对路径
    if root and target:
        try:
            rel = str(Path(target).resolve().relative_to(Path(root).resolve()))
        except (ValueError, OSError):
            rel = target
        # 用 slash 分隔
        rel = rel.replace("\\", "/")
        rule_str = f"{fn}({rel})"
        rule, _ = parse_rule(rule_str)
        if rule.tool:
            rule.allow = True
            return rule, rule_str, True

    return Rule("", "", False), "", False


def _escape_glob(s: str) -> str:
    """转义 glob 元字符 *, ?, [, ] 为字面匹配。"""
    result = []
    for ch in s:
        if ch in "*?[]":
            result.append(f"[{ch}]")
        else:
            result.append(ch)
    return "".join(result)


def persist_local_allow(engine, call: ToolCall) -> None:
    """人在回路「永久」：把精确 allow 规则写入本地层配置文件 + 同步内存。

    异常向上抛，调用方（agent）捕获后只记日志不阻断。
    """
    _, rule_str, ok = rule_for(call, engine.root)
    if not ok:
        logger.warning("无法为工具调用生成持久化规则: %s", call.name)
        return

    # 加载现有配置（含缺失处理）
    try:
        settings = load_settings(engine.local_path)
    except Exception:
        from novacode.permission.settings import Settings

        settings = Settings()

    # 去重
    if rule_str not in settings.permissions.allow:
        settings.permissions.allow.append(rule_str)

    # 确保目录存在
    local_file = Path(engine.local_path)
    local_file.parent.mkdir(parents=True, exist_ok=True)

    # 写 YAML
    data = {}
    if settings.default_mode:
        data["default_mode"] = settings.default_mode
    data["permissions"] = {
        "allow": settings.permissions.allow,
        "deny": settings.permissions.deny,
    }
    local_file.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    # 同步内存
    rule, _ = parse_rule(rule_str)
    if rule.tool:
        rule.allow = True
        engine.local.allow.append(rule)
