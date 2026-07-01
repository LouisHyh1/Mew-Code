"""Built-in sensitive file access detection.

This layer is intentionally not configurable. It blocks common secret file
selectors before user allow rules or bypass mode can take effect.
"""

from __future__ import annotations

import json
import re
from fnmatch import fnmatch

from novacode.llm import ToolCall
from novacode.permission.rule import match_pattern
from novacode.permission.settings import extract_file_selectors

_SENSITIVE_GLOBS = (
    ".env",
    ".env.*",
    "*.env",
    "*.env.*",
    "**/.env",
    "**/.env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    ".npmrc",
    ".pypirc",
)

_FILE_READ_COMMAND_RE = re.compile(
    r"(?i)(^|[;&|]\s*|\b)"
    r"(cat|type|get-content|gc|grep|rg|findstr|select-string|ls|dir|find|"
    r"head|tail|less|more|sed|awk)\b"
)

_TOKEN_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'|([^\s;&|]+)')


def detect_sensitive_tool_call(call: ToolCall) -> tuple[bool, str]:
    """Return True when a tool call tries to touch built-in sensitive files."""
    if call.name == "bash":
        command = _bash_command(call.input)
        return detect_sensitive_command(command)

    for selector in extract_file_selectors(call):
        if is_sensitive_selector(selector):
            return True, f"built-in sensitive file deny: {selector}"
    return False, ""


def detect_sensitive_command(command: str) -> tuple[bool, str]:
    """Detect shell commands that read, search, or list sensitive file selectors."""
    if not command or not _FILE_READ_COMMAND_RE.search(command):
        return False, ""

    for token in _command_tokens(command):
        if is_sensitive_selector(token):
            return True, f"built-in sensitive file deny: {token}"

    if ".env" in command.lower():
        return True, "built-in sensitive file deny: .env*"
    return False, ""


def is_sensitive_selector(selector: str) -> bool:
    """Return True for paths or glob selectors that target common secret files."""
    cleaned = _clean_selector(selector)
    if not cleaned:
        return False

    lowered = cleaned.lower()
    if ".env" in lowered:
        return True

    basename = lowered.rsplit("/", 1)[-1]
    candidates = {lowered, basename}
    for pattern in _SENSITIVE_GLOBS:
        pat = pattern.lower()
        for candidate in candidates:
            if fnmatch(candidate, pat) or match_pattern(pat, candidate):
                return True
    return False


def _bash_command(raw_args: str) -> str:
    try:
        parsed = json.loads(raw_args or "{}")
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    command = parsed.get("command", "")
    return command if isinstance(command, str) else ""


def _command_tokens(command: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(command):
        token = next((part for part in match.groups() if part is not None), "")
        if token:
            tokens.append(token)
    return tokens


def _clean_selector(selector: str) -> str:
    cleaned = selector.strip().strip("\"'")
    cleaned = cleaned.replace("\\", "/")
    cleaned = cleaned.rstrip(",:)")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned
