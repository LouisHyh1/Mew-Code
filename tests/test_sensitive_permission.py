import json

from novacode.llm import ToolCall
from novacode.permission import Decision, Mode
from novacode.permission.engine import new_engine
from novacode.permission.rule import Rule
from novacode.permission.sensitive import (
    detect_sensitive_command,
    detect_sensitive_tool_call,
    is_sensitive_selector,
)
from novacode.permission.settings import extract_file_selectors


def test_sensitive_selector_matches_env_variants():
    for selector in [
        ".env",
        ".env.local",
        "sub/.env.production",
        "*.env*",
        "**/*.env*",
        "src/**/.env*",
    ]:
        assert is_sensitive_selector(selector), selector


def test_sensitive_selector_matches_common_secret_files():
    for selector in ["secret.pem", "private.key", "id_rsa", "sub/id_ed25519", ".npmrc"]:
        assert is_sensitive_selector(selector), selector


def test_extract_file_selectors_for_glob_and_grep():
    glob_call = _glob_call("*.env*", path="config")
    assert extract_file_selectors(glob_call) == ["*.env*", "config", "config/*.env*"]

    grep_call = _grep_call("API_KEY", glob_filter=".env*", path=".")
    assert extract_file_selectors(grep_call) == [".env*"]


def test_read_sensitive_files_are_denied(tmp_path):
    engine = _engine(tmp_path)
    for path in [".env", ".env.local", "sub/.env.production"]:
        decision, reason = engine.check(Mode.DEFAULT, _read_call(path), True)
        assert decision == Decision.DENY
        assert "sensitive" in reason


def test_non_sensitive_read_is_allowed(tmp_path):
    engine = _engine(tmp_path)
    decision, _ = engine.check(Mode.DEFAULT, _read_call("README.md"), True)
    assert decision == Decision.ALLOW


def test_write_and_edit_sensitive_files_are_denied(tmp_path):
    engine = _engine(tmp_path)
    for call in [_write_call(".env"), _edit_call(".env.local")]:
        decision, reason = engine.check(Mode.DEFAULT, call, False)
        assert decision == Decision.DENY
        assert "sensitive" in reason


def test_glob_sensitive_file_enumeration_is_denied(tmp_path):
    engine = _engine(tmp_path)
    for pattern in ["*.env*", "**/*.env*", "src/**/.env*"]:
        decision, reason = engine.check(Mode.DEFAULT, _glob_call(pattern), True)
        assert decision == Decision.DENY
        assert "sensitive" in reason


def test_grep_sensitive_file_range_is_denied(tmp_path):
    engine = _engine(tmp_path)
    decision, reason = engine.check(Mode.DEFAULT, _grep_call("API_KEY", glob_filter=".env*"), True)
    assert decision == Decision.DENY
    assert "sensitive" in reason

    decision, _ = engine.check(Mode.DEFAULT, _grep_call("API_KEY", glob_filter="*.py"), True)
    assert decision == Decision.ALLOW


def test_bash_sensitive_read_bypasses_safe_command_allow(tmp_path):
    engine = _engine(tmp_path)
    for command in ["cat .env", "type .env", "Get-Content .env", "ls *.env*"]:
        decision, reason = engine.check(Mode.DEFAULT, _bash_call(command), False)
        assert decision == Decision.DENY
        assert "sensitive" in reason


def test_sensitive_deny_beats_allow_rules_and_bypass_mode(tmp_path):
    engine = _engine(tmp_path)
    engine.local.allow.append(Rule("Glob", "**/*", True))

    decision, reason = engine.check(Mode.BYPASS, _glob_call("**/*.env*"), True)
    assert decision == Decision.DENY
    assert "sensitive" in reason


def test_detect_sensitive_command_ignores_harmless_safe_command():
    hit, _ = detect_sensitive_command("git status")
    assert not hit

    hit, reason = detect_sensitive_tool_call(_bash_call("cat .env"))
    assert hit
    assert "sensitive" in reason


def _engine(tmp_path):
    return new_engine(str(tmp_path.resolve()))[0]


def _tool_call(name: str, payload: dict) -> ToolCall:
    return ToolCall(id="t1", name=name, input=json.dumps(payload))


def _bash_call(command: str) -> ToolCall:
    return _tool_call("bash", {"command": command})


def _read_call(path: str) -> ToolCall:
    return _tool_call("read_file", {"path": path})


def _write_call(path: str) -> ToolCall:
    return _tool_call("write_file", {"path": path})


def _edit_call(path: str) -> ToolCall:
    return _tool_call("edit_file", {"path": path, "old_string": "a", "new_string": "b"})


def _glob_call(pattern: str, path: str | None = None) -> ToolCall:
    payload = {"pattern": pattern}
    if path is not None:
        payload["path"] = path
    return _tool_call("glob", payload)


def _grep_call(
    pattern: str,
    glob_filter: str | None = None,
    path: str | None = None,
) -> ToolCall:
    payload = {"pattern": pattern}
    if glob_filter is not None:
        payload["glob"] = glob_filter
    if path is not None:
        payload["path"] = path
    return _tool_call("grep", payload)
