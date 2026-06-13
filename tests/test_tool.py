"""Tests for tool system: registry and individual tools."""

import json
from pathlib import Path

import pytest

from mewcode.tool import Registry, new_default_registry
from mewcode.tool.bash import BashTool
from mewcode.tool.edit_file import EditFileTool
from mewcode.tool.glob_tool import GlobTool
from mewcode.tool.grep_tool import GrepTool
from mewcode.tool.read_file import ReadFileTool
from mewcode.tool.write_file import WriteFileTool


@pytest.mark.asyncio
async def test_registry_definitions():
    """注册中心导出 6 条工具定义且按名可查 (AC1)."""
    reg = new_default_registry()
    defs = reg.definitions()
    assert len(defs) == 6
    names = [d.name for d in defs]
    assert names == ["read_file", "write_file", "edit_file", "bash", "glob", "grep"]
    for name in names:
        t = reg.get(name)
        assert t is not None
        assert t.name() == name
    assert reg.get("nonexistent") is None


@pytest.mark.asyncio
async def test_registry_execute_unknown():
    reg = Registry()
    r = await reg.execute("unknown", "{}")
    assert r.is_error
    assert "未知工具" in r.content


@pytest.mark.asyncio
async def test_read_file_exists():
    tool = ReadFileTool()
    r = await tool.execute('{"path": "pyproject.toml"}')
    assert not r.is_error
    assert "pyproject.toml" not in r.content  # numbered lines, not raw path
    assert r.content.strip() != ""


@pytest.mark.asyncio
async def test_read_file_not_exists():
    tool = ReadFileTool()
    r = await tool.execute('{"path": "/nonexistent/file.txt"}')
    assert r.is_error
    assert "不存在" in r.content


@pytest.mark.asyncio
async def test_read_file_missing_path():
    tool = ReadFileTool()
    r = await tool.execute("{}")
    assert r.is_error
    assert "path" in r.content


@pytest.mark.asyncio
async def test_write_file_create_and_nested(tmp_path: Path):
    tool = WriteFileTool()
    nested = tmp_path / "a" / "b" / "c.txt"
    args = _json_args({"path": str(nested), "content": "hello world"})
    r = await tool.execute(args)
    assert not r.is_error
    assert nested.read_text() == "hello world"


@pytest.mark.asyncio
async def test_write_file_missing_content():
    tool = WriteFileTool()
    r = await tool.execute('{"path": "test.txt"}')
    assert r.is_error
    assert "content" in r.content


def _json_args(d: dict) -> str:
    """Serialize dict to JSON string for tool args."""
    return json.dumps(d)


@pytest.mark.asyncio
async def test_edit_file_unique_match(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    tool = EditFileTool()
    args = _json_args({"path": str(f), "old_string": "hello", "new_string": "hi"})
    r = await tool.execute(args)
    assert not r.is_error
    assert f.read_text() == "hi world"


@pytest.mark.asyncio
async def test_edit_file_zero_matches(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    tool = EditFileTool()
    args = _json_args({"path": str(f), "old_string": "nonexistent", "new_string": "x"})
    r = await tool.execute(args)
    assert r.is_error
    assert "未找到匹配" in r.content


@pytest.mark.asyncio
async def test_edit_file_multiple_matches(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_text("hello hello hello")
    tool = EditFileTool()
    args = _json_args({"path": str(f), "old_string": "hello", "new_string": "hi"})
    r = await tool.execute(args)
    assert r.is_error
    assert "不唯一" in r.content
    assert "3" in r.content


@pytest.mark.asyncio
async def test_bash_echo():
    tool = BashTool()
    r = await tool.execute('{"command": "echo hello"}')
    assert not r.is_error
    assert "hello" in r.content
    assert "exit_code: 0" in r.content


@pytest.mark.asyncio
async def test_bash_timeout():
    """bash 超时命令被 Registry 超时机制终止并返回结构化错误 (AC5)."""
    from mewcode.tool import Registry
    from mewcode.tool.bash import BashTool

    reg = Registry()
    reg.register(BashTool())
    r = await reg.execute("bash", '{"command": "sleep 30"}', timeout=0.5)
    assert r.is_error
    assert "超时" in r.content


@pytest.mark.asyncio
async def test_glob_finds_py_files():
    tool = GlobTool()
    r = await tool.execute('{"pattern": "**/*.py", "path": "src"}')
    assert not r.is_error
    assert len(r.content.splitlines()) > 0


@pytest.mark.asyncio
async def test_glob_no_match():
    tool = GlobTool()
    r = await tool.execute('{"pattern": "*.nonexistent_ext"}')
    assert not r.is_error
    assert "未匹配到" in r.content


@pytest.mark.asyncio
async def test_grep_find_keyword():
    tool = GrepTool()
    r = await tool.execute('{"pattern": "def test_", "path": "tests"}')
    assert not r.is_error
    lines = r.content.splitlines()
    assert len(lines) > 0
    assert any("test_tool" in line or "test_" in line for line in lines)


@pytest.mark.asyncio
async def test_grep_invalid_regex():
    tool = GrepTool()
    r = await tool.execute('{"pattern": "[invalid"}')
    assert r.is_error
    assert "正则非法" in r.content


@pytest.mark.asyncio
async def test_grep_no_match(tmp_path: Path):
    # 在空目录中搜索，确保无命中
    (tmp_path / "empty.txt").write_text("no match here")
    tool = GrepTool()
    args = _json_args({"pattern": "xyznonexistent_9876543210", "path": str(tmp_path)})
    r = await tool.execute(args)
    assert not r.is_error
    assert "无命中" in r.content  # 格式变为 "在 X 下搜索 pattern='Y' 无命中"
