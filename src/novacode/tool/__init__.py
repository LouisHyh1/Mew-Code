"""Tool abstraction: Protocol, Result, Registry."""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from novacode.llm import ToolDefinition

DEFAULT_TIMEOUT: float = 30.0


@dataclass
class Result:
    """工具执行结果——永远以值类型返回，从不抛 Python 异常给上层。"""
    content: str
    is_error: bool = False


@runtime_checkable
class Tool(Protocol):
    """统一工具抽象。"""
    def name(self) -> str: ...
    def description(self) -> str: ...
    def parameters(self) -> dict[str, Any]: ...
    async def execute(self, args: str) -> Result: ...


def _truncate(s: str, max_lines: int, max_chars: int) -> str:
    """截断文本到 max_lines 行 / max_chars 字符，超出尾部追加 [truncated] 标注。"""
    lines = s.splitlines()
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars]
        truncated = True
    if truncated:
        result += "\n[truncated]"
    return result


def new_default_registry() -> "Registry":
    """构造并注册 6 个核心工具。"""
    from novacode.tool.bash import BashTool
    from novacode.tool.edit_file import EditFileTool
    from novacode.tool.glob_tool import GlobTool
    from novacode.tool.grep_tool import GrepTool
    from novacode.tool.read_file import ReadFileTool
    from novacode.tool.write_file import WriteFileTool

    registry = Registry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(BashTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    return registry


class Registry:
    """集中登记、按名查找、导出定义、按名执行。"""

    def __init__(self) -> None:
        self._order: list[str] = []
        self._tools: dict[str, Tool] = {}

    def register(self, t: Tool) -> None:
        name = t.name()
        if name in self._tools:
            raise ValueError(f"工具 '{name}' 已注册")
        self._order.append(name)
        self._tools[name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        result: list[ToolDefinition] = []
        for name in self._order:
            t = self._tools[name]
            result.append(ToolDefinition(
                name=t.name(),
                description=t.description(),
                input_schema=t.parameters(),
            ))
        return result

    async def execute(
        self, name: str, args: str, timeout: float = DEFAULT_TIMEOUT
    ) -> Result:
        tool = self.get(name)
        if tool is None:
            return Result(content=f"未知工具: {name}", is_error=True)
        try:
            return await asyncio.wait_for(tool.execute(args), timeout=timeout)
        except TimeoutError:
            return Result(content=f"工具 {name} 执行超时（{timeout}s）", is_error=True)
        except Exception as e:
            return Result(content=f"工具 {name} 异常: {e}", is_error=True)
