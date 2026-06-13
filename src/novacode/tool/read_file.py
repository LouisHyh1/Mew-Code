"""Read file tool."""

import json
from pathlib import Path

from novacode.tool import Result, _truncate


class ReadFileTool:
    def name(self) -> str:
        return "read_file"

    def description(self) -> str:
        return "读取指定路径的文件内容，返回带行号的文本。文件不存在或不可读时返回结构化错误。"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要读取的文件路径",
                },
            },
            "required": ["path"],
        }

    async def execute(self, args: str) -> Result:
        try:
            data = json.loads(args or "{}")
        except json.JSONDecodeError as e:
            return Result(content=f"参数 JSON 解析失败: {e}", is_error=True)
        path_str = data.get("path")
        if not path_str:
            return Result(content="缺少必填参数: path", is_error=True)
        p = Path(path_str)
        if not p.exists():
            return Result(content=f"文件不存在: {path_str}", is_error=True)
        if p.is_dir():
            return Result(content=f"路径是目录而非文件: {path_str}", is_error=True)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except PermissionError:
            return Result(content=f"无权限读取文件: {path_str}", is_error=True)
        except OSError as e:
            return Result(content=f"读取文件失败: {e}", is_error=True)
        lines = text.splitlines()
        numbered = [f"{i:6d}\t{line}" for i, line in enumerate(lines, 1)]
        result = "\n".join(numbered)
        result = _truncate(result, max_lines=2000, max_chars=256 * 1024)
        return Result(content=result)
