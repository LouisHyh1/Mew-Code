"""Edit file tool — unique-match replace."""

import json
from pathlib import Path

from novacode.tool import Result


class EditFileTool:
    read_only = False

    def name(self) -> str:
        return "edit_file"

    def description(self) -> str:
        return (
            "对文件中的唯一匹配文本做精确替换。"
            "old_string 必须在文件中恰好出现一次，否则返回可区分错误。"
        )

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要编辑的文件路径",
                },
                "old_string": {
                    "type": "string",
                    "description": "文件中待替换的原文片段（需唯一匹配）",
                },
                "new_string": {
                    "type": "string",
                    "description": "替换后的新文本片段",
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    async def execute(self, args: str) -> Result:
        try:
            data = json.loads(args or "{}")
        except json.JSONDecodeError as e:
            return Result(content=f"参数 JSON 解析失败: {e}", is_error=True)
        path_str = data.get("path")
        if not path_str:
            return Result(content="缺少必填参数: path", is_error=True)
        if "old_string" not in data:
            return Result(content="缺少必填参数: old_string", is_error=True)
        if "new_string" not in data:
            return Result(content="缺少必填参数: new_string", is_error=True)
        old = data["old_string"]
        new = data["new_string"]
        p = Path(path_str)
        if not p.exists():
            return Result(content=f"文件不存在: {path_str}", is_error=True)
        try:
            content = p.read_text(encoding="utf-8")
        except OSError as e:
            return Result(content=f"读取文件失败: {e}", is_error=True)
        n = content.count(old)
        if n == 0:
            return Result(content="未找到匹配的内容", is_error=True)
        if n > 1:
            return Result(
                content=f"匹配到 {n} 处，old_string 不唯一，请提供更长上下文使其唯一",
                is_error=True,
            )
        replaced = content.replace(old, new, 1)
        try:
            p.write_text(replaced, encoding="utf-8")
        except OSError as e:
            return Result(content=f"写入文件失败: {e}", is_error=True)
        return Result(content=f"已编辑 {path_str}：完成 1 处替换")
