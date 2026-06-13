"""Write file tool."""

import json
from pathlib import Path

from novacode.tool import Result


class WriteFileTool:
    def name(self) -> str:
        return "write_file"

    def description(self) -> str:
        return "将内容写入指定路径的文件（覆盖写）。父目录不存在时自动创建。返回成功或结构化错误。"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要写入的文件路径",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的文件内容",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, args: str) -> Result:
        try:
            data = json.loads(args or "{}")
        except json.JSONDecodeError as e:
            return Result(content=f"参数 JSON 解析失败: {e}", is_error=True)
        path_str = data.get("path")
        if not path_str:
            return Result(content="缺少必填参数: path", is_error=True)
        if "content" not in data:
            return Result(content="缺少必填参数: content", is_error=True)
        content = data["content"]
        p = Path(path_str)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return Result(content=f"已写入 {path_str}（{len(content.encode())} 字节）")
        except OSError as e:
            return Result(content=f"写入文件失败: {e}", is_error=True)
