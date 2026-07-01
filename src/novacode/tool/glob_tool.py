"""Glob file search tool."""

import asyncio
import json
from pathlib import Path

from novacode.permission.sensitive import is_sensitive_selector
from novacode.tool import Result


class GlobTool:
    read_only = True

    def name(self) -> str:
        return "glob"

    def description(self) -> str:
        return (
            "按 glob 模式查找匹配的文件（不包含目录）。"
            "非递归：`*.py` 只匹配根目录下的 .py 文件。"
            "递归：`**/*.py` 匹配所有子目录中的 .py 文件。"
            "返回排序后的相对路径列表，最多 100 条。"
        )

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob 模式。非递归：`*.py` 只查顶层；"
                        "递归：`**/*.py` 遍历所有子目录。"
                        "也可用 `src/**/*.py` 限定范围。"
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "搜索的根目录，默认为当前工作目录",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, args: str) -> Result:
        try:
            data = json.loads(args or "{}")
        except json.JSONDecodeError as e:
            return Result(content=f"参数 JSON 解析失败: {e}", is_error=True)
        pattern = data.get("pattern")
        if not pattern:
            return Result(content="缺少必填参数: pattern", is_error=True)
        root = Path(data.get("path") or ".")
        if not root.exists():
            return Result(content=f"目录不存在: {root}", is_error=True)
        try:
            matches = []
            count = 0
            for p in root.glob(pattern):
                if count > 0 and count % 100 == 0:
                    await asyncio.sleep(0)
                if p.is_file():
                    try:
                        rel = p.relative_to(root)
                    except ValueError:
                        rel = p
                    # 统一用正斜杠，避免 Windows 反斜杠在 Rich 渲染中被当转义符吃掉
                    matches.append(str(rel).replace("\\", "/"))
                count += 1
        except OSError as e:
            return Result(content=f"glob 搜索失败: {e}", is_error=True)
        matches = [m for m in matches if not is_sensitive_selector(m)]
        if not matches:
            hint = "（提示：非递归模式 `*.py` 只查顶层，递归请用 `**/*.py`）"
            return Result(content=f"在 {root} 下未匹配到 pattern='{pattern}' 的文件。{hint}")
        matches.sort()
        if len(matches) > 100:
            matches = matches[:100]
            matches.append("[truncated: 结果超过100条]")
        return Result(content="\n".join(matches))
