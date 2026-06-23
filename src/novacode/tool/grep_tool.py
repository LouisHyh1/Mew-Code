"""Grep content search tool."""

import asyncio
import json
import re
from pathlib import Path

from novacode.tool import Result


class GrepTool:
    read_only = True

    def name(self) -> str:
        return "grep"

    def description(self) -> str:
        return (
            "在文件内容中按正则表达式搜索，返回匹配位置（文件名:行号:内容）。最多返回 100 条命中。"
        )

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python 正则表达式搜索模式",
                },
                "path": {
                    "type": "string",
                    "description": "搜索的根目录，默认为当前工作目录",
                },
                "glob": {
                    "type": "string",
                    "description": "文件名过滤 glob 模式，如 *.py",
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
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return Result(content=f"正则非法: {e}", is_error=True)
        root = Path(data.get("path") or ".")
        glob_filter = data.get("glob")
        hits: list[str] = []
        file_count = 0
        try:
            if glob_filter:
                iterator = root.rglob(glob_filter)
            else:
                iterator = root.rglob("*")
            for filepath in iterator:
                if not filepath.is_file():
                    continue
                try:
                    with open(filepath, encoding="utf-8", errors="replace") as f:
                        for lineno, line in enumerate(f, 1):
                            if len(hits) >= 100:
                                break
                            if len(line) > 1024 * 1024:
                                hits.append(f"{filepath}:{lineno}:[该行过长（>1MB），未完整搜索]")
                                continue
                            if rx.search(line):
                                try:
                                    rel = filepath.relative_to(root)
                                except ValueError:
                                    rel = filepath
                                # 统一用正斜杠，避免反斜杠在渲染中被当转义符吃掉
                                hits.append(
                                    f"{str(rel).replace(chr(92), '/')}:{lineno}:{line.rstrip()}"
                                )
                except (OSError, UnicodeDecodeError):
                    continue
                if len(hits) >= 100:
                    break
                file_count += 1
                if file_count % 20 == 0:
                    await asyncio.sleep(0)
        except OSError as e:
            return Result(content=f"grep 搜索失败: {e}", is_error=True)
        if not hits:
            return Result(content=f"在 {root} 下搜索 pattern='{pattern}' 无命中")
        if len(hits) > 100:
            hits = hits[:100]
            hits.append("[truncated: 命中超过100条]")
        return Result(content="\n".join(hits))
