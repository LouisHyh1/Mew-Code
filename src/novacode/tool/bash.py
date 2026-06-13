"""Bash command execution tool."""

import asyncio
import json
import locale

from novacode.tool import Result, _truncate


class BashTool:
    def name(self) -> str:
        return "bash"

    def description(self) -> str:
        return (
            "在当前工作目录下执行 shell 命令，返回 stdout、stderr 和退出码。"
            "命令受超时约束（30s）。"
            "Windows 下使用 cmd /C，Linux/Mac 下使用 /bin/sh -c。"
        )

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 shell 命令",
                },
            },
            "required": ["command"],
        }

    async def execute(self, args: str) -> Result:
        try:
            data = json.loads(args or "{}")
        except json.JSONDecodeError as e:
            return Result(content=f"参数 JSON 解析失败: {e}", is_error=True)
        cmd = data.get("command")
        if not cmd:
            return Result(content="缺少必填参数: command", is_error=True)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await proc.communicate()
        except OSError as e:
            return Result(content=f"命令执行失败: {e}", is_error=True)

        stdout = _try_decode(stdout_b)
        stderr = _try_decode(stderr_b)

        output = f"exit_code: {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        output = _truncate(output, max_lines=10000, max_chars=30000)
        return Result(content=output)


def _try_decode(data: bytes) -> str:
    """多编码尝试解码，优先系统编码（解决 Windows 中文版 GBK 乱码）。"""
    encodings = [locale.getpreferredencoding(False), "utf-8", "gbk", "cp936"]
    for enc in encodings:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")
