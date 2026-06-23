"""跳过备用屏（alternate screen）的 driver，使终端原生文本选择+右键复制可用。"""

import os
import sys

if sys.platform == "win32":
    from textual.drivers.windows_driver import WindowsDriver as _BaseDriver
else:
    from textual.drivers.linux_driver import LinuxDriver as _BaseDriver


class NoAltScreenDriver(_BaseDriver):
    """去掉 alt screen 切换码，让终端保持在普通模式。

    在普通模式下，用户可以用 Shift+拖拽 进行终端原生文本选择，
    然后通过右键（Windows Terminal）或 Ctrl+Shift+C 复制选中文字。
    """

    def start_application_mode(self):
        try:
            rows = os.get_terminal_size().lines
        except OSError:
            rows = 24
        sys.stdout.write("\n" * rows)
        sys.stdout.flush()
        super().start_application_mode()

    def write(self, data: str) -> None:
        if "\x1b[?1049h" in data:
            data = data.replace("\x1b[?1049h", "")
        if "\x1b[?1049l" in data:
            data = data.replace("\x1b[?1049l", "")
        if data:
            super().write(data)
