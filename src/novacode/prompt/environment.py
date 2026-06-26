"""环境信息采集与渲染——不进缓存的变化内容。"""

import os
import platform as _platform
import subprocess
from dataclasses import dataclass
from datetime import date


@dataclass
class Environment:
    """运行环境信息——供模型感知当前上下文，不进缓存。

    各字段采集失败时降级为空字符串，不抛异常（N4）。
    """

    working_dir: str = ""
    platform: str = ""
    date: str = ""
    git_status: str = ""
    version: str = ""
    model: str = ""

    def render(self) -> str:
        """渲染为「环境信息」段文本。

        空值项省略不输出；非空项逐行 "Key: Value"。
        """
        lines: list[str] = []
        if self.working_dir:
            lines.append(f"Working Directory: {self.working_dir}")
        if self.platform:
            lines.append(f"Platform: {self.platform}")
        if self.date:
            lines.append(f"Date: {self.date}")
        if self.git_status:
            lines.append(f"Git Status: {self.git_status}")
        if self.version:
            lines.append(f"Version: {self.version}")
        if self.model:
            lines.append(f"Model: {self.model}")
        return "\n".join(lines)


def gather_environment(version: str, model: str) -> Environment:
    """采集当前运行环境信息。

    - working_dir: os.getcwd()，失败留空
    - platform: platform.system().lower()，失败留空
    - date: datetime.date.today().isoformat()，失败留空
    - git_status: git status --porcelain 摘要（2s 超时），
      非 git 目录 / git 不可用 / 超时则留空
    - version / model: 由调用方传入

    不读取任何环境变量（N5）。
    """
    # 工作目录
    try:
        wd = os.getcwd()
    except OSError:
        wd = ""

    # 平台
    try:
        plat = _platform.system().lower()
    except Exception:
        plat = ""

    # 日期
    try:
        today = date.today().isoformat()
    except Exception:
        today = ""

    # git 状态（2s 超时，失败/非 git 目录降级为空）
    git_stat = ""
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if r.returncode == 0 and r.stdout.strip():
            lines = r.stdout.strip().splitlines()
            git_stat = f"{len(lines)} file(s) changed"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return Environment(
        working_dir=wd,
        platform=plat,
        date=today,
        git_status=git_stat,
        version=version,
        model=model,
    )
