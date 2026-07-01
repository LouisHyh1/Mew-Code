"""路径沙箱——把文件类工具的读写限定在项目根目录内（N2）。

先解析符号链接再做前缀判断，防止用软链接指向外部目录绕过。
对尚不存在的目标解析其最近的已存在祖先目录后再判断。
"""

import os
from pathlib import Path


def resolve_root(root: str) -> str:
    """解析项目根为绝对、已解析符号链接的真实路径。失败抛异常。"""
    return str(Path(root).expanduser().resolve(strict=True))


def eval_symlinks_or_ancestor(abs_path: str) -> str:
    """解析目标真实路径；不存在则回退到最近已存在祖先目录再拼接。

    覆盖「新建文件、含未创建中间目录」的情况，不因目标不存在而误判。
    """
    p = Path(abs_path)
    if p.exists():
        return str(p.resolve(strict=True))
    # 逐级回退找已存在祖先
    current = p
    suffix_parts: list[str] = []
    while current != current.parent:
        if current.exists():
            resolved_ancestor = str(current.resolve(strict=True))
            if suffix_parts:
                return os.path.join(resolved_ancestor, *reversed(suffix_parts))
            return resolved_ancestor
        suffix_parts.append(current.name)
        current = current.parent
    # 退到了根（极端情况：根不存在）
    return abs_path


def sandbox_ok(root: str, path: str) -> bool:
    """判断 path 是否落在 root 之内（先解析符号链接、再前缀比对）。

    空 path 视为 root。相对路径相对于 root 解析为绝对路径。
    """
    if not path:
        return True
    # 相对路径 → 绝对路径（相对 root）
    if not os.path.isabs(path):
        abs_path = os.path.normpath(os.path.join(root, path))
    else:
        abs_path = os.path.normpath(path)
    resolved = eval_symlinks_or_ancestor(abs_path)
    # 前缀判断：用 os.sep 保证无斜杠差异
    root_sep = root.rstrip(os.sep) + os.sep
    resolved_sep = resolved.rstrip(os.sep) + os.sep
    return resolved == root or resolved_sep.startswith(root_sep)
