"""规则与匹配——Rule/RuleSet、parse_rule、match_pattern（glob）。

规则以「工具名(模式)」声明，结果只有 allow 或 deny 两种。
工具名用友好名（Bash/Read/Write/Edit/Glob/Grep）。
模式段支持精确匹配与 glob 匹配（* 任意串、** 跨目录段仅对文件路径有意义）。
"""

import re
from dataclasses import dataclass, field
from fnmatch import translate as fnmatch_translate

from novacode.permission import Decision


@dataclass
class Rule:
    """单条规则：工具友好名 + 模式段 + allow/deny。"""

    tool: str  # 友好名：Bash/Read/Write/Edit/Glob/Grep
    pattern: str  # 模式段；"" 表示匹配该工具全部调用
    allow: bool  # True=allow, False=deny


@dataclass
class RuleSet:
    """一组 allow 与 deny 规则。"""

    allow: list[Rule] = field(default_factory=list)
    deny: list[Rule] = field(default_factory=list)

    def match(self, friendly: str, target: str) -> tuple[Decision, bool]:
        """先 deny 再 allow；返回 (Allow|Deny, 命中?)。"""
        for r in self.deny:
            if r.tool == friendly and match_pattern(r.pattern, target):
                return Decision.DENY, True
        for r in self.allow:
            if r.tool == friendly and match_pattern(r.pattern, target):
                return Decision.ALLOW, True
        return Decision.ALLOW, False


def parse_rule(s: str) -> tuple[Rule, bool]:
    """解析 'Tool(pattern)' 或 'Tool' 为 Rule。

    返回 (rule, ok)。非法格式（空、括号不配对）返回 (Rule("","",False), False)。
    注：allow/deny 归属由调用方根据来源列表决定。
    """
    s = s.strip()
    if not s:
        return Rule("", "", False), False
    # 提取工具名与可选模式
    paren = s.find("(")
    if paren == -1:
        # 无括号：匹配该工具全部
        tool = s.strip()
        if not tool:
            return Rule("", "", False), False
        return Rule(tool=tool, pattern="", allow=True), True
    # 有括号
    if not s.endswith(")"):
        return Rule("", "", False), False
    tool = s[:paren].strip()
    pattern = s[paren + 1 : -1]  # 去掉首尾括号
    if not tool:
        return Rule("", "", False), False
    return Rule(tool=tool, pattern=pattern, allow=True), True


def match_pattern(pattern: str, target: str) -> bool:
    """glob 匹配：pattern=="" 恒匹配。

    命令串走「命令 glob」——* 匹配任意字符含空格，其余字面，** 等价 *。
    文件路径按 / 分段：* 匹配段内任意字符，** 跨段匹配。
    """
    if pattern == "":
        return True
    # 判断是否是文件路径（含 / 且 target 含路径）
    if "/" in target:
        # 文件路径 glob：* 段内、** 跨段
        return _match_path_glob(pattern, target)
    # 命令 glob：* 匹配任意字符（含空格），** 等价 *
    regex = _command_glob_to_regex(pattern)
    return bool(re.fullmatch(regex, target))


def _command_glob_to_regex(pattern: str) -> str:
    """将命令 glob 转换为正则：*→.*，其余字面转义。"""
    result = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            # ** 等价于 *
            while i < len(pattern) and pattern[i] == "*":
                i += 1
            result.append(".*")
        else:
            result.append(re.escape(c))
            i += 1
    return "".join(result)


def _match_path_glob(pattern: str, target: str) -> bool:
    """按 / 分段匹配文件路径 glob：* 段内任意、** 跨段。"""
    pat_parts = pattern.split("/")
    tgt_parts = target.split("/")
    return _match_segments(pat_parts, tgt_parts)


def _match_segments(pat_parts: list[str], tgt_parts: list[str]) -> bool:
    """递归/DP 段匹配：** 跨段，* 段内任意字符序列。"""
    # 转 fnmatch pattern 后逐段匹配
    # 用简单 DP
    pn, tn = len(pat_parts), len(tgt_parts)
    # dp[i][j] = pat_parts[:i] 匹配 tgt_parts[:j]
    dp = [[False] * (tn + 1) for _ in range(pn + 1)]
    dp[0][0] = True

    # ** 可以匹配零段
    for i in range(1, pn + 1):
        if pat_parts[i - 1] == "**":
            dp[i][0] = dp[i - 1][0]

    for i in range(1, pn + 1):
        pp = pat_parts[i - 1]
        for j in range(1, tn + 1):
            tp = tgt_parts[j - 1]
            if pp == "**":
                # ** 可以匹配零段（dp[i-1][j]）或多段（dp[i][j-1]）
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
            elif pp == "*":
                # * 匹配任意单段
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 精确段：fnmatch 段
                dp[i][j] = dp[i - 1][j - 1] and _fnmatch_single(pp, tp)

    return dp[pn][tn]


def _fnmatch_single(pat: str, name: str) -> bool:
    """单段 fnmatch。"""
    return bool(re.fullmatch(fnmatch_translate(pat), name))
