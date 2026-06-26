"""Tests for ch05 prompt package — module assembly, environment, reminder."""

from novacode.prompt import (
    EXECUTE_DIRECTIVE,
    Module,
    assemble_system,
    build_system_prompt,
    fixed_modules,
    gather_environment,
    optional_modules,
    plan_reminder,
    system_reminder,
)
from novacode.prompt.environment import Environment

# ── T1: 模块化装配 ──────────────────────────────────────────


class TestModuleAssembly:
    """AC1/F1 — 模块化装配：按优先级排列、空行分隔、挂载即扩展。"""

    def test_fixed_modules_order(self):
        """七个固定模块按优先级升序排列。"""
        mods = fixed_modules()
        priorities = [m.priority for m in mods]
        assert priorities == sorted(priorities), "Priorities must be ascending"
        # 身份(10) 在 工具使用(50) 之前
        names = [m.name for m in mods]
        assert names.index("身份") < names.index("工具使用")

    def test_build_system_prompt_structure(self):
        """系统提示含关键模块内容，模块间以双换行分隔。"""
        s = build_system_prompt()
        assert "NovaCode" in s
        assert "read_file" in s
        assert "ReAct" in s
        # 模块间以空行分隔（双换行）
        assert "\n\n" in s

    def test_hook_and_play(self):
        """挂载即扩展：额外模块自动按优先级插入，不改 assemble_system 逻辑。"""
        extra = Module(name="test_extra", priority=15, content="EXTRA")
        mods = fixed_modules() + [extra]
        assembled = assemble_system(mods)
        # EXTRA 应在身份(10)之后、系统约束(20)之前
        idx_extra = assembled.index("EXTRA")
        idx_nova = assembled.index("NovaCode")
        idx_constraint = assembled.index("Operate within")
        assert idx_nova < idx_extra < idx_constraint, (
            f"EXTRA should be between identity and constraint: "
            f"nova={idx_nova}, extra={idx_extra}, constraint={idx_constraint}"
        )

    def test_optional_modules_have_empty_content(self):
        """三个可选空槽 content 均为空字符串。"""
        mods = optional_modules()
        assert len(mods) == 3
        for m in mods:
            assert m.content == "", f"{m.name} should have empty content"
        assert [m.priority for m in mods] == [80, 90, 100]

    def test_assemble_skips_empty(self):
        """AC2 — 空 content 模块被跳过，不产生多余空行。"""
        mods = [
            Module(name="a", priority=1, content="Hello"),
            Module(name="empty1", priority=2, content=""),
            Module(name="b", priority=3, content="World"),
            Module(name="empty2", priority=4, content=""),
        ]
        result = assemble_system(mods)
        assert result == "Hello\n\nWorld"
        assert "\n\n\n" not in result

    def test_assemble_stable_sort(self):
        """同 priority 模块保持传入顺序（稳定排序）。"""
        mods = [
            Module(name="first", priority=5, content="AAA"),
            Module(name="second", priority=5, content="BBB"),
        ]
        result = assemble_system(mods)
        assert result == "AAA\n\nBBB"


class TestDeterminism:
    """AC5 / N1 — 缓存确定性：连续构造逐字节相等。"""

    def test_build_system_prompt_deterministic(self):
        """两次 build_system_prompt() 结果完全相等。"""
        s1 = build_system_prompt()
        s2 = build_system_prompt()
        assert s1 == s2
        assert len(s1) > 0

    def test_fixed_modules_deterministic(self):
        """两次 fixed_modules() 返回内容相等。"""
        m1 = fixed_modules()
        m2 = fixed_modules()
        assert len(m1) == len(m2)
        for a, b in zip(m1, m2):
            assert a.name == b.name
            assert a.priority == b.priority
            assert a.content == b.content

    def test_stable_block_no_environment_leak(self):
        """稳定块不含日期/路径/git 等环境相关内容。"""
        s = build_system_prompt()
        import datetime

        today = datetime.date.today().isoformat()
        assert today not in s, "Stable system prompt must not contain today's date"
        assert "Working Directory" not in s
        assert "Platform:" not in s


# ── F5: 双重强化 ───────────────────────────────────────────


class TestDualReinforcement:
    """AC7/F5 — 关键约定在系统提示模块与工具描述中双重出现。"""

    def test_system_prompt_contains_tool_rules(self):
        """系统提示「工具使用」模块含专用工具优先 + 编辑前先读。"""
        s = build_system_prompt()
        assert "read_file" in s
        assert "edit_file" in s
        # 编辑前先读
        assert "MUST" in s and "read" in s
        # 优先用专用工具而非 bash
        assert "Prefer dedicated tools" in s or "bash" in s.lower()

    def test_edit_file_description_reinforcement(self):
        """edit_file 工具描述含「编辑前先读」强化语句。"""
        from novacode.tool.edit_file import EditFileTool

        desc = EditFileTool().description()
        assert "read_file" in desc, f"edit_file description should mention read_file: {desc}"

    def test_bash_description_reinforcement(self):
        """bash 工具描述含「优先用专用工具而非 bash 拼凑」强化语句。"""
        from novacode.tool.bash import BashTool

        desc = BashTool().description()
        assert "read_file" in desc or "glob" in desc, (
            f"bash description should mention dedicated tools: {desc}"
        )


# ── T2: 环境采集与渲染 ─────────────────────────────────────


class TestEnvironment:
    """AC3/F2 — 环境信息采集与渲染。"""

    def test_gather_environment_basic(self):
        """基本采集：含工作目录、平台、日期。"""
        env = gather_environment("1.0.0", "test-model")
        assert env.version == "1.0.0"
        assert env.model == "test-model"
        assert len(env.working_dir) > 0
        assert len(env.platform) > 0
        assert len(env.date) > 0

    def test_render_includes_key_fields(self):
        """render() 输出含关键字段标签。"""
        env = gather_environment("2.0.0", "claude-3")
        r = env.render()
        assert "Working Directory" in r
        assert "Platform:" in r or "Platform" in r
        assert "Date:" in r or "Date" in r
        assert "2.0.0" in r
        assert "claude-3" in r

    def test_git_status_empty_in_non_git_dir(self):
        """非 git 目录下 git_status 应为空字符串。"""
        import subprocess

        # 如果当前目录是 git 仓库，跳过此测试
        try:
            r = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            in_git = r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            in_git = False

        if not in_git:
            env = gather_environment("1.0.0", "test")
            assert env.git_status == "", (
                f"git_status should be empty in non-git dir, got: {env.git_status!r}"
            )

    def test_render_omits_empty_fields(self):
        """空值字段在 render() 中省略。"""
        env = Environment(working_dir="", platform="linux", date="", version="v1", model="m1")
        r = env.render()
        assert "Working Directory" not in r
        assert "Date" not in r
        assert "Platform" in r
        assert "v1" in r

    def test_all_fields_empty_render(self):
        """全空字段 render 返回空字符串。"""
        env = Environment()
        assert env.render() == ""


# ── T3: 补充消息与规划提醒 ─────────────────────────────────


class TestReminder:
    """AC8/F6 — system-reminder 标签包裹 + 规划按轮次详略。"""

    def test_system_reminder_wraps_with_tags(self):
        """system_reminder 以 <system-reminder> 标签包裹。"""
        result = system_reminder("hello world")
        assert result.startswith("<system-reminder>")
        assert result.endswith("</system-reminder>")
        assert "hello world" in result

    def test_plan_reminder_full(self):
        """完整版含 PLAN MODE 描述与标签。"""
        r = plan_reminder(True)
        assert "<system-reminder>" in r
        assert "PLAN MODE" in r
        assert "read_file" in r or "read-only" in r.lower()
        assert "step-by-step" in r or "/do" in r

    def test_plan_reminder_concise(self):
        """精简版更短，含标签。"""
        r = plan_reminder(False)
        assert "<system-reminder>" in r
        assert "PLAN MODE" in r
        assert len(r) < len(plan_reminder(True))

    def test_execute_directive(self):
        """EXECUTE_DIRECTIVE 为非空中文字符串。"""
        assert len(EXECUTE_DIRECTIVE) > 0
        assert isinstance(EXECUTE_DIRECTIVE, str)


# ── F8: 跨协议一致 ──────────────────────────────────────────


class TestCrossProtocol:
    """AC10/F8 — 同一套模块化系统提示应可在两个协议中使用。"""

    def test_system_prompt_same_for_both_protocols(self):
        """build_system_prompt 输出对两协议相同（内容一致）。"""
        s = build_system_prompt()
        # 只是验证内容非空且结构正确——实际跨协议验证在集成测试
        assert len(s) > 500
        assert "NovaCode" in s

    def test_reminder_same_for_both_protocols(self):
        """plan_reminder 输出对两协议一致。"""
        r = plan_reminder(True)
        assert "<system-reminder>" in r
        assert "PLAN MODE" in r
