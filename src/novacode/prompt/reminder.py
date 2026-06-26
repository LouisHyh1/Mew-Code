"""补充消息注入（system-reminder）与规划模式按轮次提醒。"""


def system_reminder(body: str) -> str:
    """用 <system-reminder> 标签包裹补充指令。

    该标签语义让模型理解这是系统补充上下文而非用户提问——
    不针对它直接回复。消息不写入持久历史，不污染缓存。
    """
    return f"<system-reminder>\n{body}\n</system-reminder>"


# ── 规划模式提醒常量 ──────────────────────────────────────────

_PLAN_REMINDER_FULL = """\
Plan mode is active. The user indicated that they do not want you to execute yet -- you MUST NOT make any edits (with the exception of the plan file mentioned below), run any non-readonly tools (including changing configs or making commits), or otherwise make any changes to the system. This supercedes any other instructions you have received.

## Plan File Info:
{plan_file_info}
You should build your plan incrementally by writing to or editing this file. NOTE that this is the only file you are allowed to edit - other than this you are only allowed to take READ-ONLY actions.

## Plan Workflow

### Phase 1: Initial Understanding
Goal: Gain a comprehensive understanding of the user's request by reading through code and asking them questions.

1. Focus on understanding the user's request and the code associated with their request. Actively search for existing functions, utilities, and patterns that can be reused.
2. Use the Agent tool with subagent_type="explore" to explore the codebase. You can launch up to 3 explore agents IN PARALLEL.

### Phase 2: Design
Goal: Design an implementation approach.
Call the Agent tool with subagent_type="plan" to design the implementation based on the user's intent and your exploration results.

### Phase 3: Review
Goal: Review the plan(s) and ensure alignment with the user's intentions.
1. Read the critical files identified by agents to deepen your understanding
2. Ensure that the plans align with the user's original request

### Phase 4: Final Plan
Goal: Write your final plan to the plan file (the only file you can edit).
- Begin with a Context section explaining why this change is being made
- Include only your recommended approach
- Include the paths of critical files to be modified
- Include a verification section describing how to test the changes

### Phase 5: Call ExitPlanMode
At the very end of your turn, call ExitPlanMode to indicate that you are done planning."""

_PLAN_REMINDER_CONCISE = (
    "Plan mode still active (see full instructions earlier in conversation). "
    "Read-only except plan file ({plan_path}). Follow 5-phase workflow."
)


def plan_reminder(full: bool) -> str:
    """规划模式提醒（已含 <system-reminder> 标签包裹）。

    full=True  → 完整版（首轮及每 PLAN_REMINDER_INTERVAL 轮重复）
    full=False → 精简版（其余轮次）
    """
    body = _PLAN_REMINDER_FULL if full else _PLAN_REMINDER_CONCISE
    return system_reminder(body)


# /do 注入的用户消息——指示模型按上文已确认的计划开始执行。
EXECUTE_DIRECTIVE = "请按上面的计划开始执行。"
