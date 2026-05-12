"""
DeepSeek-V4 特殊 Token 的统一定义。

设计原则：
1. 所有特殊 token 的字符串只在本文件出现，其他模块通过常量引用。
2. ID 分配按"通用 → 对话 → 任务 → DSML"分段，便于阅读与扩展。
3. 提供 SpecialTokens 单例供 tokenizer / encoding 共用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================
# 1. 基础对话 Token
# ============================================================

BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
PAD_TOKEN = "<｜▁pad▁｜>"
UNK_TOKEN = "<｜▁unk▁｜>"

USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
SYSTEM_TOKEN = "<｜System｜>"
TOOL_TOKEN = "<｜Tool｜>"
LATEST_REMINDER_TOKEN = "<｜latest_reminder｜>"


# ============================================================
# 2. 思考 / 推理 Token
# ============================================================

THINK_START = "ᵔ"
THINK_END = "ᵕ"
TOOL_CALL_START = "ᵖ"
TOOL_CALL_END = "ᵗ"
TOOL_RESPONSE_START = "ᵘ"
TOOL_RESPONSE_END = "ᵙ"
TOOL_RESULT_START = "<tool_result>"
TOOL_RESULT_END = "</tool_result>"


# ============================================================
# 3. DSML（DeepSeek Markup Language）
# ============================================================

DSML_TOKEN = "｜DSML｜"          # 用于结构化函数调用
DSML_PARAM_TOKEN = f"<{DSML_TOKEN}parameter"
DSML_INVOKE_TOKEN = f"<{DSML_TOKEN}invoke"
DSML_TOOL_CALLS_TOKEN = f"<{DSML_TOKEN}tool_calls"


# ============================================================
# 4. 内部任务 Token（用于细分类任务，如搜索、动作识别）
# ============================================================

TASK_ACTION = "<｜action｜>"
TASK_QUERY = "<｜query｜>"
TASK_AUTHORITY = "<｜authority｜>"
TASK_DOMAIN = "<｜domain｜>"
TASK_TITLE = "<｜title｜>"
TASK_READ_URL = "<｜read_url｜>"

TASK_TOKENS = {
    "action":    TASK_ACTION,
    "query":     TASK_QUERY,
    "authority": TASK_AUTHORITY,
    "domain":    TASK_DOMAIN,
    "title":     TASK_TITLE,
    "read_url":  TASK_READ_URL,
}
VALID_TASKS = set(TASK_TOKENS.keys())


# ============================================================
# 5. 函数调用相关
# ============================================================

FIM_PREFIX = "<｜fim▁begin｜>"
FIM_MIDDLE = "<｜fim▁hole｜>"
FIM_SUFFIX = "<｜fim▁end｜>"

# ============================================================
# 6. ID 表 —— 与官方 V4 对齐（前缀几十个为特殊 token）
# ============================================================

# 排序非常重要：训练 tokenizer 时这些 token 必须按此顺序被 add_tokens，
# 保证从 0 开始的连续 ID 与 model embedding 行号对齐。
ALL_SPECIAL_TOKENS: List[str] = [
    BOS_TOKEN,              # 0
    EOS_TOKEN,              # 1
    PAD_TOKEN,              # 2
    UNK_TOKEN,              # 3
    USER_TOKEN,             # 4
    ASSISTANT_TOKEN,        # 5
    SYSTEM_TOKEN,           # 6
    TOOL_TOKEN,             # 7
    LATEST_REMINDER_TOKEN,  # 8
    THINK_START,            # 9
    THINK_END,              # 10
    TOOL_CALL_START,        # 11
    TOOL_CALL_END,          # 12
    TOOL_RESPONSE_START,    # 13
    TOOL_RESPONSE_END,      # 14
    TOOL_RESULT_START,      # 15
    TOOL_RESULT_END,        # 16
    DSML_TOKEN,             # 17
    TASK_ACTION,            # 18
    TASK_QUERY,             # 19
    TASK_AUTHORITY,         # 20
    TASK_DOMAIN,            # 21
    TASK_TITLE,             # 22
    TASK_READ_URL,          # 23
    FIM_PREFIX,             # 24
    FIM_MIDDLE,             # 25
    FIM_SUFFIX,             # 26
]


@dataclass(frozen=True)
class SpecialTokens:
    """
    特殊 Token 的不可变快照。

    用法：
        st = SpecialTokens.default()
        prompt = st.user + content + st.eos
    """
    bos: str = BOS_TOKEN
    eos: str = EOS_TOKEN
    pad: str = PAD_TOKEN
    unk: str = UNK_TOKEN
    user: str = USER_TOKEN
    assistant: str = ASSISTANT_TOKEN
    system: str = SYSTEM_TOKEN
    tool: str = TOOL_TOKEN
    latest_reminder: str = LATEST_REMINDER_TOKEN

    think_start: str = THINK_START
    think_end: str = THINK_END
    tool_call_start: str = TOOL_CALL_START
    tool_call_end: str = TOOL_CALL_END
    tool_response_start: str = TOOL_RESPONSE_START
    tool_response_end: str = TOOL_RESPONSE_END
    tool_result_start: str = TOOL_RESULT_START
    tool_result_end: str = TOOL_RESULT_END

    dsml: str = DSML_TOKEN
    fim_prefix: str = FIM_PREFIX
    fim_middle: str = FIM_MIDDLE
    fim_suffix: str = FIM_SUFFIX

    task_tokens: Dict[str, str] = field(default_factory=lambda: dict(TASK_TOKENS))

    # ---------- 便捷方法 ----------

    @classmethod
    def default(cls) -> "SpecialTokens":
        return cls()

    @property
    def all_tokens(self) -> List[str]:
        return list(ALL_SPECIAL_TOKENS)

    @property
    def all_ids_map(self) -> Dict[str, int]:
        return {tok: i for i, tok in enumerate(ALL_SPECIAL_TOKENS)}

    def is_special(self, token: str) -> bool:
        return token in self.all_ids_map
