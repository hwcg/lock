"""
增量 (streaming) 解析器：
将原始 token 流增量识别为以下结构化字段：
    - reasoning_content    (<think> ... </think>)
    - content
    - tool_calls           (DSML 块)

设计：
- 状态机三个状态：BEFORE_THINK / IN_THINK / AFTER_THINK
- 一旦遇到 \n\n<｜DSML｜tool_calls> 进入工具块缓存模式，
  完整收齐后解析为 OpenAI tool_calls 增量
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from deepseek_v4.tokenizer.special_tokens import (
    DSML_TOKEN,
    EOS_TOKEN,
    THINK_END,
    THINK_START,
)
from deepseek_v4.training.tool_use.schema import parse_dsml_tool_calls


# Streaming 状态
S_BEFORE_THINK = 0       # 未进入 <think>
S_IN_THINK = 1           # 进入 <think> 内部
S_AFTER_THINK = 2        # </think> 之后（正文 + 可能的 tool_calls）

# tool_calls 整段触发器
TOOL_CALLS_OPEN = f"\n\n<{DSML_TOKEN}tool_calls>"
TOOL_CALLS_CLOSE = f"</{DSML_TOKEN}tool_calls>"


@dataclass
class IncrementalParser:
    """
    增量解析器。每来一段新 token 调用 `feed()`，返回 OpenAI 兼容的 delta dict。

    返回 delta 例：
        {"reasoning_content": "..."}
        {"content": "..."}
        {"tool_calls": [{"index": 0, "id": ..., "type": "function", "function": {"name": ..., "arguments": "..."}}]}
        {}                            # 无可输出
    """
    thinking_mode: str = "chat"      # 决定起点是否处于 IN_THINK
    state: int = S_BEFORE_THINK
    buffer: str = ""                 # 整体已 feed 的文本（清理后）
    # tool_calls 缓存
    in_tool_block: bool = False
    tool_block_buffer: str = ""
    emitted_tool_calls: int = 0
    finished: bool = False
    finish_reason: Optional[str] = None

    def __post_init__(self):
        if self.thinking_mode == "thinking":
            # 渲染端会在 prompt 末尾追加 <think>，模型直接开始 reasoning
            self.state = S_IN_THINK

    # -------- 主入口 --------

    def feed(self, new_text: str) -> List[Dict[str, Any]]:
        """
        喂入新文本（可能多 token 累计），返回一个或多个 delta 字典。
        """
        if not new_text or self.finished:
            return []

        # 去掉 EOS（如果模型产生）
        if EOS_TOKEN in new_text:
            new_text, _ = new_text.split(EOS_TOKEN, 1)
            self.finished = True
            self.finish_reason = "stop"

        deltas: List[Dict[str, Any]] = []
        idx = 0

        while idx < len(new_text):
            if self.in_tool_block:
                # 等待 tool_calls 块结束
                close_idx = new_text.find(TOOL_CALLS_CLOSE, idx)
                if close_idx == -1:
                    # 全部进 buffer
                    self.tool_block_buffer += new_text[idx:]
                    idx = len(new_text)
                else:
                    # 收齐
                    self.tool_block_buffer += new_text[idx:close_idx + len(TOOL_CALLS_CLOSE)]
                    idx = close_idx + len(TOOL_CALLS_CLOSE)
                    deltas.extend(self._flush_tool_block())
                    self.finish_reason = "tool_calls"
                continue

            if self.state == S_BEFORE_THINK:
                # 期待 <think>
                te = new_text.find(THINK_START, idx)
                if te == -1:
                    chunk = new_text[idx:]
                    idx = len(new_text)
                    if chunk:
                        deltas.append({"content": chunk})
                else:
                    pre = new_text[idx:te]
                    if pre:
                        deltas.append({"content": pre})
                    idx = te + len(THINK_START)
                    self.state = S_IN_THINK

            elif self.state == S_IN_THINK:
                # 期待 </think>
                te = new_text.find(THINK_END, idx)
                if te == -1:
                    chunk = new_text[idx:]
                    idx = len(new_text)
                    if chunk:
                        deltas.append({"reasoning_content": chunk})
                else:
                    pre = new_text[idx:te]
                    if pre:
                        deltas.append({"reasoning_content": pre})
                    idx = te + len(THINK_END)
                    self.state = S_AFTER_THINK

            elif self.state == S_AFTER_THINK:
                # 检测 tool_calls 起点
                to = new_text.find(TOOL_CALLS_OPEN, idx)
                if to == -1:
                    chunk = new_text[idx:]
                    idx = len(new_text)
                    if chunk:
                        deltas.append({"content": chunk})
                else:
                    pre = new_text[idx:to]
                    if pre:
                        deltas.append({"content": pre})
                    idx = to + len(TOOL_CALLS_OPEN)
                    self.in_tool_block = True
                    # 把开头标签写进 buffer，便于 parse_dsml_tool_calls
                    self.tool_block_buffer = f"<{DSML_TOKEN}tool_calls>"

        # 合并相同 key 的连续 delta
        return _merge_deltas(deltas)

    def finalize(self) -> List[Dict[str, Any]]:
        """生成结束，flush 残留 buffer。"""
        if not self.in_tool_block:
            return []
        # 未闭合的 tool block：尝试 best-effort 解析
        return self._flush_tool_block(allow_incomplete=True)

    # -------- tool_calls --------

    def _flush_tool_block(self, allow_incomplete: bool = False) -> List[Dict[str, Any]]:
        """把 tool_block_buffer 解析为 OpenAI tool_calls 增量。"""
        text = self.tool_block_buffer
        if allow_incomplete and TOOL_CALLS_CLOSE not in text:
            text = text + TOOL_CALLS_CLOSE  # 强行补全尾标签让正则能匹配
        try:
            calls = parse_dsml_tool_calls(text)
        except Exception:
            calls = []
        self.in_tool_block = False
        self.tool_block_buffer = ""

        deltas: List[Dict[str, Any]] = []
        for i, c in enumerate(calls):
            args = c.get("arguments", {})
            args_str = json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args
            deltas.append({
                "tool_calls": [{
                    "index": self.emitted_tool_calls + i,
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": args_str,
                    },
                }],
            })
        self.emitted_tool_calls += len(calls)
        return deltas


def _merge_deltas(deltas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把连续的同类 delta 合并以减少 SSE 帧数（保留 tool_calls 单帧）。"""
    if not deltas:
        return []
    merged: List[Dict[str, Any]] = []
    for d in deltas:
        if not d:
            continue
        if "tool_calls" in d:
            merged.append(d)
            continue
        # content / reasoning_content：合并最近一帧
        if merged and set(d.keys()) == set(merged[-1].keys()):
            for k in d:
                merged[-1][k] = merged[-1].get(k, "") + d[k]
        else:
            merged.append(dict(d))
    return merged


# ============================================================
# 非流式解析（完整文本一次性 parse）
# ============================================================

def parse_full_completion(
    text: str, thinking_mode: str = "chat",
) -> Tuple[str, str, List[Dict[str, Any]], str]:
    """
    完整解析一段 assistant 输出。

    Returns:
        (content, reasoning_content, tool_calls, finish_reason)
    """
    parser = IncrementalParser(thinking_mode=thinking_mode)
    deltas = parser.feed(text)
    deltas.extend(parser.finalize())

    content = ""
    reasoning = ""
    tool_calls: List[Dict[str, Any]] = []
    for d in deltas:
        if "content" in d:
            content += d["content"]
        if "reasoning_content" in d:
            reasoning += d["reasoning_content"]
        if "tool_calls" in d:
            for tc in d["tool_calls"]:
                tool_calls.append({
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": tc["function"],
                })

    finish_reason = parser.finish_reason or "stop"
    if tool_calls and not parser.in_tool_block:
        finish_reason = "tool_calls"
    return content, reasoning, tool_calls, finish_reason
