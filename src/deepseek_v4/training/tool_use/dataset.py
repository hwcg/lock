"""
Tool Use Dataset：在 SFTDataset 基础上做以下额外处理：

1. 自动把 `tools` 字段注入到 system message
2. assistant 的 tool_call block 也参与 loss
3. tool 结果（user 角色内 <tool_result>）保持 mask
4. 训练前预校验 schema（开发阶段检查数据正确性）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from deepseek_v4.data.dataset import SFTDataset
from deepseek_v4.tokenizer.encoding import encode_messages
from deepseek_v4.training.tool_use.schema import (
    ToolSchema, parse_dsml_tool_calls, validate_tool_call,
)
from deepseek_v4.utils.io import read_jsonl
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


class ToolUseDataset(SFTDataset):
    """
    Tool Use SFT 数据集。

    支持的 jsonl 格式：
        {
          "messages": [
              {"role": "user", "content": "compute 1+1"},
              {"role": "assistant", "content": "", "tool_calls": [...]},
              {"role": "tool", "tool_call_id": "x", "content": "2"},
              {"role": "assistant", "content": "1+1=2"},
          ],
          "tools": [...]    # OpenAI 格式 tool 列表
        }

    构造时会：
    1. 把 tools 注入第一条 system / 自动新建 system
    2. 走 encode_messages 渲染（包括 tool_result 合并到 user）
    3. 渐进式 assistant token 标 mask（与父类一致）
    """

    def __init__(
        self,
        paths: List[Union[str, Path]],
        tokenizer,
        max_seq_len: int = 4096,
        cache_dir: Optional[Union[str, Path]] = None,
        ignore_index: int = -100,
        validate_schemas: bool = True,
        skip_invalid: bool = True,
        thinking_mode_default: str = "chat",
    ):
        # validate_schemas / skip_invalid 在 _build 前处理
        self.validate_schemas = validate_schemas
        self.skip_invalid = skip_invalid
        self._n_validation_errors = 0
        super().__init__(
            paths=paths, tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            cache_dir=cache_dir,
            ignore_index=ignore_index,
            thinking_mode_default=thinking_mode_default,
            mask_user=True,
        )
        if self._n_validation_errors > 0:
            logger.warning(
                f"[ToolUseDataset] 校验失败的 tool_call 共 {self._n_validation_errors} 条"
                + ("，已跳过" if skip_invalid else "")
            )

    def _build(self) -> None:
        for p in self.paths:
            for row in read_jsonl(p):
                msgs = row.get("messages")
                tools = row.get("tools")
                if not msgs:
                    continue
                msgs = self._normalize_messages(msgs)
                if not msgs:
                    continue

                # 把 tools 注入第一条 system
                if tools:
                    msgs = self._inject_tools(msgs, tools)

                # 校验（可选）
                if self.validate_schemas and tools:
                    schemas = {
                        t["function"]["name"] if "function" in t else t["name"]:
                        ToolSchema.from_dict(t)
                        for t in tools
                    }
                    if not self._validate_assistant_calls(msgs, schemas):
                        self._n_validation_errors += 1
                        if self.skip_invalid:
                            continue

                # 走父类同样的渐进式 encode
                input_ids, label_ids = self._encode_with_mask(
                    msgs, thinking_mode=("thinking" if row.get("thinking") else self.thinking_mode_default)
                )
                if not input_ids:
                    continue
                if len(input_ids) > self.max_seq_len:
                    input_ids = input_ids[:self.max_seq_len]
                    label_ids = label_ids[:self.max_seq_len]
                if not any(l != self.ignore_index for l in label_ids):
                    continue
                self.examples.append((input_ids, label_ids))

    @staticmethod
    def _inject_tools(msgs: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把 tools 字段挂到 system；若无 system 则新建。"""
        for m in msgs:
            if m.get("role") == "system":
                m["tools"] = tools
                return msgs
        # 没有 system → 新建
        return [{"role": "system", "content": "", "tools": tools}] + msgs

    @staticmethod
    def _validate_assistant_calls(
        msgs: List[Dict[str, Any]],
        schemas: Dict[str, ToolSchema],
    ) -> bool:
        """对 assistant 中所有 tool_calls 进行 schema 校验，全部通过返回 True。"""
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            tcs = m.get("tool_calls") or []
            for tc in tcs:
                # tc OpenAI 格式：{"function": {"name", "arguments"}}
                fn = tc.get("function", tc)
                name = fn.get("name")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    return False
                if not validate_tool_call({"name": name, "arguments": args}, schemas).ok:
                    return False
        return True
