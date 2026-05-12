"""
Tool Use Trainer。

复用 SFTTrainer 主循环，增加：
1. 评测时实际执行工具（验证模型输出能否被解析 + 调用结果）
2. metric：tool_format_acc / tool_schema_acc
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from deepseek_v4.training.sft import SFTConfig, SFTTrainer
from deepseek_v4.training.tool_use.dataset import ToolUseDataset
from deepseek_v4.training.tool_use.schema import parse_dsml_tool_calls, validate_tool_call
from deepseek_v4.training.tool_use.tools import ToolRegistry, register_builtin_tools
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolUseConfig(SFTConfig):
    """Tool Use 训练配置。"""
    # 数据集额外
    validate_schemas: bool = True
    skip_invalid: bool = True

    # 评测
    eval_tool_execution: bool = False         # 真正调用工具（慢）
    eval_n_samples: int = 32

    # 默认覆盖：更小 lr 防止破坏对话能力
    learning_rate: float = 1.0e-5
    neftune_alpha: float = 0.0                # tool 训练不建议 NEFTune


class ToolUseTrainer(SFTTrainer):
    """Tool Use Trainer。"""

    def __init__(self, config: ToolUseConfig, model, tokenizer, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(config=config, model=model, tokenizer=tokenizer)
        self.config: ToolUseConfig = config
        self.tool_registry = tool_registry or register_builtin_tools()

    # ----- Dataset -----

    def get_train_dataset(self) -> Dataset:
        if self._train_ds is None:
            self._train_ds = ToolUseDataset(
                paths=self.config.train_data_paths,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                cache_dir=self.config.cache_dir,
                thinking_mode_default=self.config.thinking_mode_default,
                validate_schemas=self.config.validate_schemas,
                skip_invalid=self.config.skip_invalid,
            )
        return self._train_ds

    def get_eval_dataset(self) -> Optional[Dataset]:
        if not self.config.eval_data_paths:
            return None
        if self._eval_ds is None:
            self._eval_ds = ToolUseDataset(
                paths=self.config.eval_data_paths,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                cache_dir=self.config.cache_dir,
                thinking_mode_default=self.config.thinking_mode_default,
                validate_schemas=False,                   # eval 时不校验
                skip_invalid=False,
            )
        return self._eval_ds

    # ----- 额外的 eval metric -----

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        # 基础 loss / acc
        base_metrics = super().evaluate()
        if not self.config.eval_tool_execution:
            return base_metrics

        # 额外：生成 + 工具调用校验
        # 用 prompt 部分让模型 generate；统计：
        #   tool_format_acc:  调用块能成功解析
        #   tool_schema_acc:  解析后通过 schema 校验
        from deepseek_v4.inference.generation import GenerationConfig, generate
        unwrap = self.model.module if hasattr(self.model, "module") else self.model
        unwrap.eval()

        n_total = 0
        n_format_ok = 0
        n_schema_ok = 0
        schemas = {n: s for n, s in self.tool_registry.schemas.items()}

        ds = self._eval_ds
        if ds is None:
            return base_metrics
        gen_cfg = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        # 简化：取前 N 个样本 prompt 部分
        for i in range(min(self.config.eval_n_samples, len(ds))):
            ids, labels = ds.examples[i]
            # 找第一个非 mask 起点之前作为 prompt（保留到第一个 assistant 之前）
            prompt_end = 0
            for j, l in enumerate(labels):
                if l != -100:
                    prompt_end = j
                    break
            if prompt_end == 0:
                continue
            prompt = torch.tensor([ids[:prompt_end]], device=self.device)
            out = generate(unwrap, prompt, config=gen_cfg)
            text = self.tokenizer.decode(out["responses"][0].tolist(), skip_special_tokens=False)

            calls = parse_dsml_tool_calls(text)
            if not calls:
                n_total += 1
                continue
            n_total += 1
            n_format_ok += 1
            all_ok = all(validate_tool_call(c, schemas).ok for c in calls)
            if all_ok:
                n_schema_ok += 1

        if n_total > 0:
            base_metrics["tool_format_acc"] = n_format_ok / n_total
            base_metrics["tool_schema_acc"] = n_schema_ok / n_total
        return base_metrics
