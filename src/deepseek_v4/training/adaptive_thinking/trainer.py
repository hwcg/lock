"""
Adaptive Thinking Trainer。

训练目标：
- 让模型在「难题」上走 <think>...</think>，「简单题」上直接回答
- 用混合 SFT 数据：同一 prompt 提供两个 reference（thinking 版 + chat 版）
- 训练时根据 ModeRouter 决定该样本用哪个 reference

奖励维度（用于可选的二次 RL）：
- correctness：答案正确
- efficiency：simple 问题不应过度思考
- mode_match：路由器选择的模式与数据 reference 一致
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from deepseek_v4.data.collator import SFTCollator
from deepseek_v4.data.dataset import _cache_key, _load_cache, _save_cache
from deepseek_v4.modeling.model import DeepseekV4ForCausalLM
from deepseek_v4.tokenizer.encoding import encode_messages
from deepseek_v4.training.adaptive_thinking.router import ModeRouter
from deepseek_v4.training.base_trainer import BaseTrainer, TrainerConfig
from deepseek_v4.training.sft import SFTConfig, SFTTrainer
from deepseek_v4.utils.io import read_jsonl
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Dataset
# ============================================================

class AdaptiveThinkingDataset(Dataset):
    """
    数据格式（jsonl 行）：
    {
      "messages_user": [...],                # 用户输入 + system
      "assistant_chat":     "直接回答",
      "assistant_thinking": "<think>...真正回答",   # 含 think 标记的版本
    }

    或紧凑格式：
    {
      "prompt": "...",
      "answer_chat": "...",
      "answer_thinking": "..."
    }
    """

    def __init__(
        self,
        paths: List[str],
        tokenizer,
        max_seq_len: int = 4096,
        router: Optional[ModeRouter] = None,
        cache_dir: Optional[str] = None,
        ignore_index: int = -100,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.router = router or ModeRouter(strategy="auto")
        self.ignore_index = ignore_index
        self.examples: List[Dict[str, Any]] = []

        cache_path = None
        if cache_dir:
            key = _cache_key(paths, f"adaptive_{max_seq_len}")
            cache_path = f"{cache_dir}/adaptive_{key}.pkl"
            try:
                self.examples = _load_cache(cache_path)
                logger.info(f"[Adaptive] loaded cache: {len(self.examples)} examples")
                return
            except Exception:
                pass

        for p in paths:
            for row in read_jsonl(p):
                ex = self._build_one(row)
                if ex is not None:
                    self.examples.append(ex)

        if cache_path:
            _save_cache(cache_path, self.examples)
        logger.info(f"[Adaptive] {len(self.examples)} examples")

    def _build_one(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 取 prompt 文本
        if "messages_user" in row:
            msgs = row["messages_user"]
            prompt_text = " ".join(
                m.get("content", "") for m in msgs if m.get("role") in ("user", "developer")
            )
        elif "prompt" in row:
            prompt_text = str(row["prompt"])
            msgs = [{"role": "user", "content": prompt_text}]
        else:
            return None

        ans_chat = row.get("assistant_chat") or row.get("answer_chat")
        ans_think = row.get("assistant_thinking") or row.get("answer_thinking")

        if not ans_chat and not ans_think:
            return None

        # 路由
        mode = self.router.route([prompt_text])[0]
        if mode == "thinking" and ans_think:
            assistant = ans_think
            thinking_mode = "thinking"
        elif mode == "chat" and ans_chat:
            assistant = ans_chat
            thinking_mode = "chat"
        else:
            # 没匹配的版本：用任一可用版本
            if ans_think:
                assistant = ans_think; thinking_mode = "thinking"
            else:
                assistant = ans_chat; thinking_mode = "chat"

        # 处理 thinking 模式：把 <think>X</think>Y 拆成 reasoning_content + content
        reasoning_content = ""
        content = assistant
        if thinking_mode == "thinking":
            import re
            m = re.match(r"^<(?:think|thinking)>(.*?)</(?:think|thinking)>(.*)$", assistant, re.DOTALL)
            if m:
                reasoning_content = m.group(1).strip()
                content = m.group(2).strip()
            else:
                reasoning_content = ""
                content = assistant

        full_msgs = list(msgs) + [{
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,
        }]

        return {
            "messages": full_msgs,
            "thinking_mode": thinking_mode,
            "prompt_text": prompt_text,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        msgs = ex["messages"]
        thinking_mode = ex["thinking_mode"]
        # 渐进式 encode + loss mask（与 SFTDataset 相同思路）
        input_ids: List[int] = []
        label_ids: List[int] = []
        prev_len = 0
        for i in range(len(msgs)):
            partial = msgs[:i + 1]
            text = encode_messages(
                partial, thinking_mode=thinking_mode,
                drop_thinking=False, add_default_bos_token=(i == 0),
            )
            ids = self.tokenizer.encode(text)
            new_part = ids[prev_len:]
            role = msgs[i].get("role")
            if role == "assistant":
                input_ids.extend(new_part)
                label_ids.extend(new_part)
            else:
                input_ids.extend(new_part)
                label_ids.extend([self.ignore_index] * len(new_part))
            prev_len = len(ids)

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            label_ids = label_ids[:self.max_seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(label_ids, dtype=torch.long),
        }


# ============================================================
# Config + Trainer
# ============================================================

@dataclass
class AdaptiveThinkingConfig(SFTConfig):
    """自适应思考训练配置。"""
    # Router
    router_strategy: str = "auto"        # fixed | auto | learned
    default_mode: str = "chat"
    difficulty_threshold: float = 0.5

    # 一些覆盖
    learning_rate: float = 1.0e-5
    use_aux_loss: bool = False


class AdaptiveThinkingTrainer(SFTTrainer):
    """
    使用 AdaptiveThinkingDataset 的 SFT 训练器。
    """

    def __init__(self, config: AdaptiveThinkingConfig, model, tokenizer):
        super().__init__(config=config, model=model, tokenizer=tokenizer)
        self.config: AdaptiveThinkingConfig = config
        self.router = ModeRouter(
            strategy=config.router_strategy,
            default_mode=config.default_mode,
            threshold=config.difficulty_threshold,
        )

    def get_train_dataset(self) -> Dataset:
        if self._train_ds is None:
            self._train_ds = AdaptiveThinkingDataset(
                paths=self.config.train_data_paths,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                router=self.router,
                cache_dir=self.config.cache_dir,
            )
        return self._train_ds

    def get_eval_dataset(self) -> Optional[Dataset]:
        if not self.config.eval_data_paths:
            return None
        if self._eval_ds is None:
            self._eval_ds = AdaptiveThinkingDataset(
                paths=self.config.eval_data_paths,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                router=self.router,
                cache_dir=self.config.cache_dir,
            )
        return self._eval_ds
