"""
Reward Model 作为 reward 函数。
"""
from __future__ import annotations

from typing import Any, List, Optional

import torch

from deepseek_v4.training.rewards.base import RewardFunction


class RewardModelReward(RewardFunction):
    """
    把训练好的 DeepseekV4RewardModel 包装为 reward 函数。

    输入 prompts + completions 文本，内部 tokenize → forward → 标量。
    """
    name = "rm"

    def __init__(
        self,
        reward_model,
        tokenizer,
        device: Optional[torch.device] = None,
        max_length: int = 2048,
        batch_size: int = 8,
        normalize: bool = False,
    ):
        self.rm = reward_model
        self.tokenizer = tokenizer
        self.device = device or next(reward_model.parameters()).device
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self._running_mean = 0.0
        self._running_var = 1.0
        self._count = 0

    @torch.no_grad()
    def __call__(
        self,
        completions: List[str],
        references: Optional[List[Any]] = None,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        if prompts is None:
            prompts = [""] * len(completions)
        assert len(prompts) == len(completions)

        scores: List[float] = []
        for start in range(0, len(completions), self.batch_size):
            batch_prompts = prompts[start:start + self.batch_size]
            batch_comp = completions[start:start + self.batch_size]
            # 拼接为完整序列
            texts = [p + c for p, c in zip(batch_prompts, batch_comp)]
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            r = self.rm(input_ids=input_ids, attention_mask=attn)
            scores.extend(r.detach().cpu().float().tolist())

        if self.normalize:
            # 更新 running stats
            for s in scores:
                self._count += 1
                d = s - self._running_mean
                self._running_mean += d / self._count
                d2 = s - self._running_mean
                self._running_var += d * d2
            if self._count > 1:
                std = max((self._running_var / (self._count - 1)) ** 0.5, 1e-6)
                scores = [(s - self._running_mean) / std for s in scores]

        return scores
