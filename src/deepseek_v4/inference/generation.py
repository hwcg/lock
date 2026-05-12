"""
最小但生产级的自回归生成实现。

设计目标：
1. 支持 V4 三种 cache（sliding / HCA / CSA）
2. 支持 greedy / temperature / top-k / top-p
3. 支持 batch 生成（左 padding）
4. 用于 PPO rollout 与简单推理；完整服务端在 Part 9。

注：
- 不依赖 transformers GenerationMixin，从 0 实现
- 同时返回 token ids 和每个采样位置的 log_prob（PPO 需要）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ============================================================
# 配置
# ============================================================

@dataclass
class GenerationConfig:
    """生成参数。"""
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 0           # 0 = 关闭
    top_p: float = 1.0       # 1.0 = 关闭
    repetition_penalty: float = 1.0
    stop_token_ids: List[int] = field(default_factory=list)
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 0
    return_log_probs: bool = False    # PPO rollout 需要 True


# ============================================================
# Logits Warpers
# ============================================================

def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0 or temperature == 1.0:
        return logits
    return logits / temperature


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # 保留累积概率刚好不超过 top_p 的部分
    sorted_mask = cum > top_p
    # 第一个总是保留
    sorted_mask[..., 0] = False
    # 把 mask 反映射回原顺序
    mask = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def _apply_repetition_penalty(
    logits: torch.Tensor, prev_ids: torch.Tensor, penalty: float,
) -> torch.Tensor:
    """对历史 token 的 logit 施加惩罚（CTRL 风格）。"""
    if penalty == 1.0:
        return logits
    # logits: [B, V], prev_ids: [B, T]
    score = torch.gather(logits, 1, prev_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, prev_ids, score)
    return logits


def prepare_logits_warper(cfg: GenerationConfig) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """构造一个串联的 logits 处理函数。"""

    def warp(logits: torch.Tensor, prev_ids: torch.Tensor) -> torch.Tensor:
        logits = _apply_repetition_penalty(logits, prev_ids, cfg.repetition_penalty)
        logits = _apply_temperature(logits, cfg.temperature)
        logits = _apply_top_k(logits, cfg.top_k)
        logits = _apply_top_p(logits, cfg.top_p)
        return logits

    return warp


# ============================================================
# Sampling
# ============================================================

def sample_token(
    logits: torch.Tensor,
    do_sample: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 logits 采样下一个 token。

    Returns:
        next_token: [B]
        log_prob:   [B]      采样位置的 log p
    """
    if do_sample:
        probs = F.softmax(logits, dim=-1)
        # 防止 -inf 造成 NaN
        probs = torch.nan_to_num(probs, nan=0.0)
        # 防全 0
        if (probs.sum(dim=-1) == 0).any():
            probs = torch.softmax(torch.zeros_like(logits), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        next_token = logits.argmax(dim=-1)
    log_prob = F.log_softmax(logits.float(), dim=-1).gather(-1, next_token[:, None]).squeeze(-1)
    return next_token, log_prob


# ============================================================
# Generate
# ============================================================

@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    config: Optional[GenerationConfig] = None,
) -> dict:
    """
    自回归生成。

    Args:
        model:          DeepseekV4ForCausalLM
        input_ids:      [B, S]，左 padding
        attention_mask: [B, S]，1=有效；None 则视为全 1
        config:         GenerationConfig
    Returns:
        {
            "sequences":   [B, S+T]                 (含 prompt)
            "responses":   [B, T_max]               (仅生成部分)
            "log_probs":   [B, T_max]   if return_log_probs
            "response_mask": [B, T_max]             (1=有效, 0=已停止)
        }
    """
    cfg = config or GenerationConfig()
    device = input_ids.device
    B, S_in = input_ids.shape

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    # 生成所有 token 的 buffer
    response_ids = torch.full(
        (B, cfg.max_new_tokens), cfg.pad_token_id, dtype=torch.long, device=device,
    )
    response_log_probs = torch.zeros(
        (B, cfg.max_new_tokens), dtype=torch.float32, device=device,
    )
    response_mask = torch.zeros((B, cfg.max_new_tokens), dtype=torch.long, device=device)
    finished = torch.zeros((B,), dtype=torch.bool, device=device)

    # ---- 第一次：把整段 prompt 送进去，建立 cache ----
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_kv = out["past_key_values"] if isinstance(out, dict) else out.past_key_values
    last_logits = (out["logits"] if isinstance(out, dict) else out.logits)[:, -1, :]

    warper = prepare_logits_warper(cfg)
    stop_set = set(cfg.stop_token_ids) | {cfg.eos_token_id}

    cur_ids = input_ids.clone()  # 用于 repetition penalty
    next_token, log_prob = sample_token(
        warper(last_logits.float(), cur_ids),
        do_sample=cfg.do_sample,
    )

    for t in range(cfg.max_new_tokens):
        # 写入 buffer（仅未结束的样本）
        active = ~finished
        response_ids[active, t] = next_token[active]
        response_log_probs[active, t] = log_prob[active]
        response_mask[active, t] = 1

        # 检查停止
        for eos in stop_set:
            finished = finished | (next_token == eos)
        if finished.all():
            break

        # 拼接 prompt（用于 repetition penalty）
        cur_ids = torch.cat([cur_ids, next_token[:, None]], dim=1)

        # 增量 forward
        if t + 1 == cfg.max_new_tokens:
            break
        # 单 token 输入，扩展 attention mask
        next_in = next_token[:, None]
        next_attn = torch.cat(
            [attention_mask, torch.ones((B, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        out = model(
            input_ids=next_in,
            attention_mask=next_attn,
            past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = out["past_key_values"] if isinstance(out, dict) else out.past_key_values
        last_logits = (out["logits"] if isinstance(out, dict) else out.logits)[:, -1, :]
        attention_mask = next_attn

        next_token, log_prob = sample_token(
            warper(last_logits.float(), cur_ids),
            do_sample=cfg.do_sample,
        )

    # 截掉后面没用到的部分
    used = int(response_mask.sum(dim=1).max().item())
    used = max(used, 1)
    response_ids = response_ids[:, :used]
    response_log_probs = response_log_probs[:, :used]
    response_mask = response_mask[:, :used]

    sequences = torch.cat([input_ids, response_ids], dim=1)
    result = {
        "sequences": sequences,
        "responses": response_ids,
        "response_mask": response_mask,
    }
    if cfg.return_log_probs:
        result["log_probs"] = response_log_probs
    return result
