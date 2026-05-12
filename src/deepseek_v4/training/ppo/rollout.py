"""
PPO Rollout 收集。

流程：
    1. 从 prompt dataset 取 batch
    2. policy.generate 出 responses
    3. 计算每 token policy log_prob（采样时已得到）
    4. ref_model 算 ref log_prob
    5. RM 算 sequence reward
    6. value_head 算 per-token value
    7. 计算 per-token reward = -β·KL_t + (RM at last)
    8. GAE 算 advantage / return
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from deepseek_v4.inference.generation import GenerationConfig, generate
from deepseek_v4.training.ppo.gae import compute_advantages_with_whitening


# ============================================================
# Buffer
# ============================================================

@dataclass
class RolloutBuffer:
    """
    存放一轮采样到的所有数据。

    所有 tensor 形状（已 padding 到 batch 内最长）：
        prompt_ids:    [N, P]
        response_ids:  [N, T]
        response_mask: [N, T]
        old_logprobs:  [N, T]    采样时的 log p
        ref_logprobs:  [N, T]    参考模型 log p
        values:        [N, T]    value 估计
        rewards:       [N, T]    每 token 即时回报（含 KL 惩罚）
        advantages:    [N, T]
        returns:       [N, T]
        rm_scores:     [N]       原始 RM 分数（仅 logging 用）
    """
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    rm_scores: torch.Tensor

    def __len__(self) -> int:
        return self.prompt_ids.shape[0]

    def to(self, device: torch.device) -> "RolloutBuffer":
        return RolloutBuffer(**{k: v.to(device) for k, v in self.__dict__.items()})

    def iter_minibatches(
        self,
        mini_bs: int,
        shuffle: bool = True,
        device: Optional[torch.device] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        n = len(self)
        idx = torch.randperm(n) if shuffle else torch.arange(n)
        for i in range(0, n, mini_bs):
            sl = idx[i:i + mini_bs]
            mb = {k: getattr(self, k)[sl] for k in self.__dataclass_fields__.keys()}
            if device is not None:
                mb = {k: v.to(device) for k, v in mb.items()}
            yield mb


# ============================================================
# 计算 per-token log_prob（给定输入序列）
# ============================================================

def _per_token_logprob(
    logits: torch.Tensor,
    sequence: torch.Tensor,
    response_mask_full: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        logits: [B, S, V]            full sequence logits
        sequence: [B, S]             full sequence ids (prompt + response)
        response_mask_full: [B, S]   1 = response token (要计 logp)
    Returns:
        per_token_logp: [B, S-1]     注意 shift 1
                                     仅 response 区域有效，其余 0
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = sequence[:, 1:]
    shift_mask = response_mask_full[:, 1:]
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    per = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return per * shift_mask.float()


# ============================================================
# Reward 处理
# ============================================================

def _build_response_mask_full(
    sequences: torch.Tensor, prompt_lens: torch.Tensor, pad_id: int,
) -> torch.Tensor:
    """构造 [B, S] 的 response mask。"""
    B, S = sequences.shape
    pos = torch.arange(S, device=sequences.device)[None, :]
    in_response = pos >= prompt_lens[:, None]
    not_pad = sequences != pad_id
    return (in_response & not_pad).long()


# ============================================================
# 收集 rollouts
# ============================================================

@torch.no_grad()
def collect_rollouts(
    policy_value_model,
    ref_model,
    reward_model,
    prompts: List[torch.Tensor],     # 每个元素 [P_i]
    pad_token_id: int,
    gen_config: GenerationConfig,
    kl_coef: float,
    gamma: float = 1.0,
    lam: float = 0.95,
    whiten_advantages: bool = True,
    device: Optional[torch.device] = None,
    micro_batch_size: int = 4,
) -> RolloutBuffer:
    """
    给一批 prompts 收集 rollout 数据。

    Args:
        prompts: list of 1D tensors（不同长度）
        kl_coef: 当前 KL 惩罚系数
    Returns:
        RolloutBuffer（在 CPU，便于多 epoch 训练，按需 to(device)）
    """
    if device is None:
        device = next(policy_value_model.parameters()).device

    all_prompt_ids: List[torch.Tensor] = []
    all_response_ids: List[torch.Tensor] = []
    all_response_mask: List[torch.Tensor] = []
    all_old_logp: List[torch.Tensor] = []
    all_ref_logp: List[torch.Tensor] = []
    all_values: List[torch.Tensor] = []
    all_rm: List[torch.Tensor] = []

    # 切 micro batch
    for start in range(0, len(prompts), micro_batch_size):
        batch_prompts = prompts[start:start + micro_batch_size]
        # 左 padding
        max_p = max(p.shape[0] for p in batch_prompts)
        prompt_ids = torch.full((len(batch_prompts), max_p), pad_token_id, dtype=torch.long, device=device)
        prompt_mask = torch.zeros_like(prompt_ids)
        for i, p in enumerate(batch_prompts):
            prompt_ids[i, -p.shape[0]:] = p.to(device)
            prompt_mask[i, -p.shape[0]:] = 1
        prompt_lens = prompt_mask.sum(dim=1)

        # ----- 1. 生成 -----
        gen_cfg = GenerationConfig(**{**gen_config.__dict__, "return_log_probs": True})
        gen = generate(
            policy_value_model.policy if hasattr(policy_value_model, "policy") else policy_value_model,
            input_ids=prompt_ids, attention_mask=prompt_mask, config=gen_cfg,
        )
        sequences = gen["sequences"]              # [B, P+T]
        response_ids = gen["responses"]           # [B, T]
        response_mask = gen["response_mask"]      # [B, T]
        old_logp_response = gen["log_probs"]      # [B, T]

        # 全序列 mask（response 部分 = 1）
        full_mask = torch.zeros_like(sequences)
        full_mask[:, max_p:] = response_mask
        seq_attn_mask = torch.ones_like(sequences)
        seq_attn_mask[:, :max_p] = prompt_mask

        # ----- 2. 计算 ref_logprob 与 value 与 RM -----
        # ref logp
        ref_out = ref_model(input_ids=sequences, attention_mask=seq_attn_mask, use_cache=False)
        ref_logits = ref_out["logits"] if isinstance(ref_out, dict) else ref_out.logits
        ref_per = _per_token_logprob(ref_logits, sequences, full_mask)  # [B, S-1]

        # value
        pv_out = policy_value_model(
            input_ids=sequences, attention_mask=seq_attn_mask,
            return_logits=False, return_values=True, use_cache=False,
        )
        values_full = pv_out["values"]   # [B, S]
        # 对齐到 shift 后（与 logp 同位）
        values_per = values_full[:, :-1] * full_mask[:, 1:].float()

        # RM
        rm_scores = reward_model(input_ids=sequences, attention_mask=seq_attn_mask)  # [B]

        # ----- 3. 提取 response 部分（去掉 prompt） -----
        # ref_per/values_per 在 [B, S-1]，response 起点为 max_p-1（shift 后第一个生成 token 对应位置）
        T = response_ids.shape[1]
        # response 区在 shift 序列里的位置：(max_p - 1) 到 (max_p - 1 + T - 1)
        ref_logp_resp = ref_per[:, max_p - 1:max_p - 1 + T] * response_mask.float()
        values_resp = values_per[:, max_p - 1:max_p - 1 + T] * response_mask.float()

        # ----- 4. KL 与 reward 信号 -----
        # KL_t ≈ logπ(a_t) - logπ_ref(a_t)
        kl_per_token = (old_logp_response - ref_logp_resp) * response_mask.float()
        rewards_per_token = -kl_coef * kl_per_token

        # 在最后有效位置加 RM 分数
        last_idx = response_mask.long().sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        for b in range(rewards_per_token.shape[0]):
            t = last_idx[b].item()
            rewards_per_token[b, t] = rewards_per_token[b, t] + rm_scores[b]

        # 收集（搬到 cpu 减小显存）
        all_prompt_ids.append(prompt_ids.cpu())
        all_response_ids.append(response_ids.cpu())
        all_response_mask.append(response_mask.cpu())
        all_old_logp.append(old_logp_response.cpu())
        all_ref_logp.append(ref_logp_resp.cpu())
        all_values.append(values_resp.cpu())
        all_rm.append(rm_scores.cpu())

    # ----- 拼接所有 micro batch（先按最大长度 padding）-----
    def _pad_cat_2d(tensors: List[torch.Tensor], pad_value=0) -> torch.Tensor:
        max_len = max(t.shape[1] for t in tensors)
        out = []
        for t in tensors:
            if t.shape[1] < max_len:
                pad = torch.full(
                    (t.shape[0], max_len - t.shape[1]), pad_value, dtype=t.dtype,
                )
                t = torch.cat([t, pad], dim=1)
            out.append(t)
        return torch.cat(out, dim=0)

    prompt_ids = _pad_cat_2d(all_prompt_ids, pad_value=pad_token_id)
    response_ids = _pad_cat_2d(all_response_ids, pad_value=pad_token_id)
    response_mask = _pad_cat_2d(all_response_mask, pad_value=0)
    old_logp = _pad_cat_2d(all_old_logp, pad_value=0.0)
    ref_logp = _pad_cat_2d(all_ref_logp, pad_value=0.0)
    values = _pad_cat_2d(all_values, pad_value=0.0)
    rm_scores = torch.cat(all_rm, dim=0)

    # 把所有 reward 信号统一对齐到最大 T
    # 重新计算 rewards 在 padded 后的对齐
    # 为简洁，重新算一次：
    rewards = -kl_coef * (old_logp - ref_logp) * response_mask.float()
    last_idx = response_mask.long().sum(dim=1) - 1
    last_idx = last_idx.clamp(min=0)
    rewards[torch.arange(rewards.shape[0]), last_idx] += rm_scores

    # GAE
    advantages, returns = compute_advantages_with_whitening(
        rewards=rewards, values=values, mask=response_mask.float(),
        gamma=gamma, lam=lam, whiten=whiten_advantages,
    )

    return RolloutBuffer(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        old_logprobs=old_logp,
        ref_logprobs=ref_logp,
        values=values,
        rewards=rewards,
        advantages=advantages,
        returns=returns,
        rm_scores=rm_scores,
    )
