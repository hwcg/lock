"""
Trajectory rollout 工具：把 AgenticTrajectory 转成可训练的 token 序列。

关键设计：
- 整条 trajectory 编码为一个长 token 序列
- response_mask 仅在 assistant 的"自回归生成部分"=1
- tool 结果（user 消息内 <tool_result>）= 0（不计 loss）
- 同时计算 old_logp（采样时记录无法跨 turn 累计，这里重新前向）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from deepseek_v4.tokenizer.encoding import encode_messages
from deepseek_v4.training.agentic_rl.environment import AgenticTrajectory


def encode_trajectory_with_mask(
    trajectory: AgenticTrajectory,
    tokenizer,
    max_seq_len: int = 4096,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    把一条 trajectory 编码为 (token_ids, response_mask)。

    渐进式 encode：每加一条消息计算增量 token 数；assistant 消息对应区域 mask=1。
    """
    msgs = trajectory.messages
    if not msgs:
        return None

    token_ids: List[int] = []
    response_mask: List[int] = []
    prev_len = 0

    for i in range(len(msgs)):
        partial = msgs[:i + 1]
        try:
            cur_text = encode_messages(
                partial, thinking_mode="chat",
                drop_thinking=False, add_default_bos_token=(i == 0),
            )
        except Exception:
            return None
        cur_ids = tokenizer.encode(cur_text)
        new_part = cur_ids[prev_len:]
        if msgs[i].get("role") == "assistant":
            token_ids.extend(new_part)
            response_mask.extend([1] * len(new_part))
        else:
            token_ids.extend(new_part)
            response_mask.extend([0] * len(new_part))
        prev_len = len(cur_ids)

    if len(token_ids) > max_seq_len:
        # 右截断（保留前面 context）
        token_ids = token_ids[:max_seq_len]
        response_mask = response_mask[:max_seq_len]

    if sum(response_mask) == 0:
        return None
    return token_ids, response_mask


@torch.no_grad()
def collect_trajectories(
    model,
    ref_model,
    tokenizer,
    initial_messages_list: List[List[Dict[str, Any]]],
    environment,
    generation_config,
    device: torch.device,
    reward_fn,
    max_seq_len: int = 4096,
) -> Dict[str, Any]:
    """
    收集一批 trajectory + 计算 per-token old_logp / ref_logp / reward。

    Returns:
        dict with:
            sequences:       [N, S]      整条 trajectory 的 token ids
            response_mask:   [N, S]      assistant 生成部分
            old_logp:        [N, S]      当前 policy 的 per-token logp
            ref_logp:        [N, S]      ref 的 per-token logp
            rewards:         [N]         trajectory 级别 reward
            success:         [N]         成功/失败标志
            trajectories:    [N]         原始 AgenticTrajectory 对象（用于 reward）
    """
    trajectories: List[AgenticTrajectory] = []
    completions: List[str] = []

    # 1. 与环境交互
    for init_msgs in initial_messages_list:
        traj = environment.run(
            model=model, tokenizer=tokenizer,
            initial_messages=init_msgs,
            generation_config=generation_config,
            device=device,
        )
        trajectories.append(traj)
        completions.append(traj.final_text)

    # 2. 计算 reward（用户自定义 reward_fn）
    rewards = torch.tensor(
        reward_fn(completions=completions, trajectories=trajectories),
        dtype=torch.float32, device=device,
    )

    # 3. 把每条 trajectory 编码成 token 序列
    encoded: List[Tuple[List[int], List[int]]] = []
    for traj in trajectories:
        out = encode_trajectory_with_mask(traj, tokenizer, max_seq_len=max_seq_len)
        encoded.append(out if out is not None else ([], []))

    # 4. padding 到同一长度
    pad_id = tokenizer.pad_token_id
    max_len = max((len(x[0]) for x in encoded), default=1)
    max_len = max(max_len, 1)
    n = len(encoded)
    sequences = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
    response_mask = torch.zeros((n, max_len), dtype=torch.long, device=device)
    for i, (ids, m) in enumerate(encoded):
        L = len(ids)
        if L == 0:
            continue
        sequences[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
        response_mask[i, :L] = torch.tensor(m, dtype=torch.long, device=device)

    # 5. 计算 old_logp / ref_logp（per-token，仅 response_mask 有效）
    from deepseek_v4.training.grpo.trainer import _per_token_logp_from_logits

    def _per_token(m, seq, mask):
        out = m(input_ids=seq, attention_mask=(seq != pad_id).long() | mask, use_cache=False)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        return _per_token_logp_from_logits(logits, seq, mask)

    # 分批以省显存
    chunk = 4
    old_logp = torch.zeros_like(sequences, dtype=torch.float32)
    ref_logp = torch.zeros_like(sequences, dtype=torch.float32)
    for s in range(0, n, chunk):
        e = s + chunk
        old_logp[s:e] = _per_token(model, sequences[s:e], response_mask[s:e]).float()
        ref_logp[s:e] = _per_token(ref_model, sequences[s:e], response_mask[s:e]).float()

    success = torch.tensor([t.success for t in trajectories], dtype=torch.float32, device=device)

    return {
        "sequences": sequences.cpu(),
        "response_mask": response_mask.cpu(),
        "old_logp": old_logp.cpu(),
        "ref_logp": ref_logp.cpu(),
        "rewards": rewards.cpu(),
        "success": success.cpu(),
        "trajectories": trajectories,
    }
