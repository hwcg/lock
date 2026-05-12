"""
GRPO (Group Relative Policy Optimization) Trainer。

参考：DeepSeek-Math (Shao et al., 2024) / DeepSeek-R1

核心思想：
- 不需要 value model（critic），从而显存减半
- 对每个 prompt 采样 G 个 completion → 用组内 reward 标准化作为 advantage：
      A_{i,t} = (r_i - mean(r_group)) / (std(r_group) + ε)   (token 内复用同一 advantage)
- 用 PPO clipped objective + KL penalty 更新 policy
- KL penalty 直接加入 loss（与 PPO 把 KL 折进 reward 不同）

损失：
    L = E[ min(ratio · A, clip(ratio, 1-ε, 1+ε) · A) ] - β · KL(π || π_ref)
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepseek_v4.distributed.utils import (
    all_reduce_mean, get_rank, get_world_size, is_distributed, is_main_process,
)
from deepseek_v4.inference.generation import GenerationConfig, generate
from deepseek_v4.modeling.model import DeepseekV4ForCausalLM
from deepseek_v4.training.base_trainer import TrainerConfig
from deepseek_v4.training.checkpoint import CheckpointManager
from deepseek_v4.training.grad_checkpoint import enable_gradient_checkpointing
from deepseek_v4.training.optim import build_optimizer, build_scheduler
from deepseek_v4.training.ppo.rollout import PromptDataset, _prompt_collate  # 复用
from deepseek_v4.training.rewards.base import RewardFunction, build_reward_from_config
from deepseek_v4.utils.logger import (
    MetricLogger, MultiLogger, PAUSE_CONTROLLER, get_logger, setup_logging,
)
from deepseek_v4.utils.seed import set_seed
from deepseek_v4.utils.timer import Stopwatch, format_time

logger = get_logger(__name__)


# ============================================================
# 配置
# ============================================================

@dataclass
class GRPOConfig(TrainerConfig):
    """GRPO 配置。"""

    # 数据
    train_data_paths: List[str] = field(default_factory=list)
    prompt_field: str = "messages"
    reference_field: Optional[str] = "answer"   # 用于 reward function
    max_prompt_len: int = 1024
    cache_dir: Optional[str] = "cache/datasets"

    # 模型
    model_config_path: str = "configs/model/mini_2b.json"
    init_from_checkpoint: str = "checkpoints/sft/checkpoint-final"
    tokenizer_path: str = "checkpoints/tokenizer"

    # 生成
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    group_size: int = 8                # 每个 prompt 采样数 G
    prompts_per_step: int = 4          # 每次取 prompt 数（per device）
    gen_micro_batch_size: int = 8

    # GRPO 损失
    cliprange: float = 0.2
    beta_kl: float = 0.04              # KL penalty 系数（DeepSeek-Math 默认 0.04）
    ppo_epochs: int = 1                # GRPO 通常 1（on-policy）
    advantage_eps: float = 1e-8

    # KL 估计方式
    kl_estimator: str = "k3"           # k1 | k2 | k3 (Schulman 2020)

    # 训练循环
    num_iters: int = 1000
    save_iters: int = 50

    # Reward 配置（列表 dict，由 rewards.base.build_reward_from_config 解析）
    rewards: List[Dict[str, Any]] = field(default_factory=list)

    # 默认覆盖
    learning_rate: float = 1.0e-6
    weight_decay: float = 0.0
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True


# ============================================================
# 内部 buffer
# ============================================================

@dataclass
class GRPOBuffer:
    """一个 prompt-batch 的所有 G·B 样本。"""
    sequences: torch.Tensor          # [N, S]
    response_mask: torch.Tensor      # [N, S]
    old_logp: torch.Tensor           # [N, S]     仅 response 区有效
    ref_logp: torch.Tensor           # [N, S]
    advantages: torch.Tensor         # [N]        每序列一个
    rewards_raw: torch.Tensor        # [N]        原始 reward（用于 logging）
    group_ids: torch.Tensor          # [N]        每个样本所属组

    def __len__(self):
        return self.sequences.shape[0]


# ============================================================
# 工具
# ============================================================

def _per_token_logp_from_logits(
    logits: torch.Tensor, sequence: torch.Tensor, response_mask: torch.Tensor,
) -> torch.Tensor:
    """logits/sequence/mask shape: [B,S,V] / [B,S] / [B,S]
       返回 [B,S]（已 shift），对应每个 token 的 log p（仅 response 部分有效）。"""
    shift_logits = logits[:, :-1, :]
    shift_labels = sequence[:, 1:]
    shift_mask = response_mask[:, 1:].float()
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    per = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    per = per * shift_mask
    # 把它 pad 回 [B,S] 以方便对齐（左 pad 一个 0）
    pad = torch.zeros(per.shape[0], 1, device=per.device, dtype=per.dtype)
    return torch.cat([pad, per], dim=1)   # 现在 per[:, t] 对应位置 t 的 logp


def _kl_per_token(
    pi_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor, estimator: str = "k3",
) -> torch.Tensor:
    """
    每 token KL 估计（仅 response 区）：
        k1: KL ≈ logπ - logπ_ref         (one-sample，无偏但方差大；可正可负)
        k2: KL ≈ 0.5 · (logπ - logπ_ref)²
        k3: KL ≈ (logπ_ref - logπ) + exp(logπ - logπ_ref) - 1  (>=0，低方差)
    """
    diff = pi_logp - ref_logp
    if estimator == "k1":
        kl = diff
    elif estimator == "k2":
        kl = 0.5 * diff ** 2
    elif estimator == "k3":
        # exp(diff) 数值上可能爆，限幅
        diff_c = diff.clamp(min=-20, max=20)
        kl = -diff + diff_c.exp() - 1
    else:
        raise ValueError(f"Unknown kl_estimator: {estimator}")
    return kl * mask


# ============================================================
# Trainer
# ============================================================

class GRPOTrainer:
    """
    GRPO 训练器（无 value model）。
    """

    def __init__(
        self,
        config: GRPOConfig,
        policy: DeepseekV4ForCausalLM,
        tokenizer,
        reward_fn: RewardFunction,
        ref_model: Optional[DeepseekV4ForCausalLM] = None,
    ):
        self.config = config
        self.policy = policy
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.ref_model = ref_model

        self.iter_idx = 0
        self.global_step = 0
        self.start_time = 0.0
        self.metric_logger = MetricLogger(window=20)
        self.timer = Stopwatch()
        self.tracker_logger: Optional[MultiLogger] = None

        self.optimizer = None
        self.scheduler = None
        self.train_loader: Optional[DataLoader] = None
        self.ckpt_mgr: Optional[CheckpointManager] = None
        self.device: Optional[torch.device] = None
        self._raw_prompts: Dict[int, str] = {}   # 索引 → 原始 prompt 文本，用于 RewardModel

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        from deepseek_v4.distributed.utils import setup_distributed
        setup_distributed(backend=self.config.distributed_backend)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        log_dir = self.config.log_dir or str(Path(self.config.output_dir) / "logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(level="INFO", log_file=f"{log_dir}/rank{get_rank()}.log", rank=get_rank())

        if is_main_process():
            logger.info("=" * 70)
            logger.info(f"  GRPOTrainer  world_size={get_world_size()}")
            logger.info(f"  group_size={self.config.group_size}  prompts_per_step={self.config.prompts_per_step}")
            logger.info(f"  beta_kl={self.config.beta_kl}  cliprange={self.config.cliprange}")
            logger.info(f"  kl_estimator={self.config.kl_estimator}")
            logger.info("=" * 70)

        set_seed(self.config.seed + get_rank())

        # ref：默认深拷贝 policy
        if self.ref_model is None:
            logger.info("[GRPO] making frozen reference (deep copy of policy)")
            self.ref_model = copy.deepcopy(self.policy)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval().to(self.device)

        # policy
        self.policy = self.policy.to(self.device)
        if self.config.gradient_checkpointing:
            enable_gradient_checkpointing(self.policy)

        # DDP
        if is_distributed():
            self.policy = nn.parallel.DistributedDataParallel(
                self.policy,
                device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        # 优化器
        self.optimizer = build_optimizer(
            self.policy.module if hasattr(self.policy, "module") else self.policy,
            name=self.config.optimizer,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            name=self.config.scheduler,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.num_iters * self.config.ppo_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

        # 数据
        ds = PromptDataset(
            paths=self.config.train_data_paths,
            tokenizer=self.tokenizer,
            prompt_field=self.config.prompt_field,
            max_prompt_len=self.config.max_prompt_len,
        )
        # 加载 reference field（如标准答案）—— 这里简单做：再过一遍 jsonl 拿 references
        from deepseek_v4.utils.io import read_jsonl
        self._references: List[Any] = []
        self._raw_prompt_texts: List[str] = []
        if self.config.reference_field:
            for p in self.config.train_data_paths:
                for row in read_jsonl(p):
                    ref = row.get(self.config.reference_field)
                    self._references.append(ref)
                    if "messages" in row:
                        try:
                            self._raw_prompt_texts.append(
                                self.tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True)
                            )
                        except Exception:
                            self._raw_prompt_texts.append("")
                    else:
                        self._raw_prompt_texts.append(str(row.get(self.config.prompt_field, "")))
        else:
            self._references = [None] * len(ds)
            self._raw_prompt_texts = [""] * len(ds)

        # 用 index 作为 batch 元素，便于查 references
        self._dataset = ds
        self.train_loader = DataLoader(
            range(len(ds)),
            batch_size=self.config.prompts_per_step,
            shuffle=True, drop_last=True,
            num_workers=0,
            collate_fn=lambda b: list(b),
        )

        # Checkpoint
        self.ckpt_mgr = CheckpointManager(
            output_dir=self.config.output_dir,
            keep_last=self.config.keep_last,
            keep_best=self.config.keep_best,
            metric_name=self.config.metric_name,
            metric_mode="max",
        )

        if is_main_process():
            self.tracker_logger = MultiLogger(
                backends=self.config.logger_backends,
                rank=0, enable_signal_control=self.config.enable_signal_control,
            )
            self.tracker_logger.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=self.config.to_dict(),
                output_dir=str(Path(self.config.output_dir) / "tracker"),
            )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self) -> None:
        self.setup()
        self.start_time = time.time()
        cfg = self.config

        prompt_iter = self._infinite_loader(self.train_loader)
        for self.iter_idx in range(cfg.num_iters):
            self._train_one_iter(prompt_iter)
            PAUSE_CONTROLLER.wait_if_paused()
            if PAUSE_CONTROLLER.should_save_and_exit():
                self._save_ckpt()
                break
            if cfg.save_iters > 0 and (self.iter_idx + 1) % cfg.save_iters == 0:
                self._save_ckpt()

        self._save_ckpt()
        if self.tracker_logger is not None:
            self.tracker_logger.finish()

    # ------------------------------------------------------------------
    # 一 iter
    # ------------------------------------------------------------------

    def _train_one_iter(self, prompt_iter):
        cfg = self.config
        indices: List[int] = next(prompt_iter)

        # ----- 1. Rollout：每个 prompt 采 G 个 -----
        with self.timer.track("rollout"):
            buffer = self._collect_rollouts(indices)

        if buffer is None or len(buffer) == 0:
            return

        # ----- 2. Optimize -----
        log_metrics = {
            "rm/mean": float(buffer.rewards_raw.mean()),
            "rm/std":  float(buffer.rewards_raw.std().clamp(min=1e-8)),
            "rm/min":  float(buffer.rewards_raw.min()),
            "rm/max":  float(buffer.rewards_raw.max()),
            "resp_len": float(buffer.response_mask.sum(dim=1).float().mean()),
            "advantage_abs_mean": float(buffer.advantages.abs().mean()),
        }

        # GRPO 通常 1 epoch（on-policy），可以 >1 但要小心
        agg_metrics: Dict[str, List[float]] = {}
        for epoch in range(cfg.ppo_epochs):
            for mb_indices in self._iter_minibatches(
                len(buffer), micro_bs=cfg.micro_batch_size,
            ):
                mb = self._slice_buffer(buffer, mb_indices)
                with self.timer.track("forward"):
                    loss_dict = self._compute_grpo_loss(mb)
                with self.timer.track("backward"):
                    self.optimizer.zero_grad(set_to_none=True)
                    loss_dict["loss"].backward()
                with self.timer.track("optim"):
                    gn = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), max_norm=cfg.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                self.global_step += 1
                for k in ("loss", "policy_loss", "kl_loss", "approx_kl", "clip_frac"):
                    agg_metrics.setdefault(k, []).append(float(loss_dict[k]))
                agg_metrics.setdefault("lr", []).append(float(self.scheduler.get_last_lr()[0]))
                agg_metrics.setdefault("grad_norm", []).append(float(gn))

        for k, v in agg_metrics.items():
            avg = sum(v) / len(v)
            log_metrics[k] = avg
            self.metric_logger.update(**{k: avg})

        if is_main_process():
            elapsed = time.time() - self.start_time
            logger.info(
                f"[GRPO] iter={self.iter_idx} step={self.global_step}  "
                f"rm={log_metrics['rm/mean']:+.3f}±{log_metrics['rm/std']:.3f}  "
                f"adv={log_metrics['advantage_abs_mean']:.3f}  "
                f"loss={log_metrics['loss']:.3f}  "
                f"kl={log_metrics['approx_kl']:.4f}  "
                f"clip={log_metrics['clip_frac']:.3f}  "
                f"lr={log_metrics['lr']:.2e}  elapsed={format_time(elapsed)}"
            )
            if self.tracker_logger is not None:
                self.tracker_logger.log(log_metrics, step=self.iter_idx)

    # ------------------------------------------------------------------
    # Rollout（重写：每个 prompt G 次复制 + reward）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_rollouts(self, indices: List[int]) -> Optional[GRPOBuffer]:
        cfg = self.config
        pad_id = self.tokenizer.pad_token_id

        # 把每个 prompt 复制 G 份
        prompts: List[torch.Tensor] = []
        prompt_texts: List[str] = []
        references: List[Any] = []
        group_ids: List[int] = []
        for g_id, idx in enumerate(indices):
            p_ids = self._dataset[idx]
            for _ in range(cfg.group_size):
                prompts.append(p_ids)
                prompt_texts.append(self._raw_prompt_texts[idx])
                references.append(self._references[idx])
                group_ids.append(g_id)

        # ---- Generate ----
        all_sequences: List[torch.Tensor] = []
        all_response_mask_full: List[torch.Tensor] = []
        all_old_logp: List[torch.Tensor] = []
        gen_cfg = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p, top_k=cfg.top_k,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            return_log_probs=True,
        )
        policy_eval = self.policy.module if hasattr(self.policy, "module") else self.policy
        policy_eval.eval()
        for start in range(0, len(prompts), cfg.gen_micro_batch_size):
            batch_prompts = prompts[start:start + cfg.gen_micro_batch_size]
            max_p = max(p.shape[0] for p in batch_prompts)
            prompt_ids = torch.full((len(batch_prompts), max_p), pad_id, dtype=torch.long, device=self.device)
            prompt_mask = torch.zeros_like(prompt_ids)
            for i, p in enumerate(batch_prompts):
                prompt_ids[i, -p.shape[0]:] = p.to(self.device)
                prompt_mask[i, -p.shape[0]:] = 1
            gen = generate(policy_eval, prompt_ids, attention_mask=prompt_mask, config=gen_cfg)
            seq = gen["sequences"]
            resp_mask = gen["response_mask"]
            old_logp_resp = gen["log_probs"]
            full_resp_mask = torch.zeros_like(seq)
            full_resp_mask[:, max_p:] = resp_mask
            # 把 old_logp 也对齐到 [B, S]
            old_logp_full = torch.zeros_like(seq, dtype=torch.float32)
            T = resp_mask.shape[1]
            old_logp_full[:, max_p:max_p + T] = old_logp_resp
            all_sequences.append(seq.cpu())
            all_response_mask_full.append(full_resp_mask.cpu())
            all_old_logp.append(old_logp_full.cpu())

        # 对齐到同一 S（右 padding pad_id / 0）
        def _pad_cat_2d(tensors, pad_value=0, dtype=None):
            max_len = max(t.shape[1] for t in tensors)
            out = []
            for t in tensors:
                if t.shape[1] < max_len:
                    pad = torch.full((t.shape[0], max_len - t.shape[1]), pad_value,
                                     dtype=t.dtype)
                    t = torch.cat([t, pad], dim=1)
                out.append(t)
            return torch.cat(out, dim=0)

        sequences = _pad_cat_2d(all_sequences, pad_value=pad_id).to(self.device)
        response_mask_full = _pad_cat_2d(all_response_mask_full, pad_value=0).to(self.device)
        old_logp_full = _pad_cat_2d(all_old_logp, pad_value=0.0).to(self.device).float()

        # ---- ref logp ----
        ref_logp_full = torch.zeros_like(old_logp_full)
        for start in range(0, sequences.shape[0], cfg.gen_micro_batch_size):
            seq_mb = sequences[start:start + cfg.gen_micro_batch_size]
            mask_mb = response_mask_full[start:start + cfg.gen_micro_batch_size]
            attn = (seq_mb != pad_id).long()
            attn = attn | mask_mb
            attn = attn.clamp(max=1)
            out = self.ref_model(input_ids=seq_mb, attention_mask=attn, use_cache=False)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            per = _per_token_logp_from_logits(logits, seq_mb, mask_mb)
            ref_logp_full[start:start + cfg.gen_micro_batch_size] = per.float()

        # ---- 计算 reward（用 completion 文本）----
        # decode responses
        completions: List[str] = []
        for i in range(sequences.shape[0]):
            resp_positions = response_mask_full[i].nonzero(as_tuple=True)[0]
            if len(resp_positions) == 0:
                completions.append("")
                continue
            start_idx = int(resp_positions[0].item())
            end_idx = int(resp_positions[-1].item()) + 1
            text = self.tokenizer.decode(sequences[i, start_idx:end_idx].tolist(),
                                         skip_special_tokens=False)
            completions.append(text)

        rewards_raw = self.reward_fn(
            completions=completions,
            references=references,
            prompts=prompt_texts,
        )
        rewards_raw = torch.tensor(rewards_raw, dtype=torch.float32, device=self.device)

        # ---- 组内标准化作为 advantage ----
        group_ids_t = torch.tensor(group_ids, dtype=torch.long, device=self.device)
        advantages = torch.zeros_like(rewards_raw)
        n_groups = group_ids_t.max().item() + 1
        for g in range(n_groups):
            mask = (group_ids_t == g)
            if mask.sum() <= 1:
                advantages[mask] = 0.0
                continue
            rg = rewards_raw[mask]
            mean = rg.mean()
            std = rg.std().clamp(min=cfg.advantage_eps)
            advantages[mask] = (rg - mean) / std

        return GRPOBuffer(
            sequences=sequences.cpu(),
            response_mask=response_mask_full.cpu(),
            old_logp=old_logp_full.cpu(),
            ref_logp=ref_logp_full.cpu(),
            advantages=advantages.cpu(),
            rewards_raw=rewards_raw.cpu(),
            group_ids=group_ids_t.cpu(),
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_grpo_loss(self, mb: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cfg = self.config
        sequences = mb["sequences"].to(self.device)
        response_mask = mb["response_mask"].to(self.device).float()
        old_logp = mb["old_logp"].to(self.device)
        ref_logp = mb["ref_logp"].to(self.device)
        advantages = mb["advantages"].to(self.device)    # [B]

        attn = (sequences != self.tokenizer.pad_token_id).long()
        attn = attn | response_mask.long()
        attn = attn.clamp(max=1)
        out = self.policy(input_ids=sequences, attention_mask=attn, use_cache=False)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        new_logp = _per_token_logp_from_logits(logits, sequences, response_mask)

        # ---- ratio ----
        log_ratio = (new_logp - old_logp) * response_mask
        ratio = log_ratio.exp()

        # 把 sequence-level advantage 广播到 token level
        adv_tok = advantages[:, None].expand_as(new_logp)
        unclipped = -adv_tok * ratio
        clipped = -adv_tok * ratio.clamp(1 - cfg.cliprange, 1 + cfg.cliprange)
        policy_loss = torch.maximum(unclipped, clipped) * response_mask
        denom = response_mask.sum().clamp(min=1.0)
        policy_loss = policy_loss.sum() / denom

        # ---- KL penalty ----
        kl = _kl_per_token(new_logp, ref_logp, response_mask, estimator=cfg.kl_estimator)
        kl_loss = kl.sum() / denom

        # ---- 总 loss ----
        loss = policy_loss + cfg.beta_kl * kl_loss

        # ---- diagnostics ----
        with torch.no_grad():
            approx_kl = (((ratio - 1) - log_ratio) * response_mask).sum() / denom
            clip_frac = (((ratio - 1).abs() > cfg.cliprange).float() * response_mask).sum() / denom

        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "approx_kl": approx_kl.detach(),
            "clip_frac": clip_frac.detach(),
        }

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _iter_minibatches(self, n: int, micro_bs: int):
        idx = torch.randperm(n)
        for i in range(0, n, micro_bs):
            yield idx[i:i + micro_bs].tolist()

    def _slice_buffer(self, buf: GRPOBuffer, indices) -> Dict[str, torch.Tensor]:
        return {
            "sequences":     buf.sequences[indices],
            "response_mask": buf.response_mask[indices],
            "old_logp":      buf.old_logp[indices],
            "ref_logp":      buf.ref_logp[indices],
            "advantages":    buf.advantages[indices],
        }

    def _infinite_loader(self, loader):
        while True:
            for b in loader:
                yield b

    def _save_ckpt(self) -> None:
        if not is_main_process():
            return
        unwrap = self.policy.module if hasattr(self.policy, "module") else self.policy
        save_dir = Path(self.config.output_dir) / f"checkpoint-iter-{self.iter_idx}"
        save_dir.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file
        save_file(
            {k: v.detach().contiguous().cpu() for k, v in unwrap.state_dict().items()},
            str(save_dir / "model.safetensors"),
            metadata={"format": "pt"},
        )
        from deepseek_v4.utils.io import safe_save_json
        safe_save_json(save_dir / "trainer_state.json", {
            "iter": self.iter_idx,
            "global_step": self.global_step,
        })
        logger.info(f"[GRPO] checkpoint saved to {save_dir}")
