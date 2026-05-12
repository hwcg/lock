"""
PPO Trainer。

主循环：
    for iter in range(num_iters):
        prompts = next batch from prompt dataset
        rollout = collect_rollouts(policy, ref, RM, prompts, kl_coef)
        for epoch in range(ppo_epochs):
            for mb in rollout.iter_minibatches(mini_bs):
                loss = compute_ppo_loss(mb)
                loss.backward(); optimizer.step()
        kl_controller.update(mean_kl)
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from deepseek_v4.distributed.utils import (
    all_reduce_mean, get_rank, get_world_size, is_distributed, is_main_process,
)
from deepseek_v4.inference.generation import GenerationConfig
from deepseek_v4.modeling.model import DeepseekV4ForCausalLM
from deepseek_v4.training.base_trainer import TrainerConfig
from deepseek_v4.training.checkpoint import CheckpointManager
from deepseek_v4.training.grad_checkpoint import enable_gradient_checkpointing
from deepseek_v4.training.optim import build_optimizer, build_scheduler
from deepseek_v4.training.ppo.kl_controller import (
    AdaptiveKLController, FixedKLController,
)
from deepseek_v4.training.ppo.rollout import RolloutBuffer, collect_rollouts
from deepseek_v4.training.ppo.value_head import PolicyValueModel
from deepseek_v4.training.reward_model import DeepseekV4RewardModel
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
class PPOConfig(TrainerConfig):
    """PPO 配置。"""

    # 数据（仅需 prompts）
    train_data_paths: List[str] = field(default_factory=list)   # jsonl with messages 或 prompt 字段
    prompt_field: str = "prompt"                                 # 或 "messages"
    max_prompt_len: int = 1024
    cache_dir: Optional[str] = "cache/datasets"

    # 模型
    model_config_path: str = "configs/model/mini_2b.json"
    init_from_checkpoint: str = "checkpoints/sft/checkpoint-final"
    reward_model_path: str = "checkpoints/reward_model/checkpoint-final"
    tokenizer_path: str = "checkpoints/tokenizer"
    share_value_backbone: bool = True

    # 生成
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    rollout_batch_size: int = 8           # 每 iter 采样 prompt 数（per device）
    gen_micro_batch_size: int = 4

    # PPO
    ppo_epochs: int = 4
    mini_batch_size: int = 2
    cliprange: float = 0.2                # 策略 ratio clip
    cliprange_value: float = 0.2          # value clip
    vf_coef: float = 0.1                  # value loss 权重
    entropy_coef: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    whiten_advantages: bool = True
    target_kl_for_early_stop: Optional[float] = 0.04   # None 关闭

    # KL 控制
    kl_controller: str = "adaptive"       # adaptive | fixed
    init_kl_coef: float = 0.2
    target_kl: float = 0.1
    kl_horizon: int = 10000

    # Reward shape
    reward_clip: Optional[float] = None   # 例如 5.0
    reward_normalize: bool = True

    # 训练循环
    num_iters: int = 1000
    save_iters: int = 50

    # 默认覆盖
    learning_rate: float = 1.0e-6
    weight_decay: float = 0.0
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False  # PPO 推理多，关 GC 加速


# ============================================================
# Prompt Dataset（最小实现：从 jsonl 取 prompt）
# ============================================================

from deepseek_v4.utils.io import read_jsonl
from torch.utils.data import Dataset as _D


class PromptDataset(_D):
    """从 jsonl 读 prompt（messages or 字符串），编码为 input_ids（不含 BOS 由 chat template 处理）。"""

    def __init__(self, paths: List[str], tokenizer, prompt_field: str = "prompt", max_prompt_len: int = 1024):
        self.tokenizer = tokenizer
        self.prompt_field = prompt_field
        self.max_prompt_len = max_prompt_len
        self.examples: List[List[int]] = []
        for p in paths:
            for row in read_jsonl(p):
                ids = self._encode_one(row)
                if ids:
                    self.examples.append(ids)
        logger.info(f"[PromptDataset] {len(self.examples)} prompts")

    def _encode_one(self, row: Dict[str, Any]) -> List[int]:
        if "messages" in row:
            text = self.tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True)
            ids = self.tokenizer.encode(text)
        elif self.prompt_field in row:
            ids = self.tokenizer.encode(str(row[self.prompt_field]))
        else:
            return []
        if len(ids) > self.max_prompt_len:
            ids = ids[-self.max_prompt_len:]
        return ids

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.examples[idx], dtype=torch.long)


def _prompt_collate(batch: List[torch.Tensor]) -> List[torch.Tensor]:
    """直接传 list of tensor（rollout 内部自己 padding）。"""
    return list(batch)


# ============================================================
# PPO Trainer
# ============================================================

class PPOTrainer:
    """
    自定义训练循环（不继承 BaseTrainer，因为 PPO 节奏完全不同）。
    """

    def __init__(
        self,
        config: PPOConfig,
        policy: DeepseekV4ForCausalLM,
        reward_model: DeepseekV4RewardModel,
        tokenizer,
        ref_model: Optional[DeepseekV4ForCausalLM] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.policy_lm = policy
        self.reward_model = reward_model
        self.ref_model = ref_model

        # 状态
        self.iter_idx = 0
        self.global_step = 0
        self.start_time = 0.0
        self.metric_logger = MetricLogger(window=20)
        self.tracker_logger: Optional[MultiLogger] = None
        self.timer = Stopwatch()

        # 后续 setup 时初始化
        self.pv_model: Optional[PolicyValueModel] = None
        self.optimizer = None
        self.scheduler = None
        self.kl_ctrl = None
        self.train_loader: Optional[DataLoader] = None
        self.ckpt_mgr: Optional[CheckpointManager] = None
        self.device: Optional[torch.device] = None
        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0

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
            logger.info(f"  PPOTrainer  world_size={get_world_size()}  device={self.device}")
            logger.info(f"  num_iters={self.config.num_iters}  rollout_bs={self.config.rollout_batch_size}")
            logger.info(f"  ppo_epochs={self.config.ppo_epochs}  cliprange={self.config.cliprange}")
            logger.info(f"  kl_ctrl={self.config.kl_controller}  init_kl={self.config.init_kl_coef}")
            logger.info("=" * 70)

        set_seed(self.config.seed + get_rank())

        # ----- 模型 -----
        # ref：默认深拷贝 policy
        if self.ref_model is None:
            logger.info("[PPO] making frozen reference (deep copy of policy)")
            self.ref_model = copy.deepcopy(self.policy_lm)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval().to(self.device)

        # RM 冻结
        for p in self.reward_model.parameters():
            p.requires_grad = False
        self.reward_model.eval().to(self.device)

        # PolicyValueModel
        self.pv_model = PolicyValueModel(
            policy_lm=self.policy_lm,
            share_backbone=self.config.share_value_backbone,
        ).to(self.device)
        if self.config.gradient_checkpointing:
            enable_gradient_checkpointing(self.pv_model.policy)
            if not self.config.share_value_backbone:
                enable_gradient_checkpointing(self.pv_model.value_backbone, layer_attr="layers")

        # DDP 包装
        if is_distributed():
            self.pv_model = nn.parallel.DistributedDataParallel(
                self.pv_model,
                device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )

        # ----- 优化器 -----
        self.optimizer = build_optimizer(
            self.pv_model.module if hasattr(self.pv_model, "module") else self.pv_model,
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

        # ----- KL 控制器 -----
        if self.config.kl_controller == "adaptive":
            self.kl_ctrl = AdaptiveKLController(
                init_kl_coef=self.config.init_kl_coef,
                target_kl=self.config.target_kl,
                horizon=self.config.kl_horizon,
            )
        else:
            self.kl_ctrl = FixedKLController(kl_coef=self.config.init_kl_coef)

        # ----- 数据 -----
        ds = PromptDataset(
            paths=self.config.train_data_paths,
            tokenizer=self.tokenizer,
            prompt_field=self.config.prompt_field,
            max_prompt_len=self.config.max_prompt_len,
        )
        self.train_loader = DataLoader(
            ds,
            batch_size=self.config.rollout_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=_prompt_collate,
        )

        # ----- Checkpoint -----
        self.ckpt_mgr = CheckpointManager(
            output_dir=self.config.output_dir,
            keep_last=self.config.keep_last,
            keep_best=self.config.keep_best,
            metric_name=self.config.metric_name,
            metric_mode="max",       # PPO 通常 maximize reward
        )

        # ----- Logger -----
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
    # Train Loop
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

    def _train_one_iter(self, prompt_iter) -> None:
        cfg = self.config
        # ----- 1. 收集 prompts -----
        prompts: List[torch.Tensor] = next(prompt_iter)

        # ----- 2. Rollout -----
        with self.timer.track("rollout"):
            self.pv_model.eval()
            gen_cfg = GenerationConfig(
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                return_log_probs=True,
            )
            buffer = collect_rollouts(
                policy_value_model=self.pv_model.module if hasattr(self.pv_model, "module") else self.pv_model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                prompts=prompts,
                pad_token_id=self.tokenizer.pad_token_id,
                gen_config=gen_cfg,
                kl_coef=self.kl_ctrl.value,
                gamma=cfg.gamma,
                lam=cfg.gae_lambda,
                whiten_advantages=cfg.whiten_advantages,
                device=self.device,
                micro_batch_size=cfg.gen_micro_batch_size,
            )

        # ----- 2.5 Reward shape -----
        if cfg.reward_normalize:
            self._update_reward_running(buffer.rm_scores)
            buffer.rm_scores = self._normalize_reward(buffer.rm_scores)
            # 重新加权到 rewards
            # （这里简化：仅记录原始；新 rm 在下一 iter 生效）
        if cfg.reward_clip is not None:
            buffer.rewards.clamp_(min=-cfg.reward_clip, max=cfg.reward_clip)

        # ----- 3. PPO 更新 -----
        approx_kl_total = 0.0
        approx_kl_count = 0
        log_metrics: Dict[str, float] = {
            "iter/rm_score_mean": float(buffer.rm_scores.mean()),
            "iter/rm_score_std":  float(buffer.rm_scores.std().clamp(min=1e-8)),
            "iter/response_len":  float(buffer.response_mask.sum(dim=1).float().mean()),
            "iter/kl_coef":       float(self.kl_ctrl.value),
        }

        self.pv_model.train()
        for epoch in range(cfg.ppo_epochs):
            for mb in buffer.iter_minibatches(
                mini_bs=cfg.mini_batch_size, shuffle=True, device=self.device,
            ):
                with self.timer.track("forward"):
                    loss_dict = self._compute_ppo_loss(mb)

                with self.timer.track("backward"):
                    self.optimizer.zero_grad(set_to_none=True)
                    loss_dict["loss"].backward()

                with self.timer.track("optim"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.pv_model.parameters(), max_norm=cfg.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()

                self.global_step += 1
                approx_kl_total += float(loss_dict["approx_kl"])
                approx_kl_count += 1

                # logging
                step_metrics = {
                    "loss":         float(loss_dict["loss"]),
                    "policy_loss":  float(loss_dict["policy_loss"]),
                    "value_loss":   float(loss_dict["value_loss"]),
                    "entropy":      float(loss_dict["entropy"]),
                    "approx_kl":    float(loss_dict["approx_kl"]),
                    "clip_frac":    float(loss_dict["clip_frac"]),
                    "lr":           float(self.scheduler.get_last_lr()[0]),
                    "grad_norm":    float(grad_norm),
                }
                self.metric_logger.update(**step_metrics)

                # early stop
                if cfg.target_kl_for_early_stop is not None and \
                   step_metrics["approx_kl"] > cfg.target_kl_for_early_stop * 1.5:
                    if is_main_process():
                        logger.warning(
                            f"[PPO] early-stop at iter={self.iter_idx} epoch={epoch} "
                            f"approx_kl={step_metrics['approx_kl']:.4f}"
                        )
                    break

        # ----- 4. 更新 KL 控制器 -----
        if approx_kl_count > 0:
            mean_kl = approx_kl_total / approx_kl_count
            self.kl_ctrl.update(current_kl=mean_kl, n_steps=approx_kl_count)

        # ----- 5. Logging -----
        if is_main_process():
            elapsed = time.time() - self.start_time
            agg = self.metric_logger.items()
            log_metrics.update({
                "loss":         agg.get("loss", 0),
                "policy_loss":  agg.get("policy_loss", 0),
                "value_loss":   agg.get("value_loss", 0),
                "approx_kl":    agg.get("approx_kl", 0),
                "lr":           agg.get("lr", 0),
            })
            logger.info(
                f"[PPO] iter={self.iter_idx} step={self.global_step}  "
                f"rm={log_metrics['iter/rm_score_mean']:.3f}±{log_metrics['iter/rm_score_std']:.3f}  "
                f"len={log_metrics['iter/response_len']:.0f}  "
                f"kl={log_metrics['approx_kl']:.4f}  "
                f"kl_coef={log_metrics['iter/kl_coef']:.3f}  "
                f"loss={log_metrics['loss']:.3f}  "
                f"lr={log_metrics['lr']:.2e}  "
                f"elapsed={format_time(elapsed)}"
            )
            if self.tracker_logger is not None:
                self.tracker_logger.log(log_metrics, step=self.iter_idx)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_ppo_loss(self, mb: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cfg = self.config
        prompt_ids = mb["prompt_ids"]
        response_ids = mb["response_ids"]
        response_mask = mb["response_mask"].float()
        old_logp = mb["old_logprobs"]
        values_old = mb["values"]
        advantages = mb["advantages"]
        returns = mb["returns"]

        # 拼成完整序列
        sequences = torch.cat([prompt_ids, response_ids], dim=1)
        seq_attn = torch.ones_like(sequences)
        seq_attn[:, :prompt_ids.shape[1]] = (prompt_ids != self.tokenizer.pad_token_id).long()

        # full response mask (with prompt 区为 0)
        full_resp_mask = torch.zeros_like(sequences)
        full_resp_mask[:, prompt_ids.shape[1]:] = mb["response_mask"]

        # 前向：得到 logits + values
        out = self.pv_model(
            input_ids=sequences, attention_mask=seq_attn,
            return_logits=True, return_values=True, use_cache=False,
        )
        logits = out["logits"]
        values_full = out["values"]
        # shift 对齐
        shift_logits = logits[:, :-1, :]
        shift_labels = sequences[:, 1:]
        shift_mask = full_resp_mask[:, 1:].float()
        # 提取 response 部分
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        new_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        # 仅 response 区
        P = prompt_ids.shape[1]
        T = response_ids.shape[1]
        new_logp_resp = new_logp[:, P - 1:P - 1 + T] * response_mask
        new_values_resp = values_full[:, :-1][:, P - 1:P - 1 + T] * response_mask

        # ---- ratio + clipped policy loss ----
        # old_logp 已含 response_mask（采样时无效位为 0），但为安全再 mask 一次
        old_logp_masked = old_logp * response_mask
        log_ratio = (new_logp_resp - old_logp_masked) * response_mask
        ratio = log_ratio.exp()

        # 仅在 mask=1 处计算
        denom = response_mask.sum().clamp(min=1.0)

        unclipped = -advantages * ratio
        clipped = -advantages * ratio.clamp(1 - cfg.cliprange, 1 + cfg.cliprange)
        policy_loss = torch.maximum(unclipped, clipped) * response_mask
        policy_loss = policy_loss.sum() / denom

        # ---- value loss (clipped) ----
        v_clipped = values_old + (new_values_resp - values_old).clamp(
            -cfg.cliprange_value, cfg.cliprange_value,
        )
        v_loss_un = (new_values_resp - returns) ** 2
        v_loss_cl = (v_clipped - returns) ** 2
        value_loss = 0.5 * torch.maximum(v_loss_un, v_loss_cl) * response_mask
        value_loss = value_loss.sum() / denom

        # ---- entropy（鼓励探索） ----
        # H = -Σ p log p ；近似用 -log_softmax(label) 不准，这里精确算（仅 response 区）
        with torch.no_grad():
            entropy_full = -(log_probs.exp() * log_probs).sum(dim=-1)  # [B, S-1]
        entropy = (entropy_full[:, P - 1:P - 1 + T] * response_mask).sum() / denom

        # ---- 总 loss ----
        loss = policy_loss + cfg.vf_coef * value_loss - cfg.entropy_coef * entropy

        # ---- diagnostics ----
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio) * response_mask
            approx_kl = approx_kl.sum() / denom
            clipped_frac = (
                (((ratio - 1).abs() > cfg.cliprange).float() * response_mask)
                .sum() / denom
            )

        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "approx_kl": approx_kl.detach(),
            "clip_frac": clipped_frac.detach(),
        }

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _infinite_loader(self, loader):
        while True:
            for b in loader:
                yield b

    def _update_reward_running(self, rewards: torch.Tensor) -> None:
        """Welford running mean/var。"""
        for x in rewards.detach().cpu().tolist():
            self._reward_count += 1
            delta = x - self._reward_running_mean
            self._reward_running_mean += delta / self._reward_count
            delta2 = x - self._reward_running_mean
            self._reward_running_var += delta * delta2

    def _normalize_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        if self._reward_count <= 1:
            return rewards
        var = self._reward_running_var / max(self._reward_count - 1, 1)
        std = max(var ** 0.5, 1e-6)
        return (rewards - self._reward_running_mean) / std

    def _save_ckpt(self) -> None:
        if not is_main_process():
            return
        unwrap = self.pv_model.module if hasattr(self.pv_model, "module") else self.pv_model
        # 保存 policy（不带 value head，便于后续 SFT/DPO 继续）
        policy_lm = unwrap.policy
        from deepseek_v4.utils.io import safe_save_json
        save_dir = Path(self.config.output_dir) / f"checkpoint-iter-{self.iter_idx}"
        save_dir.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file
        save_file(
            {k: v.detach().contiguous().cpu() for k, v in policy_lm.state_dict().items()},
            str(save_dir / "model.safetensors"),
            metadata={"format": "pt"},
        )
        # 把 value head 也保存（可选恢复）
        save_file(
            {k: v.detach().contiguous().cpu() for k, v in unwrap.value_head.state_dict().items()},
            str(save_dir / "value_head.safetensors"),
            metadata={"format": "pt"},
        )
        safe_save_json(save_dir / "trainer_state.json", {
            "iter": self.iter_idx,
            "global_step": self.global_step,
            "kl_coef": float(self.kl_ctrl.value),
        })
        logger.info(f"[PPO] checkpoint saved to {save_dir}")
