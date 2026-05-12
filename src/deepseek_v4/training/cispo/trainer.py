"""
CISPO (Clipped IS-weight Policy Optimization) Trainer。

参考：MiniMax M1 报告（2024）。

核心思想：
- PPO/GRPO 在 ratio = π/π_old 上做 clip，但当一个 token 的 ratio 偏大时整个 step 的梯度被"截断"
- CISPO 改为对重要性权重 IS = ratio 本身做"截断"（不参与梯度）
- 损失：L = -E_t[ stop_grad(min(ratio, c)) · A · log π_θ(a_t|s_t) ]
       其中 c = (1+ε_hi)，min 也可在两侧 clip
- 优势：所有 token 都贡献梯度，避免 PPO 中"重要更新的 token 被 clip 掉"

实现要点：
- ratio 用 detach 包装，仅做权重，不传梯度
- 实际梯度来自 log π 的一阶项
- 与 PPO 一样 token-level loss，token-level advantage 由 sequence-level reward 广播或带 GAE
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
    get_rank, get_world_size, is_distributed, is_main_process,
)
from deepseek_v4.inference.generation import GenerationConfig, generate
from deepseek_v4.modeling.model import DeepseekV4ForCausalLM
from deepseek_v4.training.base_trainer import TrainerConfig
from deepseek_v4.training.checkpoint import CheckpointManager
from deepseek_v4.training.grad_checkpoint import enable_gradient_checkpointing
from deepseek_v4.training.optim import build_optimizer, build_scheduler
from deepseek_v4.training.ppo.rollout import PromptDataset
from deepseek_v4.training.rewards.base import RewardFunction
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
class CISPOConfig(TrainerConfig):
    """CISPO 配置。"""

    # 数据
    train_data_paths: List[str] = field(default_factory=list)
    prompt_field: str = "messages"
    reference_field: Optional[str] = "answer"
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
    group_size: int = 8
    prompts_per_step: int = 4
    gen_micro_batch_size: int = 8

    # CISPO 损失
    is_clip_high: float = 1.5            # 上限（关键：MiniMax 论文用 1.0~3.0）
    is_clip_low: float = 0.0             # 下限（一般不 clip 下界 = 0）
    beta_kl: float = 0.0                  # 可选 KL（通常 0）
    advantage_eps: float = 1e-8
    ppo_epochs: int = 1
    use_group_baseline: bool = True       # True: 组内 reward 标准化作 token-level advantage
    kl_estimator: str = "k3"

    # 训练循环
    num_iters: int = 1000
    save_iters: int = 50
    rewards: List[Dict[str, Any]] = field(default_factory=list)

    # 默认覆盖
    learning_rate: float = 1.0e-6
    weight_decay: float = 0.0
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True


# ============================================================
# Trainer
# ============================================================

class CISPOTrainer:
    """CISPO 训练器（与 GRPO 结构相似，仅 loss 不同）。"""

    def __init__(
        self,
        config: CISPOConfig,
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
        self.device: Optional[torch.device] = None
        self._dataset = None
        self._raw_prompt_texts: List[str] = []
        self._references: List[Any] = []

    # ----- Setup -----

    def setup(self) -> None:
        # 与 GRPO 相同的 setup（只是 trainer 名字不同），直接复用 GRPO 的 setup 实现
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
            logger.info(f"  CISPOTrainer  world_size={get_world_size()}")
            logger.info(f"  group_size={self.config.group_size}  prompts_per_step={self.config.prompts_per_step}")
            logger.info(f"  is_clip=[{self.config.is_clip_low}, {self.config.is_clip_high}]")
            logger.info(f"  beta_kl={self.config.beta_kl}")
            logger.info("=" * 70)

        set_seed(self.config.seed + get_rank())

        # ref
        if self.ref_model is None:
            logger.info("[CISPO] making frozen reference (deep copy)")
            self.ref_model = copy.deepcopy(self.policy)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval().to(self.device)

        # policy
        self.policy = self.policy.to(self.device)
        if self.config.gradient_checkpointing:
            enable_gradient_checkpointing(self.policy)

        if is_distributed():
            self.policy = nn.parallel.DistributedDataParallel(
                self.policy,
                device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

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
        from deepseek_v4.utils.io import read_jsonl
        if self.config.reference_field:
            for p in self.config.train_data_paths:
                for row in read_jsonl(p):
                    self._references.append(row.get(self.config.reference_field))
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
        self._dataset = ds
        self.train_loader = DataLoader(
            range(len(ds)),
            batch_size=self.config.prompts_per_step,
            shuffle=True, drop_last=True,
            num_workers=0,
            collate_fn=lambda b: list(b),
        )

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

    # ----- Train Loop -----

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

    def _train_one_iter(self, prompt_iter):
        cfg = self.config
        indices: List[int] = next(prompt_iter)

        # ----- Rollout（复用 GRPO 的逻辑） -----
        from deepseek_v4.training.grpo.trainer import (
            GRPOTrainer, _per_token_logp_from_logits, _kl_per_token,
        )
        with self.timer.track("rollout"):
            # 临时借用 GRPO 的 collect 函数：构造一个临时 GRPOConfig
            grpo_like = type("X", (), {
                "group_size": cfg.group_size,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "gen_micro_batch_size": cfg.gen_micro_batch_size,
                "advantage_eps": cfg.advantage_eps,
            })()
            buffer = self._collect_rollouts(indices, grpo_like)
        if buffer is None or len(buffer["sequences"]) == 0:
            return

        # Logging
        rewards_raw = buffer["rewards_raw"]
        log_metrics = {
            "rm/mean": float(rewards_raw.mean()),
            "rm/std":  float(rewards_raw.std().clamp(min=1e-8)),
            "rm/min":  float(rewards_raw.min()),
            "rm/max":  float(rewards_raw.max()),
            "resp_len": float(buffer["response_mask"].sum(dim=1).float().mean()),
            "advantage_abs_mean": float(buffer["advantages"].abs().mean()),
        }

        agg = {}
        for epoch in range(cfg.ppo_epochs):
            for mb_idx in self._iter_minibatches(len(buffer["sequences"]), cfg.micro_batch_size):
                mb = {k: v[mb_idx] for k, v in buffer.items()}
                with self.timer.track("forward"):
                    loss_dict = self._compute_cispo_loss(mb)
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
                for k in ("loss", "policy_loss", "kl_loss", "approx_kl",
                          "is_clip_frac_high", "is_clip_frac_low",
                          "is_weight_mean", "is_weight_max"):
                    agg.setdefault(k, []).append(float(loss_dict[k]))
                agg.setdefault("lr", []).append(float(self.scheduler.get_last_lr()[0]))
                agg.setdefault("grad_norm", []).append(float(gn))

        for k, v in agg.items():
            avg = sum(v) / len(v)
            log_metrics[k] = avg
            self.metric_logger.update(**{k: avg})

        if is_main_process():
            elapsed = time.time() - self.start_time
            logger.info(
                f"[CISPO] iter={self.iter_idx} step={self.global_step}  "
                f"rm={log_metrics['rm/mean']:+.3f}±{log_metrics['rm/std']:.3f}  "
                f"adv={log_metrics['advantage_abs_mean']:.3f}  "
                f"loss={log_metrics['loss']:.3f}  "
                f"kl={log_metrics['approx_kl']:.4f}  "
                f"is_hi={log_metrics['is_clip_frac_high']:.3f}  "
                f"is_w={log_metrics['is_weight_mean']:.2f}/{log_metrics['is_weight_max']:.2f}  "
                f"lr={log_metrics['lr']:.2e}  elapsed={format_time(elapsed)}"
            )
            if self.tracker_logger is not None:
                self.tracker_logger.log(log_metrics, step=self.iter_idx)

    # ----- Rollout：复用 GRPO 的核心 -----

    @torch.no_grad()
    def _collect_rollouts(self, indices, grpo_like):
        cfg = self.config
        pad_id = self.tokenizer.pad_token_id

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

        all_sequences, all_resp_mask, all_old_logp = [], [], []
        gen_cfg = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens, do_sample=True,
            temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
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
            prompt_ids = torch.full((len(batch_prompts), max_p), pad_id,
                                    dtype=torch.long, device=self.device)
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
            old_logp_full = torch.zeros_like(seq, dtype=torch.float32)
            T = resp_mask.shape[1]
            old_logp_full[:, max_p:max_p + T] = old_logp_resp
            all_sequences.append(seq.cpu())
            all_resp_mask.append(full_resp_mask.cpu())
            all_old_logp.append(old_logp_full.cpu())

        def _pad_cat(tensors, pv, dt=None):
            ml = max(t.shape[1] for t in tensors)
            out = []
            for t in tensors:
                if t.shape[1] < ml:
                    pad = torch.full((t.shape[0], ml - t.shape[1]), pv, dtype=t.dtype)
                    t = torch.cat([t, pad], dim=1)
                out.append(t)
            return torch.cat(out, dim=0)

        sequences = _pad_cat(all_sequences, pad_id).to(self.device)
        response_mask = _pad_cat(all_resp_mask, 0).to(self.device)
        old_logp = _pad_cat(all_old_logp, 0.0).to(self.device).float()

        # ref logp
        from deepseek_v4.training.grpo.trainer import _per_token_logp_from_logits
        ref_logp = torch.zeros_like(old_logp)
        for start in range(0, sequences.shape[0], cfg.gen_micro_batch_size):
            seq_mb = sequences[start:start + cfg.gen_micro_batch_size]
            mask_mb = response_mask[start:start + cfg.gen_micro_batch_size]
            attn = (seq_mb != pad_id).long() | mask_mb
            attn = attn.clamp(max=1)
            out = self.ref_model(input_ids=seq_mb, attention_mask=attn, use_cache=False)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            per = _per_token_logp_from_logits(logits, seq_mb, mask_mb)
            ref_logp[start:start + cfg.gen_micro_batch_size] = per.float()

        # reward
        completions: List[str] = []
        for i in range(sequences.shape[0]):
            pos = response_mask[i].nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                completions.append("")
                continue
            start_idx = int(pos[0].item())
            end_idx = int(pos[-1].item()) + 1
            text = self.tokenizer.decode(sequences[i, start_idx:end_idx].tolist(),
                                         skip_special_tokens=False)
            completions.append(text)
        rewards_raw = torch.tensor(
            self.reward_fn(completions=completions, references=references, prompts=prompt_texts),
            dtype=torch.float32, device=self.device,
        )

        # advantage：组内标准化（与 GRPO 一致）
        group_ids_t = torch.tensor(group_ids, dtype=torch.long, device=self.device)
        advantages = torch.zeros_like(rewards_raw)
        n_groups = group_ids_t.max().item() + 1
        for g in range(n_groups):
            mask = (group_ids_t == g)
            if mask.sum() <= 1:
                advantages[mask] = 0.0
                continue
            rg = rewards_raw[mask]
            advantages[mask] = (rg - rg.mean()) / rg.std().clamp(min=cfg.advantage_eps)

        return {
            "sequences": sequences.cpu(),
            "response_mask": response_mask.cpu(),
            "old_logp": old_logp.cpu(),
            "ref_logp": ref_logp.cpu(),
            "advantages": advantages.cpu(),
            "rewards_raw": rewards_raw.cpu(),
        }

    # ----- Loss -----

    def _compute_cispo_loss(self, mb: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cfg = self.config
        sequences = mb["sequences"].to(self.device)
        response_mask = mb["response_mask"].to(self.device).float()
        old_logp = mb["old_logp"].to(self.device)
        ref_logp = mb["ref_logp"].to(self.device)
        advantages = mb["advantages"].to(self.device)   # [B]

        attn = (sequences != self.tokenizer.pad_token_id).long() | response_mask.long()
        attn = attn.clamp(max=1)
        out = self.policy(input_ids=sequences, attention_mask=attn, use_cache=False)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        from deepseek_v4.training.grpo.trainer import (
            _per_token_logp_from_logits, _kl_per_token,
        )
        new_logp = _per_token_logp_from_logits(logits, sequences, response_mask)

        # ---- IS weight (detach) ----
        log_ratio = (new_logp - old_logp) * response_mask
        ratio = log_ratio.exp()
        is_weight = ratio.clamp(min=cfg.is_clip_low, max=cfg.is_clip_high).detach()

        # ---- 损失 ----
        # L = - E[ is_weight · A · log π(a|s) ]
        adv_tok = advantages[:, None].expand_as(new_logp)
        # 注意：log π = new_logp（仅 response 区有效）
        per_token_obj = is_weight * adv_tok * new_logp * response_mask
        denom = response_mask.sum().clamp(min=1.0)
        policy_loss = -per_token_obj.sum() / denom

        # ---- KL（可选） ----
        if cfg.beta_kl > 0:
            kl = _kl_per_token(new_logp, ref_logp, response_mask, estimator=cfg.kl_estimator)
            kl_loss = kl.sum() / denom
        else:
            kl_loss = torch.tensor(0.0, device=self.device)

        loss = policy_loss + cfg.beta_kl * kl_loss

        with torch.no_grad():
            approx_kl = (((ratio - 1) - log_ratio) * response_mask).sum() / denom
            is_clip_frac_high = ((ratio > cfg.is_clip_high).float() * response_mask).sum() / denom
            is_clip_frac_low = ((ratio < cfg.is_clip_low).float() * response_mask).sum() / denom \
                if cfg.is_clip_low > 0 else torch.tensor(0.0, device=self.device)
            valid_mask = response_mask.bool()
            is_w_valid = is_weight[valid_mask]
            is_w_mean = is_w_valid.mean() if is_w_valid.numel() > 0 else torch.tensor(0.0, device=self.device)
            is_w_max = is_w_valid.max() if is_w_valid.numel() > 0 else torch.tensor(0.0, device=self.device)

        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "approx_kl": approx_kl.detach(),
            "is_clip_frac_high": is_clip_frac_high.detach(),
            "is_clip_frac_low": is_clip_frac_low.detach(),
            "is_weight_mean": is_w_mean.detach(),
            "is_weight_max": is_w_max.detach(),
        }

    # ----- 工具 -----

    def _iter_minibatches(self, n: int, micro_bs: int):
        idx = torch.randperm(n)
        for i in range(0, n, micro_bs):
            yield idx[i:i + micro_bs].tolist()

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
        logger.info(f"[CISPO] checkpoint saved to {save_dir}")
