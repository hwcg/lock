"""
Agentic RL Trainer。

基于 GRPO 框架，主要差异：
- rollout 阶段调用 ToolEnvironment 而非简单 generate
- reward 函数接受 (completions, trajectories) 双输入
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepseek_v4.distributed.utils import (
    get_rank, get_world_size, is_distributed, is_main_process,
)
from deepseek_v4.inference.generation import GenerationConfig
from deepseek_v4.modeling.model import DeepseekV4ForCausalLM
from deepseek_v4.training.agentic_rl.environment import ToolEnvironment
from deepseek_v4.training.agentic_rl.trajectory import collect_trajectories
from deepseek_v4.training.base_trainer import TrainerConfig
from deepseek_v4.training.checkpoint import CheckpointManager
from deepseek_v4.training.grad_checkpoint import enable_gradient_checkpointing
from deepseek_v4.training.grpo.trainer import (
    _kl_per_token, _per_token_logp_from_logits,
)
from deepseek_v4.training.optim import build_optimizer, build_scheduler
from deepseek_v4.training.tool_use.tools import ToolRegistry, register_builtin_tools
from deepseek_v4.utils.io import read_jsonl
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
class AgenticRLConfig(TrainerConfig):
    """Agentic RL 配置。"""
    # 数据：jsonl 含 messages 与可选 answer/expected_tool_calls
    train_data_paths: List[str] = field(default_factory=list)
    reference_field: Optional[str] = "answer"
    max_prompt_len: int = 1024
    max_seq_len: int = 4096

    # 模型
    model_config_path: str = "configs/model/mini_2b.json"
    init_from_checkpoint: str = "checkpoints/sft/checkpoint-final"
    tokenizer_path: str = "checkpoints/tokenizer"

    # 环境
    max_turns: int = 5
    final_answer_tool: Optional[str] = None

    # 生成
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    group_size: int = 4
    prompts_per_step: int = 4

    # GRPO-like
    cliprange: float = 0.2
    beta_kl: float = 0.02
    ppo_epochs: int = 1
    advantage_eps: float = 1e-8
    kl_estimator: str = "k3"

    # Reward 权重（多目标）
    reward_correctness_weight: float = 1.0     # 最终答案正确性
    reward_format_weight: float = 0.1          # tool call 格式
    reward_efficiency_weight: float = 0.1      # 步数惩罚
    max_efficient_turns: int = 3

    # 训练循环
    num_iters: int = 1000
    save_iters: int = 50

    learning_rate: float = 5.0e-7
    weight_decay: float = 0.0
    micro_batch_size: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True


# ============================================================
# Trainer
# ============================================================

class AgenticRLTrainer:
    """多轮工具调用 RL（GRPO 风格）。"""

    def __init__(
        self,
        config: AgenticRLConfig,
        policy: DeepseekV4ForCausalLM,
        tokenizer,
        tool_registry: Optional[ToolRegistry] = None,
        answer_reward_fn: Optional[Callable] = None,
        ref_model: Optional[DeepseekV4ForCausalLM] = None,
    ):
        self.config = config
        self.policy = policy
        self.tokenizer = tokenizer
        self.tool_registry = tool_registry or register_builtin_tools()
        self.answer_reward_fn = answer_reward_fn   # (completions, references) -> List[float]
        self.ref_model = ref_model

        self.iter_idx = 0
        self.global_step = 0
        self.start_time = 0.0
        self.metric_logger = MetricLogger(window=20)
        self.timer = Stopwatch()
        self.tracker_logger: Optional[MultiLogger] = None
        self.device: Optional[torch.device] = None
        self.env: Optional[ToolEnvironment] = None

    # ----- Setup -----

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
            logger.info(f"  AgenticRLTrainer  world_size={get_world_size()}")
            logger.info(f"  max_turns={self.config.max_turns}")
            logger.info(f"  group_size={self.config.group_size}")
            logger.info(f"  tools: {list(self.tool_registry.tools.keys())}")
            logger.info("=" * 70)

        set_seed(self.config.seed + get_rank())

        # ref
        if self.ref_model is None:
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
                find_unused_parameters=False, broadcast_buffers=False,
            )

        self.optimizer = build_optimizer(
            self.policy.module if hasattr(self.policy, "module") else self.policy,
            name=self.config.optimizer, lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
        )
        self.scheduler = build_scheduler(
            self.optimizer, name=self.config.scheduler,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.num_iters * self.config.ppo_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

        # Env
        self.env = ToolEnvironment(
            tool_registry=self.tool_registry,
            max_turns=self.config.max_turns,
            final_answer_tool=self.config.final_answer_tool,
        )

        # 数据：list of (initial_messages, reference)
        self.tasks: List[Dict[str, Any]] = []
        for p in self.config.train_data_paths:
            for row in read_jsonl(p):
                msgs = row.get("messages")
                if not msgs:
                    continue
                # 注入 tools
                tools = self.tool_registry.to_openai_tools()
                for m in msgs:
                    if m.get("role") == "system":
                        m["tools"] = tools
                        break
                else:
                    msgs = [{"role": "system", "content": "", "tools": tools}] + msgs
                # 把最后一条 assistant 去掉（如果有），保留 prompt 部分
                while msgs and msgs[-1].get("role") == "assistant":
                    msgs.pop()
                self.tasks.append({
                    "messages": msgs,
                    "reference": row.get(self.config.reference_field),
                })
        logger.info(f"[AgenticRL] loaded {len(self.tasks)} tasks")

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
        rng = torch.Generator().manual_seed(cfg.seed)

        for self.iter_idx in range(cfg.num_iters):
            # 随机抽 prompts_per_step 个 task
            idx = torch.randint(0, len(self.tasks), (cfg.prompts_per_step,), generator=rng).tolist()
            self._train_one_iter(idx)

            PAUSE_CONTROLLER.wait_if_paused()
            if PAUSE_CONTROLLER.should_save_and_exit():
                self._save_ckpt(); break
            if cfg.save_iters > 0 and (self.iter_idx + 1) % cfg.save_iters == 0:
                self._save_ckpt()

        self._save_ckpt()
        if self.tracker_logger is not None:
            self.tracker_logger.finish()

    # ----- 一 iter -----

    def _train_one_iter(self, task_indices: List[int]):
        cfg = self.config
        # 每个 task 复制 group_size 份
        initial_messages_list: List[List[Dict[str, Any]]] = []
        references: List[Any] = []
        group_ids: List[int] = []
        for g, idx in enumerate(task_indices):
            t = self.tasks[idx]
            for _ in range(cfg.group_size):
                initial_messages_list.append(copy.deepcopy(t["messages"]))
                references.append(t["reference"])
                group_ids.append(g)

        # 定义 reward 函数：correctness + format + efficiency
        def _reward_fn(completions, trajectories):
            return self._compute_reward(completions, trajectories, references)

        # rollout
        gen_cfg = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens, do_sample=True,
            temperature=cfg.temperature, top_p=cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        policy_eval = self.policy.module if hasattr(self.policy, "module") else self.policy
        policy_eval.eval()
        with self.timer.track("rollout"):
            buf = collect_trajectories(
                model=policy_eval,
                ref_model=self.ref_model,
                tokenizer=self.tokenizer,
                initial_messages_list=initial_messages_list,
                environment=self.env,
                generation_config=gen_cfg,
                device=self.device,
                reward_fn=_reward_fn,
                max_seq_len=cfg.max_seq_len,
            )

        rewards = buf["rewards"].to(self.device)
        group_ids_t = torch.tensor(group_ids, dtype=torch.long, device=self.device)
        # 组内标准化
        advantages = torch.zeros_like(rewards)
        n_groups = group_ids_t.max().item() + 1
        for g in range(n_groups):
            mask = (group_ids_t == g)
            if mask.sum() <= 1:
                continue
            rg = rewards[mask]
            advantages[mask] = (rg - rg.mean()) / rg.std().clamp(min=cfg.advantage_eps)

        buf["advantages"] = advantages.cpu()

        # optimize
        sizes = buf["sequences"].shape
        for epoch in range(cfg.ppo_epochs):
            for mb_idx in self._iter_minibatches(sizes[0], cfg.micro_batch_size):
                mb = {k: v[mb_idx] for k, v in buf.items() if isinstance(v, torch.Tensor)}
                self.policy.train()
                with self.timer.track("forward"):
                    loss_dict = self._compute_loss(mb)
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

                self.metric_logger.update(
                    loss=float(loss_dict["loss"]),
                    policy_loss=float(loss_dict["policy_loss"]),
                    kl_loss=float(loss_dict["kl_loss"]),
                    approx_kl=float(loss_dict["approx_kl"]),
                    lr=float(self.scheduler.get_last_lr()[0]),
                    grad_norm=float(gn),
                )

        if is_main_process():
            avg_turns = sum(len(t.steps) for t in buf["trajectories"]) / max(len(buf["trajectories"]), 1)
            success_rate = float(buf["success"].float().mean())
            metrics = self.metric_logger.items()
            logger.info(
                f"[AgenticRL] iter={self.iter_idx} step={self.global_step}  "
                f"reward={rewards.mean().item():+.3f}  "
                f"success={success_rate:.2f}  "
                f"avg_turns={avg_turns:.2f}  "
                f"loss={metrics.get('loss', 0):.3f}  "
                f"kl={metrics.get('approx_kl', 0):.4f}  "
                f"elapsed={format_time(time.time() - self.start_time)}"
            )
            if self.tracker_logger is not None:
                self.tracker_logger.log({
                    "reward/mean": rewards.mean().item(),
                    "reward/std":  rewards.std().clamp(min=1e-8).item(),
                    "success_rate": success_rate,
                    "avg_turns": avg_turns,
                    **{k: v for k, v in metrics.items()},
                }, step=self.iter_idx)

    # ----- Reward 计算 -----

    def _compute_reward(
        self,
        completions: List[str],
        trajectories: List[Any],
        references: List[Any],
    ) -> List[float]:
        cfg = self.config
        # 1. correctness
        if self.answer_reward_fn is not None:
            correctness = self.answer_reward_fn(
                completions=completions,
                references=references,
            )
        else:
            from deepseek_v4.training.rewards import math_correctness_reward
            fn = math_correctness_reward()
            try:
                correctness = fn(completions=completions, references=references)
            except Exception:
                correctness = [0.0] * len(completions)

        # 2. format：每步 tool_calls 都成功解析了？trajectory 结构本身就保证（除非 parse 失败）
        # 简单：每条 trajectory 看 tool_calls 数大于 0 视为良好（避免没有调用直接乱答）
        from deepseek_v4.training.tool_use.schema import validate_tool_call
        schemas = self.tool_registry.schemas
        format_scores: List[float] = []
        for t in trajectories:
            total_calls = sum(len(s.tool_calls) for s in t.steps)
            valid_calls = 0
            for s in t.steps:
                for c in s.tool_calls:
                    if validate_tool_call(c, schemas).ok:
                        valid_calls += 1
            if total_calls == 0:
                format_scores.append(0.0)
            else:
                format_scores.append(valid_calls / total_calls)

        # 3. efficiency：step 数越多惩罚越大
        efficiency_scores: List[float] = []
        for t in trajectories:
            n = len(t.steps)
            if n <= cfg.max_efficient_turns:
                efficiency_scores.append(1.0)
            else:
                # 线性衰减
                excess = n - cfg.max_efficient_turns
                efficiency_scores.append(max(0.0, 1.0 - 0.2 * excess))

        # 综合
        out: List[float] = []
        for c, f, e in zip(correctness, format_scores, efficiency_scores):
            r = (
                cfg.reward_correctness_weight * c
                + cfg.reward_format_weight * f
                + cfg.reward_efficiency_weight * e
            )
            out.append(float(r))

        # 设置 success 标记
        for t, c in zip(trajectories, correctness):
            t.success = c > 0.5
        return out

    # ----- Loss（与 GRPO 相同） -----

    def _compute_loss(self, mb: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cfg = self.config
        sequences = mb["sequences"].to(self.device)
        response_mask = mb["response_mask"].to(self.device).float()
        old_logp = mb["old_logp"].to(self.device)
        ref_logp = mb["ref_logp"].to(self.device)
        advantages = mb["advantages"].to(self.device)

        attn = (sequences != self.tokenizer.pad_token_id).long() | response_mask.long()
        attn = attn.clamp(max=1)
        out = self.policy(input_ids=sequences, attention_mask=attn, use_cache=False)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        new_logp = _per_token_logp_from_logits(logits, sequences, response_mask)

        log_ratio = (new_logp - old_logp) * response_mask
        ratio = log_ratio.exp()
        adv_tok = advantages[:, None].expand_as(new_logp)

        unclipped = -adv_tok * ratio
        clipped = -adv_tok * ratio.clamp(1 - cfg.cliprange, 1 + cfg.cliprange)
        denom = response_mask.sum().clamp(min=1.0)
        policy_loss = (torch.maximum(unclipped, clipped) * response_mask).sum() / denom

        kl = _kl_per_token(new_logp, ref_logp, response_mask, estimator=cfg.kl_estimator)
        kl_loss = kl.sum() / denom

        loss = policy_loss + cfg.beta_kl * kl_loss

        with torch.no_grad():
            approx_kl = (((ratio - 1) - log_ratio) * response_mask).sum() / denom

        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "approx_kl": approx_kl.detach(),
        }

    # ----- 工具 -----

    def _iter_minibatches(self, n: int, mb: int):
        idx = torch.randperm(n)
        for i in range(0, n, mb):
            yield idx[i:i + mb].tolist()

    def _save_ckpt(self):
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
            "iter": self.iter_idx, "global_step": self.global_step,
        })
        logger.info(f"[AgenticRL] checkpoint saved to {save_dir}")
