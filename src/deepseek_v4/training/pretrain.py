"""
Pretrain Trainer：

- 基于 BaseTrainer
- 标准 next-token prediction 损失
- 可选 ZLoss（防止 logit 爆炸，DeepSeek 默认开）
- 可选 MoE auxiliary balance loss（通过 router 钩子）
- MFU 计算与 tokens/sec 上报
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from deepseek_v4.data.collator import PretrainCollator
from deepseek_v4.data.dataset import PackedDataset, PretrainDataset
from deepseek_v4.modeling.model import (
    DeepseekV4ForCausalLM, DeepseekV4SparseMoeBlock, DeepseekV4TopKRouter,
)
from deepseek_v4.training.base_trainer import BaseTrainer, TrainerConfig
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 配置
# ============================================================

@dataclass
class PretrainConfig(TrainerConfig):
    # 数据
    train_data_paths: List[str] = field(default_factory=list)
    eval_data_paths: List[str] = field(default_factory=list)
    text_field: str = "text"
    max_seq_len: int = 4096
    use_packed_dataset: bool = True
    cache_dir: Optional[str] = "cache/datasets"

    # 模型
    model_config_path: str = "configs/model/mini_2b.json"
    init_from_checkpoint: Optional[str] = None
    tokenizer_path: str = "checkpoints/tokenizer"

    # 损失
    z_loss_weight: float = 1e-4          # 经验值 1e-4 ~ 1e-3
    aux_loss_weight: float = 0.01        # MoE balance loss
    aux_loss_topk: int = 6
    use_aux_loss: bool = True

    # MFU 估算
    estimate_mfu: bool = True
    gpu_peak_flops: float = 312e12       # A100 BF16 = 312 TFLOPs


# ============================================================
# Aux loss 钩子
# ============================================================

class _RouterHook:
    """
    Switch Transformer / DeepSeek MoE 风格 load-balancing aux loss。

    通过 forward hook 收集每个 router 的：
        - probs:   [N, num_experts]
        - indices: [N, top_k]
    然后计算：
        f_i = (1/N/topk) * Σ 1{i ∈ top_k(t)}
        p_i = (1/N)      * Σ probs[t, i]
        aux = num_experts * Σ f_i * p_i
    """

    def __init__(self):
        self.collected: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        self.handles: List[Any] = []

    def hook(self, module, args, kwargs, output):
        # output: (logits, weights, indices)
        if isinstance(output, tuple) and len(output) == 3:
            logits, weights, indices = output
            with torch.no_grad():
                # router probs 用 softmax(logits) 重新计算（评分不一定是 prob）
                probs = F.softmax(logits.float(), dim=-1)
            self.collected.append((probs, indices, module.num_experts))

    def attach(self, model: nn.Module) -> None:
        self.detach()
        for m in model.modules():
            if isinstance(m, DeepseekV4TopKRouter):
                # 只 hook topk router（hash router 不参与 balance loss）
                self.handles.append(
                    m.register_forward_hook(self.hook, with_kwargs=True)
                )

    def detach(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
        self.collected.clear()

    def compute_aux(self, top_k: int) -> torch.Tensor:
        if not self.collected:
            return torch.tensor(0.0)
        total = 0.0
        n = len(self.collected)
        for probs, indices, num_experts in self.collected:
            # f_i: 每个 expert 命中比例
            mask = F.one_hot(indices, num_classes=num_experts).float().sum(dim=1)  # [N, E]
            f = mask.mean(dim=0) / top_k                                            # [E]
            p = probs.mean(dim=0)                                                   # [E]
            total = total + num_experts * (f * p.detach()).sum()
        # 收集完即释放（下一 step 重置）
        self.collected.clear()
        return total / n


# ============================================================
# PretrainTrainer
# ============================================================

class PretrainTrainer(BaseTrainer):
    """
    预训练训练器。

    用法：
        trainer = PretrainTrainer(config, model, tokenizer)
        trainer.train()
    """

    def __init__(
        self,
        config: PretrainConfig,
        model: DeepseekV4ForCausalLM,
        tokenizer,
    ):
        super().__init__(config=config, model=model)
        self.config: PretrainConfig = config
        self.tokenizer = tokenizer
        self._train_ds: Optional[Dataset] = None
        self._eval_ds: Optional[Dataset] = None
        self._collator: Optional[Callable] = None
        self._router_hook: Optional[_RouterHook] = None
        self._activated_params: Optional[int] = None

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------

    def get_train_dataset(self) -> Dataset:
        if self._train_ds is None:
            cls = PackedDataset if self.config.use_packed_dataset else PretrainDataset
            self._train_ds = cls(
                paths=self.config.train_data_paths,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                text_field=self.config.text_field,
                cache_dir=self.config.cache_dir,
            )
        return self._train_ds

    def get_eval_dataset(self) -> Optional[Dataset]:
        if self.config.eval_data_paths and self._eval_ds is None:
            self._eval_ds = PretrainDataset(
                paths=self.config.eval_data_paths,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                text_field=self.config.text_field,
                cache_dir=self.config.cache_dir,
            )
        return self._eval_ds

    def get_collator(self) -> Callable:
        if self._collator is None:
            self._collator = PretrainCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                ignore_index=-100,
                pad_to_multiple_of=8,
            )
        return self._collator

    # ------------------------------------------------------------------
    # Setup hook
    # ------------------------------------------------------------------

    def setup(self) -> None:
        super().setup()
        if self.config.use_aux_loss:
            self._router_hook = _RouterHook()
            unwrap = self.model.module if hasattr(self.model, "module") else self.model
            self._router_hook.attach(unwrap)
            logger.info(f"[PretrainTrainer] aux loss enabled, "
                        f"weight={self.config.aux_loss_weight}")
        # 估算激活参数数（用于 MFU）
        if self.config.estimate_mfu:
            self._activated_params = self._estimate_activated_params()
            logger.info(f"[MFU] activated params (per token): "
                        f"{self._activated_params/1e9:.2f}B")

    def _estimate_activated_params(self) -> int:
        """估算每 token 激活参数数（MoE：用 top_k/n_experts 比例）。"""
        unwrap = self.model.module if hasattr(self.model, "module") else self.model
        cfg = unwrap.config if hasattr(unwrap, "config") else None
        if cfg is None:
            return sum(p.numel() for p in self.model.parameters())
        total = sum(p.numel() for n, p in self.model.named_parameters())
        # 路由专家
        routed = 0
        for n, p in self.model.named_parameters():
            if ".mlp.experts." in n:
                routed += p.numel()
        ratio = cfg.num_experts_per_tok / max(cfg.n_routed_experts, 1)
        activated_routed = int(routed * ratio)
        return total - routed + activated_routed

    # ------------------------------------------------------------------
    # 损失
    # ------------------------------------------------------------------

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids.clone())

        # 模型前向
        outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        # ce loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        out = {"loss": ce_loss, "ce_loss": ce_loss.detach()}

        # ZLoss
        if self.config.z_loss_weight > 0:
            log_z = torch.logsumexp(shift_logits.float(), dim=-1)
            z_loss = (log_z ** 2).mean()
            out["loss"] = out["loss"] + self.config.z_loss_weight * z_loss
            out["z_loss"] = z_loss.detach()

        # Aux loss（MoE balance）
        if self.config.use_aux_loss and self._router_hook is not None:
            aux = self._router_hook.compute_aux(self.config.aux_loss_topk)
            if aux.requires_grad or aux.is_leaf:
                aux = aux.to(out["loss"].device)
                out["loss"] = out["loss"] + self.config.aux_loss_weight * aux
                out["aux_loss"] = aux.detach()

        # ppl
        with torch.no_grad():
            out["ppl"] = ce_loss.detach().exp()

        return out

    # ------------------------------------------------------------------
    # MFU 估算
    # ------------------------------------------------------------------

    def on_step_end(self, metrics: Dict[str, float]) -> None:
        if not self.config.estimate_mfu or self._activated_params is None:
            return
        tokens = metrics.get("tokens", 0)
        # forward + backward ≈ 6 N D
        flops = 6 * tokens * self._activated_params
        # 取最近一段时间
        # 用 metric_logger 里 step time 平均
        step_time = self.timer.records.get("forward", 0) + self.timer.records.get("backward", 0) \
                    + self.timer.records.get("optim", 0)
        if self.timer.counts.get("forward", 0) > 0:
            n = self.timer.counts["forward"]
            avg_step_time = step_time / n
        else:
            return
        if avg_step_time <= 0:
            return
        from deepseek_v4.distributed.utils import get_world_size
        flops_per_sec = flops / avg_step_time
        mfu = flops_per_sec / (self.config.gpu_peak_flops * get_world_size())
        metrics["mfu"] = mfu
        self.metric_logger.update(mfu=mfu)

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def __del__(self):
        if self._router_hook is not None:
            try:
                self._router_hook.detach()
            except Exception:
                pass
