"""
SFT (Supervised Fine-Tuning) Trainer。

设计要点：
1. 复用 BaseTrainer，仅覆写 compute_loss 与数据集
2. Loss mask：只在 assistant 部分计算 cross entropy
3. 支持多轮对话、思考模式、工具调用
4. 不需要 ZLoss / Aux Loss（继承自 BaseTrainer，可选关闭）
5. 采用 NEFTune（noisy embedding）增强（可选）
6. 支持 token-level / sequence-level loss reduction

NEFTune 参考：Jain et al. (2023) NEFTune: Noisy Embeddings Improve Instruction Finetuning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from deepseek_v4.data.collator import SFTCollator
from deepseek_v4.data.dataset import SFTDataset
from deepseek_v4.modeling.model import DeepseekV4ForCausalLM
from deepseek_v4.training.base_trainer import BaseTrainer, TrainerConfig
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 配置
# ============================================================

@dataclass
class SFTConfig(TrainerConfig):
    """SFT 阶段配置。"""

    # 数据
    train_data_paths: List[str] = field(default_factory=list)
    eval_data_paths: List[str] = field(default_factory=list)
    max_seq_len: int = 4096
    cache_dir: Optional[str] = "cache/datasets"
    thinking_mode_default: str = "chat"   # chat | thinking
    mask_user: bool = True                 # 是否对 user / system mask（默认 True）

    # 模型
    model_config_path: str = "configs/model/mini_2b.json"
    init_from_checkpoint: str = "checkpoints/pretrain/checkpoint-final"
    tokenizer_path: str = "checkpoints/tokenizer"

    # 损失
    z_loss_weight: float = 0.0             # SFT 默认关 z_loss
    aux_loss_weight: float = 0.001         # SFT 维持非常小的 aux loss 防止 expert collapse
    use_aux_loss: bool = True
    loss_reduction: str = "token"          # token | sequence
    label_smoothing: float = 0.0

    # NEFTune
    neftune_alpha: float = 0.0             # 0 关闭，常用 5

    # 默认覆盖（SFT 用更小 lr）
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    save_steps: int = 500
    eval_steps: int = 500
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    metric_name: str = "eval_loss"


# ============================================================
# NEFTune 实现
# ============================================================

class _NEFTuneEmbedding:
    """
    给 Embedding 层挂 forward hook：训练时给输出加均匀噪声。

    噪声幅度公式：
        noise ~ U(-α/√(d·L), α/√(d·L))
    其中 d=hidden_size，L=序列长度。
    """
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.handle = None

    def _hook(self, module, args, output):
        if not module.training or self.alpha <= 0:
            return output
        # output: [B, L, D]
        emb = output
        L = emb.shape[1]
        D = emb.shape[-1]
        mag = self.alpha / ((L * D) ** 0.5)
        noise = torch.empty_like(emb).uniform_(-mag, mag)
        return emb + noise

    def attach(self, embedding: nn.Embedding):
        self.detach()
        self.handle = embedding.register_forward_hook(self._hook)

    def detach(self):
        if self.handle is not None:
            try:
                self.handle.remove()
            except Exception:
                pass
            self.handle = None


# ============================================================
# Aux loss hook（轻量复用 pretrain 的实现）
# ============================================================

class _SFTRouterHook:
    """与 PretrainTrainer 中相同，简化版。"""
    def __init__(self):
        self.collected: List = []
        self.handles: List = []

    def hook(self, module, args, kwargs, output):
        if isinstance(output, tuple) and len(output) == 3:
            logits, weights, indices = output
            with torch.no_grad():
                probs = F.softmax(logits.float(), dim=-1)
            self.collected.append((probs, indices, module.num_experts))

    def attach(self, model):
        from deepseek_v4.modeling.model import DeepseekV4TopKRouter
        self.detach()
        for m in model.modules():
            if isinstance(m, DeepseekV4TopKRouter):
                self.handles.append(m.register_forward_hook(self.hook, with_kwargs=True))

    def detach(self):
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
            mask = F.one_hot(indices, num_classes=num_experts).float().sum(dim=1)
            f = mask.mean(dim=0) / top_k
            p = probs.mean(dim=0)
            total = total + num_experts * (f * p.detach()).sum()
        self.collected.clear()
        return total / n


# ============================================================
# SFTTrainer
# ============================================================

class SFTTrainer(BaseTrainer):
    """
    SFT 训练器。

    核心逻辑：
        1. 模型前向得到 logits
        2. shift labels（next-token prediction）
        3. 仅 assistant 部分参与 loss（label = -100 处忽略）
        4. 可选 NEFTune 噪声、aux loss
    """

    def __init__(
        self,
        config: SFTConfig,
        model: DeepseekV4ForCausalLM,
        tokenizer,
    ):
        super().__init__(config=config, model=model)
        self.config: SFTConfig = config
        self.tokenizer = tokenizer
        self._train_ds: Optional[Dataset] = None
        self._eval_ds: Optional[Dataset] = None
        self._collator: Optional[Callable] = None
        self._neftune: Optional[_NEFTuneEmbedding] = None
        self._router_hook: Optional[_SFTRouterHook] = None

    # ----- Dataset / Collator -----

    def get_train_dataset(self) -> Dataset:
        if self._train_ds is None:
            self._train_ds = SFTDataset(
                paths=self.config.train_data_paths,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                thinking_mode_default=self.config.thinking_mode_default,
                mask_user=self.config.mask_user,
                cache_dir=self.config.cache_dir,
                ignore_index=-100,
            )
        return self._train_ds

    def get_eval_dataset(self) -> Optional[Dataset]:
        if self.config.eval_data_paths and self._eval_ds is None:
            self._eval_ds = SFTDataset(
                paths=self.config.eval_data_paths,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                thinking_mode_default=self.config.thinking_mode_default,
                mask_user=self.config.mask_user,
                cache_dir=self.config.cache_dir,
                ignore_index=-100,
            )
        return self._eval_ds

    def get_collator(self) -> Callable:
        if self._collator is None:
            self._collator = SFTCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                ignore_index=-100,
                pad_to_multiple_of=8,
            )
        return self._collator

    # ----- Setup -----

    def setup(self) -> None:
        super().setup()
        unwrap = self.model.module if hasattr(self.model, "module") else self.model
        # NEFTune
        if self.config.neftune_alpha > 0:
            self._neftune = _NEFTuneEmbedding(self.config.neftune_alpha)
            embed = unwrap.model.embed_tokens if hasattr(unwrap, "model") else unwrap.embed_tokens
            self._neftune.attach(embed)
            logger.info(f"[SFT] NEFTune enabled (alpha={self.config.neftune_alpha})")
        # Aux loss
        if self.config.use_aux_loss:
            self._router_hook = _SFTRouterHook()
            self._router_hook.attach(unwrap)

    # ----- Compute Loss -----

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids.clone())
        attention_mask = batch.get("attention_mask")

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        # shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # ----- 主损失 -----
        if self.config.loss_reduction == "token":
            loss = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
                reduction="mean",
            )
        elif self.config.loss_reduction == "sequence":
            # 每个序列内 token-level mean，再跨 batch mean（更公平对待短/长序列）
            B, L, V = shift_logits.shape
            ce = F.cross_entropy(
                shift_logits.float().view(-1, V),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
                label_smoothing=self.config.label_smoothing,
            ).view(B, L)
            mask = (shift_labels != -100).float()
            seq_loss = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            loss = seq_loss.mean()
        else:
            raise ValueError(self.config.loss_reduction)

        out = {"loss": loss, "ce_loss": loss.detach()}

        # ----- aux -----
        if self.config.use_aux_loss and self._router_hook is not None:
            aux = self._router_hook.compute_aux(top_k=self.tokenizer is not None and 4 or 4).to(loss.device)
            out["loss"] = out["loss"] + self.config.aux_loss_weight * aux
            out["aux_loss"] = aux.detach()

        # ----- 准确率（仅有效位置）-----
        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)
            valid = shift_labels != -100
            acc = ((preds == shift_labels) & valid).sum().float() / valid.sum().clamp(min=1.0).float()
            out["acc"] = acc.detach()

        return out

    def __del__(self):
        if self._neftune is not None:
            try:
                self._neftune.detach()
            except Exception:
                pass
        if self._router_hook is not None:
            try:
                self._router_hook.detach()
            except Exception:
                pass
