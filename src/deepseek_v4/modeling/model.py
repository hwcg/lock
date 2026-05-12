""" ==============================================================================
DeepSeek-V4 完整 PyTorch 实现 (Pure PyTorch Implementation)
==============================================================================
特点：
1. 仅依赖 torch / torch.nn / torch.nn.functional 以及标准库
2. 完整保留 V4 的所有核心结构：
   - MLA：Q LoRA + 单 KV 头 (shared-KV MQA) + 分组输出 LoRA + per-head attention sink
   - 三种层类型：sliding_attention / compressed_sparse_attention / heavily_compressed_attention
   - Hyper-Connections (mHC)：hc_mult 并行残差流 + Sinkhorn 投影
   - Lightning Indexer 稀疏注意力 (CSA)
   - 混合路由 MoE (前 num_hash_layers 层 hash routing，其余 sqrtsoftplus topk)
   - YaRN RoPE + interleaved partial rotary + 输出端反向 RoPE
   - 滑动窗口 + 压缩 KV cache
3. 支持加载 HuggingFace 官方 checkpoint
4. 提供完整版与 2B mini 版的配置工厂函数

作者注：本实现严格对齐 transformers 中的 modeling_deepseek_v4 命名约定，
       确保 state_dict 键与 HF checkpoint 完全兼容。
"""
import math
import json
import os
import glob
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 第一部分：模型配置
# ============================================================================

@dataclass
class DeepseekV4Config:
    """
    DeepSeek-V4 完整配置类。

    字段命名与 HuggingFace config.json 保持一致，可直接 from_json_file 加载。
    """
    # ---------- 基础结构 ----------
    vocab_size: int = 129280
    hidden_size: int = 7168
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 1            # MLA 单 KV 头（shared-KV MQA）
    head_dim: int = 512                     # 每个注意力头维度
    qk_rope_head_dim: int = 64              # 应用 RoPE 的子维度（位于每头尾部）
    max_position_embeddings: int = 1048576  # 1M 上下文
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # ---------- MLA / 注意力 ----------
    q_lora_rank: int = 1536                 # Q 的低秩瓶颈
    o_lora_rank: int = 1024                 # 输出投影低秩
    o_groups: int = 16                      # 输出投影分组数
    sliding_window: int = 128               # 滑动窗口大小
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # ---------- 压缩注意力（CSA / HCA） ----------
    # compress_ratios[i] 指定第 i 层压缩比：
    #   0   -> sliding_attention（纯滑动窗口，最后一层）
    #   4   -> compressed_sparse_attention（CSA，带 Lightning Indexer）
    #   128 -> heavily_compressed_attention（HCA，纯长程压缩）
    compress_ratios: Tuple[int, ...] = (
        128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 0,
    )
    compress_rope_theta: float = 160000.0   # 压缩 RoPE 的 theta（大于主路径）

    # ---------- Lightning Indexer（仅 CSA 层使用） ----------
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 1024                  # 每个 query 选 top-1024 压缩 KV

    # ---------- RoPE / YaRN ----------
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    # ---------- MoE ----------
    moe_intermediate_size: int = 3072
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    num_hash_layers: int = 3                # 前 N 层用 hash routing
    norm_topk_prob: bool = True
    scoring_func: str = "sqrtsoftplus"      # 路由评分函数：softmax/sigmoid/sqrtsoftplus
    routed_scaling_factor: float = 2.5
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0              # SwiGLU 输入裁剪范围

    # ---------- Hyper-Connections (mHC) ----------
    hc_mult: int = 4                        # 并行残差流数量
    hc_sinkhorn_iters: int = 20             # Sinkhorn 迭代次数
    hc_eps: float = 1e-6

    # ---------- Token IDs ----------
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: Optional[int] = None
    use_cache: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self):
        # ----- 默认 rope_scaling（V4 YaRN）-----
        if self.rope_scaling is None:
            self.rope_scaling = {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "type": "yarn",
            }

        # ----- 自动推断每层类型 -----
        self.layer_types: List[str] = []
        self.mlp_layer_types: List[str] = []
        for i in range(self.num_hidden_layers):
            ratio = self.compress_ratios[i] if i < len(self.compress_ratios) else 0
            if ratio == 0:
                self.layer_types.append("sliding_attention")
            elif ratio == 4:
                self.layer_types.append("compressed_sparse_attention")
            elif ratio == 128:
                self.layer_types.append("heavily_compressed_attention")
            else:
                raise ValueError(f"未知 compress_ratio={ratio} (layer {i})")
            self.mlp_layer_types.append(
                "hash_moe" if i < self.num_hash_layers else "top_k"
            )

        # ----- 压缩比字典 -----
        self.compress_rates = {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }

        # ----- rope_parameters：main / compress 两类 -----
        partial = self.qk_rope_head_dim / self.head_dim
        rope_extra = {k: v for k, v in self.rope_scaling.items() if k != "type"}
        rope_type = self.rope_scaling.get("type", "yarn")
        self.rope_parameters = {
            "main": {
                "rope_type": rope_type,
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": partial,
                **rope_extra,
            },
            "compress": {
                "rope_type": rope_type,
                "rope_theta": self.compress_rope_theta,
                "partial_rotary_factor": partial,
                **rope_extra,
            },
        }

        # ----- 共享专家 intermediate_size -----
        # 共享专家容量 = moe_intermediate_size * n_shared_experts
        self.intermediate_size = self.moe_intermediate_size * self.n_shared_experts

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeepseekV4Config":
        """从 HuggingFace config.json dict 构造。仅保留本类已定义的字段。"""
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**keep)

    @classmethod
    def from_json_file(cls, path: str) -> "DeepseekV4Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ============================================================================
# 第二部分：RoPE 工具函数（YaRN）
# ============================================================================

def _yarn_find_correction_dim(num_rot, dim, base, max_pos):
    """YaRN: 给定旋转圈数找对应维度（公式 (4)）"""
    return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_pos):
    """YaRN: 找到 [low_rot, high_rot] 对应的维度范围"""
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_, max_, dim):
    """YaRN: 在 [min_, max_] 间线性过渡的 mask（0 → 1）"""
    if min_ == max_:
        max_ += 0.001
    linear = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
    return torch.clamp(linear, 0, 1)


def compute_rope_inv_freq(config: DeepseekV4Config, layer_type: str = "main"):
    """
    根据 layer_type ('main' / 'compress') 计算 RoPE 的 inv_freq。

    返回：(inv_freq: [rope_head_dim//2], attention_scaling: float)
    """
    params = config.rope_parameters[layer_type]
    base = params["rope_theta"]
    partial_factor = params.get("partial_rotary_factor", 1.0)
    dim = int(config.head_dim * partial_factor)  # rope_head_dim
    rope_type = params["rope_type"]

    if rope_type == "default":
        # 标准 RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        return inv_freq, 1.0

    if rope_type in ("yarn", "deepseek_yarn"):
        # YaRN：内插 + 外推 + 线性过渡
        factor = params["factor"]
        beta_fast = params.get("beta_fast", 32)
        beta_slow = params.get("beta_slow", 1)
        original_max_pos = params["original_max_position_embeddings"]

        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        inv_freq_extra = 1.0 / pos_freqs
        inv_freq_inter = 1.0 / (factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, dim, base, original_max_pos
        )
        extra_factor = 1 - _yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = inv_freq_inter * (1 - extra_factor) + inv_freq_extra * extra_factor
        # V4 关闭 mscale，故 attention_scaling = 1.0
        return inv_freq, 1.0

    raise ValueError(f"未知 rope_type: {rope_type}")


# ============================================================================
# 第三部分：Normalization 层
# ============================================================================

class DeepseekV4RMSNorm(nn.Module):
    """带可学习权重的 RMS Norm（T5 风格）"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x.to(dtype)


class DeepseekV4UnweightedRMSNorm(nn.Module):
    """无可学习权重的 RMS Norm（用于 q_b_norm 和 HC 内部归一化）"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        ).to(x.dtype)


# ============================================================================
# 第四部分：Rotary Embedding 与 apply_rotary_pos_emb（interleaved partial RoPE）
# ============================================================================

class DeepseekV4RotaryEmbedding(nn.Module):
    """
    多层类型 RoPE：每个 rope 类型 ('main' / 'compress') 各持一份 inv_freq buffer。

    V4 特点：
    - interleaved（交错）RoPE：[a0,b0,a1,b1,...] 配对
    - partial RoPE：仅作用于每头最后 rope_head_dim 个通道
    - cos/sin 返回为半尺寸 ([..., rope_head_dim//2])，由 apply_rotary_pos_emb
      内部做 repeat_interleave(2) 扩展
    """
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        # 仅保留嵌套 dict 的 key 为 layer_type
        self.layer_types = [
            k for k, v in config.rope_parameters.items() if isinstance(v, dict)
        ]
        self.rope_type: Dict[str, str] = {}
        for lt in self.layer_types:
            params = config.rope_parameters[lt]
            self.rope_type[lt] = params["rope_type"]
            inv_freq, attn_scaling = compute_rope_inv_freq(config, layer_type=lt)
            self.register_buffer(f"{lt}_inv_freq", inv_freq, persistent=False)
            setattr(self, f"{lt}_attention_scaling", attn_scaling)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str = "main",
    ):
        """
        Args:
            x: 用于推断 dtype/device 的引用张量
            position_ids: [B, S]
            layer_type: 'main' / 'compress'
        Returns:
            cos, sin: 形状均为 [B, S, rope_head_dim//2]
        """
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attn_scaling = getattr(self, f"{layer_type}_attention_scaling")

        # [B, dim//2, 1]
        inv_freq_exp = inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)

        # [B, 1, S]
        pos_exp = position_ids[:, None, :].float()

        # 强制 FP32 计算（避免低精度溢出）
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_exp.float() @ pos_exp.float()).transpose(1, 2)
            cos = freqs.cos() * attn_scaling
            sin = freqs.sin() * attn_scaling

        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """interleaved RoPE 的 rotate_half：[a0,b0,a1,b1,...] → [-b0,a0,-b1,a1,...]"""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """
    将 RoPE 应用于 x 的最后 rope_head_dim 个通道（partial rotary）。

    cos/sin 是半尺寸（每对一个值），需要 repeat_interleave(2) 扩展。
    对输出端反向 RoPE，调用方传入 -sin 即可。
    """
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


# ============================================================================
# 第五部分：分组线性层（输出投影 LoRA）
# ============================================================================

class DeepseekV4GroupedLinear(nn.Linear):
    """
    块对角分组线性层。

    将 num_attention_heads*head_dim 输入按头分为 o_groups 组，每组独立做投影。
    每组：(num_heads*head_dim/o_groups) -> o_lora_rank

    形状：
    - weight: [o_groups * o_lora_rank, num_heads*head_dim/o_groups]
    - 内部 view 为 [o_groups, o_lora_rank, hidden_per_group]
    """
    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        # weight: [n_groups, out_per_group, in_per_group] -> 转置后做 bmm
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        # x: [..., n_groups, in_per_group] -> [n_groups, N, in_per_group]
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


# ============================================================================
# 第六部分：Cache 类（三种类型）
# ============================================================================

class DeepseekV4SlidingCache:
    """
    滑动窗口 KV 缓存（V4 中 K==V 共享存储）

    update() 返回当前注意力本步可见的全部 KV（cache 历史 + 新输入），
    内部仅保留最近 sliding_window-1 个用于下一步。
    """
    def __init__(self, config: DeepseekV4Config):
        self.sliding_window = config.sliding_window
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.cumulative_length = 0
        self.is_initialized = False

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            # 懒初始化为空 cache
            self.keys = torch.empty(
                *key_states.shape[:-2], 0, key_states.shape[-1],
                dtype=key_states.dtype, device=key_states.device,
            )
            self.is_initialized = True
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        # 保留最新 sliding_window-1 个（给下次拼接腾位）
        if full.shape[-2] >= self.sliding_window:
            self.keys = full[..., -self.sliding_window + 1:, :].contiguous()
        else:
            self.keys = full
        self.values = self.keys
        return full, full


class DeepseekV4HCACache(DeepseekV4SlidingCache):
    """
    HCA 层缓存：在滑动窗口基础上加 compressor 状态（无 overlap，无 indexer）。

    状态用 dict 索引；HCA 只有 'compressor'，CSA 还有 'indexer'。
    """
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.buffer_kv: Dict[str, Optional[torch.Tensor]] = {"compressor": None}
        self.buffer_gate: Dict[str, Optional[torch.Tensor]] = {"compressor": None}
        self.compressed_kv: Dict[str, Optional[torch.Tensor]] = {"compressor": None}
        self.entry_count: Dict[str, int] = {"compressor": 0}

    def store_compression_weights(
        self, name: str, kv: torch.Tensor, gate: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        把新到的 (kv, gate) 拼到 buffer，剥离 compress_rate 的整数倍前缀，
        余数留在 buffer 给下次。返回 (chunk_kv, chunk_gate, first_window_position)。
        """
        first_window_position = self.entry_count[name] * self.compress_rate
        buf_kv, buf_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buf_kv is not None and buf_kv.shape[1] > 0:
            kv = torch.cat([buf_kv, kv], dim=1)
            gate = torch.cat([buf_gate, gate], dim=1)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        self.buffer_kv[name] = kv[:, usable:]
        self.buffer_gate[name] = gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_window_position

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        """追加新压缩 entries，更新 entry_count，返回累计的全部 compressed_kv。"""
        if self.compressed_kv[name] is None:
            self.compressed_kv[name] = compressed
        elif compressed.shape[1] > 0:
            self.compressed_kv[name] = torch.cat(
                [self.compressed_kv[name], compressed], dim=1
            )
        self.entry_count[name] += compressed.shape[1]
        return self.compressed_kv[name]


class DeepseekV4CSACache(DeepseekV4HCACache):
    """
    CSA 层缓存：HCA 基础上增加 'indexer' 通道和 overlap 状态。

    overlap_kv/gate 保存上一次 forward 最后一个窗口的 Ca 切片（前 head_dim），
    供下次 forward 的第一个窗口使用。
    """
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        # 添加 indexer 通道
        self.buffer_kv["indexer"] = None
        self.buffer_gate["indexer"] = None
        self.compressed_kv["indexer"] = None
        self.entry_count["indexer"] = 0
        # overlap 状态
        self.overlap_kv: Dict[str, Optional[torch.Tensor]] = {
            "compressor": None, "indexer": None,
        }
        self.overlap_gate: Dict[str, Optional[torch.Tensor]] = {
            "compressor": None, "indexer": None,
        }

    def update_overlap_state(
        self, name: str, chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, head_dim: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        读取上次保存的 overlap Ca 切片（用于本次第 0 个窗口），
        并保存本次最后一个窗口的 Ca 切片给下次。
        第一次调用返回 (None, None)。
        """
        prior_kv = self.overlap_kv[name]
        prior_gate = self.overlap_gate[name]
        # 只保存 Ca（前 head_dim），Cb 已用过
        self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
        self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
        return prior_kv, prior_gate


class DeepseekV4Cache:
    """整模型多层 cache 集合，按层类型分配不同的 cache。"""
    def __init__(self, config: DeepseekV4Config):
        self.layers: List[Union[
            DeepseekV4SlidingCache, DeepseekV4HCACache, DeepseekV4CSACache
        ]] = []
        for i in range(config.num_hidden_layers):
            t = config.layer_types[i]
            if t == "sliding_attention":
                self.layers.append(DeepseekV4SlidingCache(config))
            elif t == "heavily_compressed_attention":
                self.layers.append(DeepseekV4HCACache(config))
            elif t == "compressed_sparse_attention":
                self.layers.append(DeepseekV4CSACache(config))

    def get_seq_length(self) -> int:
        return self.layers[0].cumulative_length if self.layers else 0


# ============================================================================
# 第七部分：HCA / CSA 压缩器 + Lightning Indexer
# ============================================================================

class DeepseekV4HCACompressor(nn.Module):
    """
    HCA 压缩器：每 compress_rate=128 个源 token 压缩为一个 KV entry。

    公式（论文 §2.3.2）：
        C^Comp_i = Σ_{j∈window} softmax(Z_j + B)_j ⊙ C_j
    RoPE 在每个窗口的"代表位置" (i * 128 + first_window_position) 应用。
    """
    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        # 公式 (20)(21): C = H·W^KV, Z = H·W^Z
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        # 窗口内位置偏置（公式 22: B 是可学习参数）
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[DeepseekV4Cache],
        layer_idx: int,
    ) -> torch.Tensor:
        batch = hidden_states.shape[0]
        cache_layer = (
            past_key_values.layers[layer_idx] if past_key_values is not None else None
        )
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            # 无 cache（单步推理）：只取整数倍前缀，余数丢弃
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights(
                "compressor", kv, gate
            )

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            # 重塑为 [B, n_windows, compress_rate, head_dim]
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + \
                         self.position_bias.to(chunk_gate.dtype)

            # 每窗口内 softmax 加权（FP32 稳定性）
            weights = chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)
            compressed = self.kv_norm((chunk_kv * weights).sum(dim=2))

            # 每窗口 RoPE 在 i*128 + first_window_position 位置
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        # 添加 head 维 [B, 1, T, head_dim]，匹配 attention KV 形状
        return compressed.unsqueeze(1)


class DeepseekV4Indexer(nn.Module):
    """
    Lightning Indexer（论文 §2.3.1, 公式 13-17）。

    为每个 query 选 index_topk 个压缩 KV 块。评分：
        score_{t,s} = Σ_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)
    Indexer 自带一个独立的小型压缩器，作用在 hidden_states 上得到压缩 keys。
    """
    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim ** -0.5
        self.weights_scaling = self.num_heads ** -0.5

        # Indexer 的内部压缩器（2*head_dim 用于 Ca/Cb overlap 布局）
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # Indexer 自己的 q_b：从 q_lora_rank 投到 num_heads*head_dim
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        # 头权重 w_{t,h}
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        # 独立 rotary（同 compress theta）
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[DeepseekV4Cache],
        layer_idx: int,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer = (
            past_key_values.layers[layer_idx] if past_key_values is not None else None
        )
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights(
                "indexer", kv, gate
            )

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + \
                         self.position_bias.to(chunk_gate.dtype)

            # ----- CSA 风格 overlap 布局 -----
            # Ca = [..., :head_dim] 贡献给「下一」窗口
            # Cb = [..., head_dim:] 贡献给「当前」窗口
            # 每个 window 的 compressed = window-1 的 Ca + window 的 Cb（softmax 加权）
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full(
                (batch, n_windows, 2 * ratio, self.head_dim), float("-inf")
            )
            # 后半填 Cb（当前窗口）
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
            # 前半填上一窗口的 Ca
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :self.head_dim]
            # 跨 forward 边界的 overlap（用上次保存的 Ca）
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state(
                    "indexer", chunk_kv, chunk_gate, self.head_dim
                )
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            weights = new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)
            compressed = self.kv_norm((new_kv * weights).sum(dim=2))

            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        compressed_kv = (
            compressed if cache_layer is None
            else cache_layer.update_compressor_states("indexer", compressed)
        )

        # ----- Query 路径：q_b_proj + RoPE -----
        cos_q, sin_q = self.rotary_emb(
            hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type
        )
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)  # [B, S, H, D]

        # ----- 评分：ReLU(q·k^T) * weights -----
        # scores: [B, S, H, T]
        scores = torch.matmul(
            q.float(),
            compressed_kv.transpose(-1, -2).float().unsqueeze(1)
        )
        scores = F.relu(scores) * self.softmax_scale
        # weights: [B, S, H]
        head_weights = self.weights_proj(hidden_states).float() * self.weights_scaling
        # index_scores: [B, S, T]
        index_scores = (scores * head_weights.unsqueeze(-1)).sum(dim=2)

        topk = min(self.index_topk, compressed_kv.shape[1])
        return index_scores.topk(topk, dim=-1).indices


class DeepseekV4CSACompressor(nn.Module):
    """
    CSA 压缩器（论文 §2.3.1）：每 compress_rate=4 个 token 压缩。

    与 HCA 不同点：
    - 使用 overlap 布局：window w 的输出由 w-1 的 Ca + w 的 Cb 合成
    - 内嵌 Lightning Indexer 选 top-k
    """
    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        # 内置 Indexer
        self.indexer = DeepseekV4Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[DeepseekV4Cache],
        layer_idx: int,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer = (
            past_key_values.layers[layer_idx] if past_key_values is not None else None
        )
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights(
                "compressor", kv, gate
            )

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + \
                         self.position_bias.to(chunk_gate.dtype)

            # overlap 布局（同 Indexer）
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full(
                (batch, n_windows, 2 * ratio, self.head_dim), float("-inf")
            )
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state(
                    "compressor", chunk_kv, chunk_gate, self.head_dim
                )
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            weights = new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)
            compressed = self.kv_norm((new_kv * weights).sum(dim=2))

            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)  # [B, 1, T, head_dim]

        # 用 Indexer 选 top-k，然后 gather
        topk = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)
        # topk: [B, S, k]
        # 把 compressed_kv 在 query 维度上展开
        expanded = compressed_kv.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)  # [B, 1, S, T, D]
        idx = topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, self.head_dim)
        gathered = torch.gather(expanded, 3, idx)  # [B, 1, S, k, D]
        # 注意：gather 出的 KV 是 per-query 的 top-k，我们 reshape 为 [B, 1, S*k, D] 让 attention 见到
        return gathered.reshape(batch, 1, -1, self.head_dim)


# ============================================================================
# 第八部分：MLA 注意力主模块
# ============================================================================

COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将单 KV 头复制到 n_rep 个头（MQA → MHA 广播）"""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class DeepseekV4Attention(nn.Module):
    """
    V4 多头潜在注意力（MLA）。

    关键设计：
    1. Shared-KV MQA：num_key_value_heads = 1，所有头共享单 KV 头
    2. Partial Rotary：仅 head_dim 的最后 rope_head_dim 个通道做 RoPE
    3. Per-head learnable attention sink（gpt-oss 风格）
    4. 分组低秩输出投影：先按 o_groups 分组各自 LoRA，再合并 mix 到 hidden_size
    5. 三种 cache 机制：sliding / sliding+CSA / sliding+HCA
    6. 输出端反向 RoPE：V 带了 RoPE，输出端用 -sin 撤销
    """
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        # 单 KV 头需广播到所有头
        self.num_key_value_groups = config.num_attention_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim ** -0.5

        # ----- Q 路径（LoRA） -----
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        # q_b_norm：在 reshape 后做 per-head RMSNorm（无可学习权重）
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)

        # ----- KV 路径（单头） -----
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # ----- 输出投影（分组 LoRA） -----
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(
            config.o_groups * config.o_lora_rank, config.hidden_size, bias=False
        )

        # ----- per-head attention sink -----
        self.sinks = nn.Parameter(torch.empty(self.num_heads))

        # ----- 压缩器（按层类型选择） -----
        compressor_cls = COMPRESSOR_CLASSES[self.layer_type]
        self.compressor = compressor_cls(config) if compressor_cls is not None else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[DeepseekV4Cache] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]  # (B, S)
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        # ===== Q 路径 =====
        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)  # [B, H, S, D]
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        # ===== KV 路径（单头） =====
        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        # 由于 num_key_value_heads=1，view 后形状为 [B, 1, S, head_dim]
        kv = apply_rotary_pos_emb(kv, cos, sin)

        # ===== 更新滑动窗口 cache，K==V 共享 =====
        if past_key_values is not None:
            kv = past_key_values.layers[self.layer_idx].update(kv, kv)[0]

        # ===== 拼接压缩 KV =====
        if self.compressor is not None:
            compressed_kv = self.compressor(
                hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
            )
            kv = torch.cat([kv, compressed_kv], dim=2)

        # ===== 右填充 mask 以覆盖压缩 KV（压缩 KV 不掩盖，pad 0） =====
        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            attention_mask = F.pad(
                attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0
            )

        # ===== Eager 注意力 =====
        key_states = repeat_kv(kv, self.num_key_value_groups)
        value_states = repeat_kv(kv, self.num_key_value_groups)
        attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # ===== Attention Sink：拼接一个额外 logit 再 softmax，丢弃 sink 列 =====
        sinks = self.sinks.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)
        combined = torch.cat([attn_weights, sinks], dim=-1)
        # 减去 max 防止 BF16/FP16 溢出
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
        scores = probs[..., :-1]   # 丢弃 sink
        scores = F.dropout(scores, p=self.attention_dropout, training=self.training)
        scores = scores.to(value_states.dtype)

        attn_output = torch.matmul(scores, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S, H, D]

        # ===== 输出端反向 RoPE：V 因 K==V 带有 RoPE，输出端用 -sin 撤销 =====
        attn_output = apply_rotary_pos_emb(
            attn_output.transpose(1, 2), cos, -sin
        ).transpose(1, 2)

        # ===== 分组输出投影 =====
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output


# ============================================================================
# 第九部分：Hyper-Connections（mHC）
# ============================================================================

class DeepseekV4HyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC，论文 §2.2)。

    将 hc_mult 个并行残差流通过学习到的双随机矩阵进行混合，
    保证信号传播的非扩张性（non-expansive）。

    工作流：
        hidden_streams [B, S, hc_mult, hidden]
            ↓ flatten + RMSNorm
            ↓ F.linear(fn) → mix-logits [B, S, (2+hc)*hc]
            ↓ 分拆为 (pre, post, comb) 三个输出
            ↓ comb 经 Sinkhorn 投影到双随机矩阵
        return (post, comb, collapsed = pre · hidden_streams)
    """
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        # (2 + hc) * hc 维：hc(pre) + hc(post) + hc*hc(comb)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # 3 个独立 scale：分别对应 pre / post / comb
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor):
        """
        Args:
            hidden_streams: [B, S, hc_mult, hidden_size]
        Returns:
            post: [B, S, hc_mult]
            comb: [B, S, hc_mult, hc_mult]
            collapsed: [B, S, hidden_size]
        """
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mix = F.linear(flat, self.fn.float())  # [B, S, (2+hc)*hc]
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        hc = self.hc_mult

        # 拆分三个输出
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc]) + self.hc_eps
        post = torch.sigmoid(mix[..., hc:2 * hc] * post_scale + self.base[hc:2 * hc]) + self.hc_eps
        comb = torch.sigmoid(
            mix[..., 2 * hc:].view(*mix.shape[:-1], hc, hc) * comb_scale +
            self.base[2 * hc:].view(hc, hc)
        ) + self.hc_eps

        # Sinkhorn-Knopp 投影到双随机矩阵
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

        # pre 加权求和把 hc_mult 流合并为单流
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class DeepseekV4HyperHead(nn.Module):
    """模型最后的 HC 收缩头：把 hc_mult 流合并成单流（位于最终 RMSNorm 之前）。"""
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, hc_mult, hidden]
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())  # [B, S, hc_mult]
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


# ============================================================================
# 第十部分：MoE 模块
# ============================================================================

class DeepseekV4MLP(nn.Module):
    """SwiGLU MLP（用于共享专家，无 swiglu clamp）。"""
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV4Experts(nn.Module):
    """
    路由专家集合：权重打包为 3D Parameter 便于 indexing。

    每个 expert 是 SwiGLU FFN，注意带 clamp（swiglu_limit）。
    """
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        # gate_up 把 gate 和 up 合并存储（前半 gate，后半 up）
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.limit = config.swiglu_limit

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        """SwiGLU with clamp：gate.clamp(max=lim), up.clamp(±lim), silu(gate)*up"""
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return F.silu(gate) * up

    def forward(
        self,
        hidden_states: torch.Tensor,    # [N, hidden_dim]，N = batch*seq
        top_k_index: torch.Tensor,      # [N, top_k]
        top_k_weights: torch.Tensor,    # [N, top_k]
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        # 找到本步真正被命中的 expert
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            # mask: [num_experts, top_k, N]
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            # 前向：gate_up → SwiGLU → down，乘上路由权重
            gate_up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
            current = self._apply_gate(gate_up)
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[
                token_idx, top_k_pos, None
            ]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class DeepseekV4TopKRouter(nn.Module):
    """
    标准 Top-K 路由器：sqrtsoftplus 评分 + 偏置修正 + noaux_tc topk。

    e_score_correction_bias 是 buffer（无梯度），仅影响 topk 选择，不影响最终权重。
    """
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.scoring_func_name = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts), persistent=True,
        )

    def _score(self, logits: torch.Tensor) -> torch.Tensor:
        if self.scoring_func_name == "softmax":
            return F.softmax(logits, dim=-1)
        elif self.scoring_func_name == "sigmoid":
            return torch.sigmoid(logits)
        elif self.scoring_func_name == "sqrtsoftplus":
            return F.softplus(logits).sqrt()
        else:
            raise ValueError(f"Unknown scoring_func: {self.scoring_func_name}")

    def forward(self, hidden_states: torch.Tensor):
        flat = hidden_states.reshape(-1, self.hidden_dim)
        # 路由 logits 计算在 FP32
        logits = F.linear(flat.float(), self.weight.float())
        scores = self._score(logits)
        # noaux_tc：用 bias 影响 topk 选择
        indices = torch.topk(
            scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False
        ).indices
        weights = scores.gather(1, indices)
        # 归一化（如果开启）
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(nn.Module):
    """
    Hash 路由器（论文 §2.1）：前 num_hash_layers 个 MoE 层使用。

    专家选择由 tid2eid[input_ids] 决定（固定表，无梯度），但 router weight
    仍负责生成 per-expert 评分用于加权。
    """
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.scoring_func_name = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer(
            "tid2eid",
            torch.zeros(config.vocab_size, self.top_k, dtype=torch.long),
            persistent=True,
        )

    def _score(self, logits: torch.Tensor) -> torch.Tensor:
        if self.scoring_func_name == "softmax":
            return F.softmax(logits, dim=-1)
        elif self.scoring_func_name == "sigmoid":
            return torch.sigmoid(logits)
        elif self.scoring_func_name == "sqrtsoftplus":
            return F.softplus(logits).sqrt()
        raise ValueError(f"Unknown scoring_func: {self.scoring_func_name}")

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self._score(logits)
        # 查表得到固定的 expert indices
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock(nn.Module):
    """
    混合路由 MoE 块：
      - 前 num_hash_layers 层：hash routing
      - 其余层：sqrtsoftplus topk + noaux_tc
      - 始终包含 1 个共享专家
    """
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


# ============================================================================
# 第十一部分：Decoder Layer
# ============================================================================

class DeepseekV4DecoderLayer(nn.Module):
    """
    V4 Decoder Block：HC + Self-Attention + HC + MoE。

    与标准 Transformer 不同：残差是 hc_mult 并行流，通过 mHC 模块在子层前后
    做 collapse / expand。
    """
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 两个独立的 HC 模块（attention 侧 / mlp 侧）
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, S, hc_mult, hidden]
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[DeepseekV4Cache] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype

        # ===== Attention 子层 =====
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output = self.self_attn(
            self.input_layernorm(collapsed),
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        # post 把子层输出放回各流，comb 在流间混合
        hidden_states = (
            post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) +
            torch.matmul(comb.to(dtype), hidden_states)
        )

        # ===== MLP 子层 =====
        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        return (
            post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) +
            torch.matmul(comb.to(dtype), hidden_states)
        )


# ============================================================================
# 第十二部分：因果掩码构造
# ============================================================================

def create_sliding_window_causal_mask(
    config: DeepseekV4Config,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[DeepseekV4Cache] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    构造 [B, 1, S, kv_len] 滑动窗口因果掩码（用于 sliding-window KV，不含压缩 KV）。

    未来位置和窗口外位置赋值为 dtype 最小值，可见位置为 0。
    压缩 KV 的右填充由 attention 内部完成。
    """
    B, S = inputs_embeds.shape[:2]
    sw = config.sliding_window
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    min_val = torch.finfo(dtype).min
    past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

    if past_len == 0:
        # ----- Prefill 阶段 -----
        kv_len = S
        q_pos = torch.arange(S, device=device)
        k_pos = torch.arange(S, device=device)
    else:
        # ----- Decode 阶段 -----
        kv_history = min(past_len, sw - 1)
        kv_len = kv_history + S
        q_pos = torch.arange(past_len, past_len + S, device=device)
        k_pos = torch.arange(past_len - kv_history, past_len + S, device=device)

    # 因果：k > q
    causal = k_pos[None, :] > q_pos[:, None]
    # 窗口外：k < q - sw + 1
    oow = k_pos[None, :] < (q_pos[:, None] - sw + 1)
    invalid = causal | oow

    mask = torch.zeros(B, 1, S, kv_len, dtype=dtype, device=device)
    mask = mask.masked_fill(invalid[None, None], min_val)
    return mask


# ============================================================================
# 第十三部分：模型主体 + ForCausalLM
# ============================================================================

class DeepseekV4Model(nn.Module):
    """完整 V4 主干：embed → HC expand → N×decoder → HC head → norm"""
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            DeepseekV4DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.hc_head = DeepseekV4HyperHead(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DeepseekV4Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("必须正好提供 input_ids 和 inputs_embeds 之一")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None and (use_cache if use_cache is not None else self.config.use_cache):
            past_key_values = DeepseekV4Cache(self.config)

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ) + past_seen
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_sliding_window_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # 扩展到 hc_mult 流：[B, S, hidden] → [B, S, hc_mult, hidden]
        hidden_states = inputs_embeds.unsqueeze(2).expand(
            -1, -1, self.config.hc_mult, -1
        ).contiguous()
        # 主路径 RoPE
        position_embeddings = self.rotary_emb(
            inputs_embeds, position_ids=position_ids, layer_type="main"
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                input_ids=input_ids,
            )

        # 最终 HC 收缩 + RMSNorm
        hidden_states = self.norm(self.hc_head(hidden_states))
        return hidden_states, past_key_values


class DeepseekV4ForCausalLM(nn.Module):
    """V4 + LM Head（语言建模）"""
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.model = DeepseekV4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 可选权重绑定
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DeepseekV4Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Dict[str, Any]:
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        # 只计算需要的 logits
        slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits, "past_key_values": past_key_values}

    # ----------------------------------------------------------------------
    # 加载 HuggingFace checkpoint
    # ----------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Union[str, torch.device] = "cpu",
        strict: bool = False,
    ) -> "DeepseekV4ForCausalLM":
        """
        从 HuggingFace safetensors checkpoint 加载。

        Args:
            model_path: 包含 config.json 和 *.safetensors 的目录
            torch_dtype: 模型权重 dtype（默认 bfloat16）
            device_map: 加载到哪个 device
            strict: 是否严格匹配 keys（默认 False，忽略 mtp.* 等）
        """
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("加载 HF checkpoint 需要 safetensors 库：pip install safetensors")

        # 加载 config
        config_path = os.path.join(model_path, "config.json")
        config = DeepseekV4Config.from_json_file(config_path)

        # 构建模型骨架
        model = cls(config)

        # 加载权重
        state_dict: Dict[str, torch.Tensor] = {}
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"未找到 *.safetensors 在 {model_path}")
        for fp in files:
            with safe_open(fp, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # 忽略 MTP 权重（V4 主模型不需要）
                    if key.startswith("mtp.") or ".mtp." in key:
                        continue
                    state_dict[key] = f.get_tensor(key)

        # 可能的命名修正（attn_sink ↔ sinks 等）
        new_sd: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            # 把 inference/原生命名转换到 transformers 命名
            nk = k
            nk = nk.replace(".attn_sink", ".sinks")
            new_sd[nk] = v

        missing, unexpected = model.load_state_dict(new_sd, strict=strict)
        print(f"[from_pretrained] loaded. missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  (前 10 个 missing) {missing[:10]}")
        if unexpected:
            print(f"  (前 10 个 unexpected) {unexpected[:10]}")

        return model.to(torch_dtype).to(device_map)

    # ----------------------------------------------------------------------
    # 权重初始化（用于从头训练）
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def init_weights(self):
        """对所有 Parameter 做标准初始化（用于从头训练）"""
        std = self.config.initializer_range
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.normal_(mean=0.0, std=std)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.normal_(mean=0.0, std=std)
            elif isinstance(m, DeepseekV4TopKRouter):
                m.weight.normal_(mean=0.0, std=std)
                m.e_score_correction_bias.zero_()
            elif isinstance(m, DeepseekV4HashRouter):
                m.weight.normal_(mean=0.0, std=std)
                m.tid2eid.zero_()
            elif isinstance(m, DeepseekV4Experts):
                m.gate_up_proj.normal_(mean=0.0, std=std)
                m.down_proj.normal_(mean=0.0, std=std)
            elif isinstance(m, DeepseekV4Attention):
                m.sinks.zero_()
            elif isinstance(m, DeepseekV4HyperConnection):
                m.fn.normal_(mean=0.0, std=std)
                m.base.zero_()
                m.scale.fill_(1.0)
            elif isinstance(m, DeepseekV4HyperHead):
                m.hc_fn.normal_(mean=0.0, std=std)
                m.hc_base.zero_()
                m.hc_scale.fill_(1.0)
            elif isinstance(m, (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer)):
                m.position_bias.zero_()
            elif isinstance(m, DeepseekV4RMSNorm):
                m.weight.fill_(1.0)


# ============================================================================
# 第十四部分：配置工厂函数（完整版 / 2B mini 版）
# ============================================================================

def get_full_config() -> DeepseekV4Config:
    """
    DeepSeek-V4-Pro 完整版配置（约 671B 总参数 / 37B 激活）。
    对齐官方 config.json。
    """
    return DeepseekV4Config(
        vocab_size=129280,
        hidden_size=7168,
        num_hidden_layers=61,
        num_attention_heads=128,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1536,
        o_lora_rank=1024,
        o_groups=16,
        moe_intermediate_size=3072,
        n_routed_experts=384,
        n_shared_experts=1,
        num_experts_per_tok=6,
        num_hash_layers=3,
        sliding_window=128,
        max_position_embeddings=1048576,
        swiglu_limit=10.0,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=1024,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        },
        compress_ratios=(
            128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
            4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
            4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
            4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 0,
        ),
    )


def get_mini_config() -> DeepseekV4Config:
    """
    2B Mini 版（约 1.9B 总参数 / 约 700M 激活）。

    保留 V4 全部结构特性，仅缩放规模便于本地开发与单机推理：
      - 12 层（含 6 HCA / 5 CSA / 1 Sliding）
      - 32 个路由专家（top-4），1 个共享专家
      - 前 1 层 hash routing
      - hidden_size = 1536, head_dim = 128, hc_mult 仍为 4
    """
    return DeepseekV4Config(
        vocab_size=129280,
        hidden_size=1536,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=1,
        head_dim=128,
        qk_rope_head_dim=32,
        q_lora_rank=768,
        o_lora_rank=384,
        o_groups=4,
        moe_intermediate_size=768,
        n_routed_experts=32,
        n_shared_experts=1,
        num_experts_per_tok=4,
        num_hash_layers=1,
        sliding_window=64,
        max_position_embeddings=4096,
        swiglu_limit=10.0,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        index_n_heads=8,
        index_head_dim=64,
        index_topk=128,
        hc_mult=4,
        hc_sinkhorn_iters=10,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        rope_scaling={
            "beta_fast": 32, "beta_slow": 1, "factor": 4,
            "original_max_position_embeddings": 1024, "type": "yarn",
        },
        # 12 层：[HCA, HCA, CSA, HCA, CSA, HCA, CSA, HCA, CSA, HCA, CSA, Sliding]
        compress_ratios=(128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0),
    )


# ============================================================================
# 模型规模分析工具（按模块分类，估算激活参数与存储）
# ============================================================================

def categorize_param(name: str) -> str:
    """根据 parameter 全名分类，按"特定优先"原则匹配。"""
    if "embed_tokens" in name:
        return "Embedding"
    if "lm_head" in name:
        return "LM Head"
    # —— 注意力子树需先匹配最特定的子模块 ——
    if ".self_attn.compressor.indexer." in name:
        return "Indexer (CSA only)"
    if ".self_attn.compressor." in name:
        return "Compressor"
    if ".self_attn." in name:
        return "Self-Attention (MLA)"
    # —— MoE 子树 ——
    if ".mlp.experts." in name:
        return "Routed Experts"
    if ".mlp.shared_experts." in name:
        return "Shared Experts"
    if ".mlp.gate." in name:
        return "MoE Router"
    # —— Hyper-Connections ——
    if ".attn_hc." in name or ".ffn_hc." in name:
        return "HC (per-layer)"
    if "model.hc_head." in name:
        return "HC Head"
    # —— 归一化 ——
    if ".input_layernorm." in name or ".post_attention_layernorm." in name:
        return "RMSNorm (per-layer)"
    if name == "model.norm.weight":
        return "RMSNorm (final)"
    return "Other"


def fmt(n: int) -> str:
    """格式化大数字：B / M / K"""
    if n >= 1e9:
        return f"{n/1e9:>8.3f} B"
    if n >= 1e6:
        return f"{n/1e6:>8.3f} M"
    if n >= 1e3:
        return f"{n/1e3:>8.3f} K"
    return f"{n:>8d}  "


def analyze(model: nn.Module, label: str, config: DeepseekV4Config) -> int:
    """
    统计模型规模并打印详细报告：
      - 架构概览
      - 层类型分布
      - 按模块分类的参数量（含百分比）
      - 激活参数估算
      - 各精度下的存储估算（含 V4 官方 FP8+FP4 混合）
    """
    cats: Dict[str, int] = {}
    total = 0
    for n, p in model.named_parameters():
        cat = categorize_param(n)
        cats[cat] = cats.get(cat, 0) + p.numel()
        total += p.numel()

    print("\n" + "=" * 76)
    print(f" {label}")
    print("=" * 76)

    # ----- 架构 -----
    print(f"\n  [架构]")
    print(f"    hidden_size           = {config.hidden_size:,}")
    print(f"    num_hidden_layers     = {config.num_hidden_layers}")
    print(f"    num_attention_heads   = {config.num_attention_heads}")
    print(f"    head_dim / rope_dim   = {config.head_dim} / {config.qk_rope_head_dim}")
    print(f"    q_lora_rank           = {config.q_lora_rank}")
    print(f"    o_lora_rank / o_grps  = {config.o_lora_rank} / {config.o_groups}")
    print(f"    n_routed_experts      = {config.n_routed_experts}  "
          f"(top-{config.num_experts_per_tok}, "
          f"{config.num_experts_per_tok/config.n_routed_experts*100:.2f}% 激活)")
    print(f"    moe_intermediate_size = {config.moe_intermediate_size}")
    print(f"    num_hash_layers       = {config.num_hash_layers}")
    print(f"    sliding_window        = {config.sliding_window}")

    sliding = sum(1 for t in config.layer_types if t == "sliding_attention")
    csa = sum(1 for t in config.layer_types if t == "compressed_sparse_attention")
    hca = sum(1 for t in config.layer_types if t == "heavily_compressed_attention")
    print(f"\n  [层类型]  HCA(128) = {hca}   CSA(4) = {csa}   Sliding(0) = {sliding}")

    # ----- 参数分类 -----
    print(f"\n  [参数分布]")
    print(f"  {'-' * 70}")
    order = [
        "Embedding", "LM Head",
        "Self-Attention (MLA)", "Compressor", "Indexer (CSA only)",
        "Routed Experts", "Shared Experts", "MoE Router",
        "HC (per-layer)", "HC Head",
        "RMSNorm (per-layer)", "RMSNorm (final)", "Other",
    ]
    for cat in order:
        if cat in cats and cats[cat] > 0:
            n = cats[cat]
            pct = 100.0 * n / total
            print(f"    {cat:30s}: {fmt(n)}   ({pct:>5.2f}%)")
    print(f"  {'-' * 70}")
    print(f"    {'TOTAL':30s}: {fmt(total)}   ({total:,})")

    # ----- 激活参数 -----
    routed = cats.get("Routed Experts", 0)
    if routed > 0:
        ratio = config.num_experts_per_tok / config.n_routed_experts
        activated_routed = int(routed * ratio)
        activated_total = total - routed + activated_routed
        print(f"\n  [激活参数] (per token, top-{config.num_experts_per_tok}/"
              f"{config.n_routed_experts} = {ratio*100:.2f}%)")
        print(f"    {fmt(activated_total)}   "
              f"({activated_total/total*100:.2f}% of total)")

    # ----- 存储估算 -----
    print(f"\n  [存储估算]")
    for dt, b in [("FP32", 4), ("BF16/FP16", 2), ("FP8", 1)]:
        gb = total * b / 1e9
        print(f"    {dt:25s}: {gb:>10.2f} GB")
    if routed > 0:
        # V4 官方部署：FP8 主路径 + FP4 专家
        v4_size = ((total - routed) * 1 + routed * 0.5) / 1e9
        print(f"    FP8 + FP4 (V4 official) : {v4_size:>10.2f} GB  ← 官方混合精度")

    return total


# ============================================================================
# 第十五部分：使用示例
# ============================================================================

if __name__ == "__main__":
    # ----- 示例 1：mini 模型从头构造 -----
    print("=" * 60)
    print("示例 1：构造 2B mini 版本")
    print("=" * 60)
    config = get_mini_config()
    print(f"层数: {config.num_hidden_layers}")
    print(f"层类型: {config.layer_types}")
    print(f"MoE 类型: {config.mlp_layer_types}")
    print(f"intermediate_size: {config.intermediate_size}")

    model = DeepseekV4ForCausalLM(config)
    model.init_weights()

    # 统计参数量
    total = sum(p.numel() for p in model.parameters())
    print([p.numel() for p in model.parameters()])
    print(f"\n总参数: {total / 1e9:.2f}B")
    print(model)

    # ----- 示例 2：前向传播 -----
    print("\n" + "=" * 60)
    print("示例 2：mini 模型前向传播")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(torch.float32).to(device)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    print(f"input shape: {input_ids.shape}")
    print(f"logits shape: {out['logits'].shape}")

    # ----- 示例 3：自回归生成（greedy） -----
    print("\n" + "=" * 60)
    print("示例 3：mini 模型 greedy 解码")
    print("=" * 60)
    past = None
    cur = input_ids
    with torch.no_grad():
        for step in range(5):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out["past_key_values"]
            next_token = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
            cur = next_token
            print(f"  step {step}: next_token = {next_token.item()}")

    # ----- 示例 4：完整版构造（仅查看配置，不实例化） -----
    print("\n" + "=" * 60)
    print("示例 4：完整 V4-Pro 配置概览")
    print("=" * 60)
    full = get_full_config()
    print(f"hidden_size: {full.hidden_size}")
    print(f"num_hidden_layers: {full.num_hidden_layers}")
    print(f"num_attention_heads: {full.num_attention_heads}")
    print(f"n_routed_experts: {full.n_routed_experts}")
    print(f"num_experts_per_tok: {full.num_experts_per_tok}")
    print(f"sliding_window: {full.sliding_window}")
    print(f"head_dim: {full.head_dim} (rope: {full.qk_rope_head_dim})")
    print(f"q_lora_rank: {full.q_lora_rank} / o_lora_rank: {full.o_lora_rank}")

    # ----- 示例 5：加载 HF checkpoint（伪代码） -----
    print("\n" + "=" * 60)
    print("示例 5：加载 HF checkpoint 用法")
    print("=" * 60)
    print("""
    # 假设你已下载完整 DeepSeek-V4 仓库到 /path/to/DeepSeek-V4-Pro
    model = DeepseekV4ForCausalLM.from_pretrained(
        "/path/to/DeepSeek-V4-Pro",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    """)
