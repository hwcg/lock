# 架构详解

## 全局视图
```
         Input IDs [B, S]
                │
        embed_tokens (vocab=129K)
                │
        ┌───────── HC expand → [B, S, hc_mult, hidden] ─────────┐
        │                                                      │
        │   for layer in layers:                                │
        │       hidden_streams                                  │
        │            ↓                                          │
        │     attn_hc.collapse → input_layernorm                │
        │            ↓                                          │
        │   ┌───────────────────────────────┐                   │
        │   │  Self-Attention (MLA)         │                   │
        │   │  - Q LoRA (1536→128*512)      │                   │
        │   │  - shared-KV (1 head)         │                   │
        │   │  - per-head sink              │                   │
        │   │  - partial RoPE (last 64 dim) │                   │
        │   │  - Compressor (HCA/CSA/none)  │                   │
        │   │  - Grouped LoRA output        │                   │
        │   └───────────────────────────────┘                   │
        │            ↓                                          │
        │  attn_hc.expand → MoE                                 │
        │            ↓                                          │
        │   ┌───────────────────────────────┐                   │
        │   │  Sparse MoE Block             │                   │
        │   │  - hash router (前 N 层)      │                   │
        │   │  - sqrtsoftplus topk          │                   │
        │   │  - 384 experts, top-6         │                   │
        │   │  - 1 shared expert            │                   │
        │   └───────────────────────────────┘                   │
        └───────────────────────────────────────────────────────┘
                        │
                hc_head.collapse
                        │
                   RMSNorm
                        │
                   lm_head
                        │
                Logits [B, S, V]
```

## 模块详解

### 1. MLA (Multi-Latent Attention)

**Q 路径（LoRA）**
```
hidden [B, S, 7168]
→ q_a_proj [7168 → 1536] (Q LoRA bottleneck)
→ q_a_norm (RMSNorm)
→ q_b_proj [1536 → 128 * 512]
→ reshape to [B, S, 128, 512]
→ q_b_norm (per-head RMSNorm, no weight)
→ apply RoPE on last 64 dims of each head

```
**KV 路径（共享 KV，单头）**
```
hidden → kv_proj [7168 → 512]
→ kv_norm
→ reshape to [B, S, 1, 512]
→ apply RoPE on last 64 dims
→ broadcast to 128 heads at attention time

```
**为什么这么设计？**
- Q LoRA 让大模型保持 expressiveness，参数省 5x
- 单 KV 头（MQA） + 头间广播 → KV cache 大小 / 128
- per-head sink：模仿 OpenAI gpt-oss，让某些 head 学会"啥都不关注"，提升稳定性
- partial RoPE：前 nope 部分提供 content lookup，后 rope 部分提供位置信息

**输出投影（分组 LoRA）**
```
attn_out [B, S, 128, 512] reshape→ [B, S, 16, 4096]
→ o_a_proj (块对角分组：每 group 独立 [4096 → 1024])
→ flatten → [B, S, 16384]
→ o_b_proj [16384 → 7168]

```

### 2. mHC (Manifold-constrained Hyper-Connections)

灵感来自 ResNet-style multi-stream，但用 Sinkhorn 双随机矩阵保证 non-expansive。
```
hidden_streams [B, S, hc_mult, H]
│
flatten + RMSNorm
│
F.linear(fn) → [B, S, (2 + hc_mult) * hc_mult]
│
分拆三块：
pre_scale [B, S, hc_mult] sigmoid → 流权重
post_scale [B, S, hc_mult] sigmoid → 子层输出权重
comb_scale [B, S, hc_mult, hc_mult] Sinkhorn(双随机) → 流间混合
│
collapsed = sum(pre · streams, dim=2) # 单流喂给子层
↓
子层输出 sub_out
↓
new_streams = post · sub_out + comb @ streams

```
为什么需要 mHC？
- 标准 residual 当层数 >50 容易 collapse
- Sinkhorn 投影把 comb 约束在双随机矩阵簇，等价于一组随机置换的凸组合 → 严格非扩张
- 经验上 4 个流 (hc_mult=4) 在不显著增加 FLOPs 下显著改善优化

### 3. CSA / HCA Compressor（KV 压缩）

V4 把 61 层分为三类：
- 1 层 sliding：标准滑动窗口注意力（最后一层）
- 30 层 CSA (compress_rate=4)：内嵌 Lightning Indexer，做 sparse 选择
- 30 层 HCA (compress_rate=128)：纯长程压缩

**HCA 工作流**
```
每 128 个源 token → 1 个压缩 KV entry

1. kv_proj + gate_proj
2. 加可学习位置偏置
3. 窗口内 softmax 加权求和
4. RMSNorm
5. 在窗口"代表位置" (i*128) 应用 RoPE
   → 压缩 KV 拼接到滑动窗口 KV 后参与 attention

```
**CSA 工作流（更复杂）**
- compress_rate=4，窗口更小，覆盖密集中程依赖
- **overlap 布局**：每个窗口的输出由 (上一窗口 Ca + 本窗口 Cb) softmax 合成
- 内嵌 **Lightning Indexer**：为每个 query 选 top-1024 个压缩 KV，避免 O(N²)

### 4. MoE 路由
```
前 num_hash_layers 层（前 3 层）：
DeepseekV4HashRouter
\- tid2eid: [vocab_size, top_k] 固定查表
\- score 仍由 router weight 计算（用于加权）
\- expert 选择确定，没有 routing loss

其余层：
DeepseekV4TopKRouter
\- logits = hidden @ router_weight.T
\- score = sqrtsoftplus(logits)
\- topk(score + bias)，bias 不参梯度（noaux_tc）
\- 384 experts × top-6 = 1.5% 激活率

```

### 5. YaRN RoPE
```
inv_freq[i] = base^(-2i/d)

YaRN 在 [low, high] 频段做线性过渡：
low, high = find_correction_range(beta_fast=32, beta_slow=1, ...)
extra_factor = 1 - linear_ramp(low, high)
inv_freq = inv_freq_inter * (1 - extra_factor) + inv_freq_extra * extra_factor

其中：
inv_freq_inter = inv_freq / s (内插，扩展范围)
inv_freq_extra = inv_freq (外推，保留高频)

```
实测：
- 65K 训练 → 1M 推理无需重训
- 仅需修改 `rope_scaling.factor` 和 `original_max_position_embeddings`，重算 `inv_freq` buffer
