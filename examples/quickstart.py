"""
30 秒 quickstart：构造 mini 模型，跑一次前向 + greedy 解码。
"""
import torch

from deepseek_v4.modeling.model import DeepseekV4ForCausalLM, get_mini_config

cfg = get_mini_config()
model = DeepseekV4ForCausalLM(cfg)
model.init_weights()
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 假设有 32 个随机 token
input_ids = torch.randint(0, cfg.vocab_size, (1, 32), device=device)

# Greedy decode 5 步
past = None
cur = input_ids
generated = []
with torch.no_grad():
    for _ in range(5):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out["past_key_values"]
        next_token = out["logits"][:, -1].argmax(-1, keepdim=True)
        generated.append(next_token.item())
        cur = next_token

print("Generated:", generated)
