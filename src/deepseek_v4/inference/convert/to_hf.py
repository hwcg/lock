"""
导出 HuggingFace 兼容目录。

产物：
    output_dir/
        config.json                              # DeepseekV4Config（含 auto_map）
        configuration_deepseek_v4.py             # config 自定义类
        modeling_deepseek_v4.py                  # 模型自定义类（trust_remote_code）
        tokenizer_config.json                    # tokenizer 配置（auto_map）
        tokenization_deepseek_v4.py              # tokenizer 自定义类
        vocab.json
        merges.txt
        special_tokens_map.json
        model.safetensors.index.json
        model-00001-of-NNNNN.safetensors
        ...
        generation_config.json                   # 推理默认值
        README.md                                # 自动生成

这样可以直接 `AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)` 加载。
"""
from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from deepseek_v4.inference.convert.safetensors_utils import (
    load_sharded_safetensors, save_sharded_safetensors,
)
from deepseek_v4.tokenizer.tokenizer import DeepseekV4Tokenizer
from deepseek_v4.utils.io import safe_save_json
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


HF_CONFIG_TEMPLATE = '''"""
HuggingFace AutoConfig 入口（trust_remote_code）。
"""
from transformers import PretrainedConfig

class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
'''


HF_MODELING_TEMPLATE = '''"""
HuggingFace AutoModel 入口（trust_remote_code）。

加载我们自定义的 DeepseekV4ForCausalLM。
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

# 假设用户已经 pip install 了本仓库：from deepseek_v4 ...
try:
    from deepseek_v4.modeling.model import (
        DeepseekV4Config as _Cfg,
        DeepseekV4ForCausalLM as DeepseekV4ForCausalLM,
        DeepseekV4Model as DeepseekV4Model,
    )
except ImportError:
    raise ImportError(
        "请先 pip install deepseek-v4-mini 或把本仓库加入 PYTHONPATH"
    )
'''


HF_TOK_TEMPLATE = '''"""
HuggingFace AutoTokenizer 入口（trust_remote_code）。
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from deepseek_v4.tokenizer.tokenizer import DeepseekV4Tokenizer
'''


def export_to_hf(
    state_dict_dir: Union[str, Path],
    model_config_path: Union[str, Path],
    tokenizer_dir: Union[str, Path],
    output_dir: Union[str, Path],
    max_shard_size: str = "5GB",
    dtype: Optional[torch.dtype] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    把训练产物导出为 HuggingFace transformers 兼容目录。

    Args:
        state_dict_dir:     训练保存目录（含 model.safetensors[.index.json]）
        model_config_path:  configs/model/mini_2b.json 风格的纯 dataclass dump
        tokenizer_dir:      已 save_pretrained 的 tokenizer 目录
        output_dir:         目标
        max_shard_size:     "5GB" / "10GB"
        dtype:              强制 cast（如 torch.bfloat16）
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. config ---
    with open(model_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["model_type"] = "deepseek_v4"
    cfg["architectures"] = ["DeepseekV4ForCausalLM"]
    cfg["auto_map"] = {
        "AutoConfig": "configuration_deepseek_v4.DeepseekV4Config",
        "AutoModel": "modeling_deepseek_v4.DeepseekV4Model",
        "AutoModelForCausalLM": "modeling_deepseek_v4.DeepseekV4ForCausalLM",
    }
    if extra_config:
        cfg.update(extra_config)
    safe_save_json(out / "config.json", cfg)

    # --- 2. 模型代码 ---
    (out / "configuration_deepseek_v4.py").write_text(HF_CONFIG_TEMPLATE, encoding="utf-8")
    (out / "modeling_deepseek_v4.py").write_text(HF_MODELING_TEMPLATE, encoding="utf-8")

    # --- 3. tokenizer ---
    tokenizer = DeepseekV4Tokenizer.from_pretrained(str(tokenizer_dir))
    tokenizer.save_pretrained(str(out))
    (out / "tokenization_deepseek_v4.py").write_text(HF_TOK_TEMPLATE, encoding="utf-8")
    # 给 tokenizer_config.json 补 auto_map（save_pretrained 已写了，这里更新一下）
    tok_cfg_path = out / "tokenizer_config.json"
    if tok_cfg_path.exists():
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        tok_cfg["auto_map"] = {
            "AutoTokenizer": ["tokenization_deepseek_v4.DeepseekV4Tokenizer", None],
        }
        safe_save_json(tok_cfg_path, tok_cfg)

    # --- 4. weights ---
    logger.info(f"[to_hf] loading state_dict from {state_dict_dir}")
    state_dict = load_sharded_safetensors(state_dict_dir)
    logger.info(f"[to_hf] {len(state_dict)} params, saving to {out}")
    save_sharded_safetensors(state_dict, out, max_shard_size=max_shard_size, dtype=dtype)

    # --- 5. generation_config.json ---
    safe_save_json(out / "generation_config.json", {
        "bos_token_id": cfg.get("bos_token_id", 0),
        "eos_token_id": cfg.get("eos_token_id", 1),
        "pad_token_id": cfg.get("pad_token_id", 2),
        "transformers_version": "auto",
        "do_sample": True, "temperature": 0.7, "top_p": 1.0,
    })

    # --- 6. README ---
    readme = textwrap.dedent(f"""
        # DeepSeek-V4-Mini (HF Export)

        Auto-generated from training checkpoint.

        Load via:

        ```python
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("{out}", trust_remote_code=True, torch_dtype="bfloat16", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("{out}", trust_remote_code=True)
        ```

        Note: requires `pip install deepseek-v4-mini` (or this repo on PYTHONPATH).
    """).strip()
    (out / "README.md").write_text(readme, encoding="utf-8")
    logger.info(f"[to_hf] ✅ exported to {out}")
    return out
