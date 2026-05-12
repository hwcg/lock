"""
导出 vLLM 兼容目录。

vLLM 与 HF transformers 共用目录格式。区别：
1. 需要保证 trust_remote_code 流程能找到模型代码
2. 需要 `config.json` 内 `architectures` 指向正确类名
3. 若 vLLM 已有官方 deepseek-v4 支持，则不需要 trust_remote_code

这里我们生成与 `to_hf` 相同的目录，但额外附 USAGE.md。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from deepseek_v4.inference.convert.to_hf import export_to_hf
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


VLLM_USAGE = """\
# Loading with vLLM

## Quick start
​```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \\
    --model {dir} \\
    --trust-remote-code \\
    --dtype bfloat16 \\
    --max-model-len 4096
```

## Python

```
from vllm import LLM, SamplingParams
llm = LLM(model="{dir}", trust_remote_code=True, dtype="bfloat16", max_model_len=4096)
out = llm.generate(["Hello"], SamplingParams(max_tokens=32))
print(out[0].outputs[0].text)
```

"""

def export_to_vllm(
state_dict_dir: Union[str, Path],
model_config_path: Union[str, Path],
tokenizer_dir: Union[str, Path],
output_dir: Union[str, Path],
max_shard_size: str = "5GB",
dtype: Optional[torch.dtype] = None,
) -> Path:
out = export_to_hf(
state_dict_dir=state_dict_dir,
model_config_path=model_config_path,
tokenizer_dir=tokenizer_dir,
output_dir=output_dir,
max_shard_size=max_shard_size,
dtype=dtype,
)
(out / "USAGE_VLLM.md").write_text(VLLM_USAGE.format(dir=out.resolve()), encoding="utf-8")
logger.info(f"[to_vllm] ✅ ready for vLLM at {out}")
return out
