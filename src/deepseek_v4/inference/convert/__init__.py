"""
模型格式转换工具。

支持目标：
- HF transformers 兼容目录（含 config.json + 分片 safetensors + tokenizer 文件）
- llama.cpp GGUF（best-effort，要求安装 llama-cpp-python 或本地 llama.cpp）
- vLLM 加载目录（与 HF 一致，仅添加 `trust_remote_code` 注释）
- Ollama Modelfile + 自动 import
"""
from deepseek_v4.inference.convert.safetensors_utils import (
    save_sharded_safetensors, load_sharded_safetensors,
)
from deepseek_v4.inference.convert.to_hf import export_to_hf
from deepseek_v4.inference.convert.to_gguf import export_to_gguf
from deepseek_v4.inference.convert.to_vllm import export_to_vllm
from deepseek_v4.inference.convert.to_ollama import build_ollama_modelfile, export_to_ollama

__all__ = [
    "save_sharded_safetensors", "load_sharded_safetensors",
    "export_to_hf",
    "export_to_gguf",
    "export_to_vllm",
    "build_ollama_modelfile", "export_to_ollama",
]
