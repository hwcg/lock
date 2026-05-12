"""
服务端配置。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from deepseek_v4.utils.config import BaseConfig


@dataclass
class ServerConfig(BaseConfig):
    """OpenAI 兼容服务端配置。"""

    # ---------- 模型 ----------
    model_path: str = "checkpoints/sft/checkpoint-final"
    tokenizer_path: str = "checkpoints/tokenizer"
    model_name: str = "deepseek-v4-mini"        # 对外暴露的 model id
    model_aliases: List[str] = field(default_factory=list)

    device: str = "cuda"
    dtype: str = "bfloat16"                     # fp32 | fp16 | bf16
    max_model_len: int = 4096
    yarn_factor: Optional[float] = None         # 推理时一键启用 YaRN

    # ---------- HTTP ----------
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: Optional[str] = None               # None=禁用鉴权
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    request_timeout: float = 600.0

    # ---------- 批处理（continuous batching 简化版） ----------
    max_batch_size: int = 8
    batch_wait_ms: int = 8                      # 凑批最大等待
    max_concurrent_requests: int = 64

    # ---------- 生成默认值 ----------
    default_max_tokens: int = 1024
    default_temperature: float = 0.7
    default_top_p: float = 1.0
    default_top_k: int = 0
    default_repetition_penalty: float = 1.0

    # ---------- Feature flags ----------
    enable_chat_completions: bool = True
    enable_completions: bool = True
    enable_embeddings: bool = False
    enable_models: bool = True
    enable_health: bool = True

    # 开放思考切换（接受 open_thinking / enable_thinking / thinking_mode 参数）
    default_thinking_mode: str = "auto"          # auto | chat | thinking
    default_reasoning_effort: Optional[str] = None
    drop_thinking_in_context: bool = True        # 历史轮次是否丢 think 节省 token

    # 工具调用
    enable_tool_calls: bool = True               # 解析 DSML 并回填 tool_calls

    # 日志
    log_level: str = "INFO"
    log_request_payload: bool = False            # 调试时开

    # API Key 文件（每行一个 key，用于多租户）
    api_keys_file: Optional[str] = None
