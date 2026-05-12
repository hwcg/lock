"""
生成 Ollama Modelfile，并可选地调用 `ollama create` 注册。

前置：本机已安装 ollama CLI（https://ollama.com）。

Modelfile 主要内容：
- FROM <gguf path>
- TEMPLATE  (将 OpenAI messages → DeepSeek-V4 chat 模板)
- PARAMETER stop <stop token>
- SYSTEM (default system prompt)
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Union

from deepseek_v4.tokenizer.special_tokens import (
    ASSISTANT_TOKEN, BOS_TOKEN, EOS_TOKEN, SYSTEM_TOKEN, USER_TOKEN,
)
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# Ollama 模板语法（Go template + 自定义 helper）
# 这里渲染策略：
#   {{- if .System }}{ system }{ /if }}{{- range .Messages }} ... {{- end }}
OLLAMA_TEMPLATE = (
    "{{ if .System }}" + SYSTEM_TOKEN + "{{ .System }}{{ end }}"
    "{{- range .Messages }}"
    "{{- if eq .Role \"user\" }}" + USER_TOKEN + "{{ .Content }}"
    "{{- else if eq .Role \"assistant\" }}" + ASSISTANT_TOKEN + "{{ .Content }}" + EOS_TOKEN
    "{{- end }}"
    "{{- end }}"
    + ASSISTANT_TOKEN
)


def build_ollama_modelfile(
    gguf_path: Union[str, Path],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    stop_strings: Optional[List[str]] = None,
    num_ctx: int = 4096,
) -> str:
    """生成 Modelfile 文本。"""
    stops = stop_strings or [EOS_TOKEN, USER_TOKEN, SYSTEM_TOKEN]
    parts: List[str] = [f"FROM {Path(gguf_path).resolve()}", ""]
    parts.append('TEMPLATE """' + OLLAMA_TEMPLATE + '"""')
    if system_prompt:
        parts.append(f'SYSTEM """{system_prompt}"""')
    parts.append(f"PARAMETER temperature {temperature}")
    parts.append(f"PARAMETER top_p {top_p}")
    parts.append(f"PARAMETER num_ctx {num_ctx}")
    for s in stops:
        parts.append(f'PARAMETER stop "{s}"')
    return "\n".join(parts) + "\n"


def export_to_ollama(
    gguf_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = "deepseek-v4-mini",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    num_ctx: int = 4096,
    run_create: bool = True,
) -> Path:
    """
    生成 Modelfile（可选）调用 `ollama create` 注册模型。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mf = build_ollama_modelfile(
        gguf_path=gguf_path,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
    )
    mf_path = output_dir / "Modelfile"
    mf_path.write_text(mf, encoding="utf-8")
    logger.info(f"[ollama] Modelfile → {mf_path}")

    if run_create:
        try:
            subprocess.run(
                ["ollama", "create", model_name, "-f", str(mf_path)],
                check=True,
            )
            logger.info(f"[ollama] ✅ registered model: {model_name}")
        except FileNotFoundError:
            logger.warning("[ollama] CLI not found, skip `ollama create`")
        except subprocess.CalledProcessError as e:
            logger.error(f"[ollama] create failed: {e}")
    return mf_path
