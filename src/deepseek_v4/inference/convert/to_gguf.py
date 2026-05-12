"""
导出 GGUF（llama.cpp）。

注意：
- llama.cpp 主线尚不一定原生支持 DeepSeek-V4 的 MLA + mHC + MoE 全部细节；
  本工具采用 **best-effort 方式**：
    1. 先导出标准 HF 目录
    2. 优先尝试用 `convert_hf_to_gguf.py`（来自 llama.cpp 仓库）
    3. 若该脚本不可用 / 不支持 V4 架构，则提示用户手动转换或使用 vLLM

实际生产中：
- 若 V4 架构尚未被 llama.cpp 支持，需要在 llama.cpp 端贡献一份 model_arch
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

from deepseek_v4.inference.convert.to_hf import export_to_hf
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_gguf(
    state_dict_dir: Union[str, Path],
    model_config_path: Union[str, Path],
    tokenizer_dir: Union[str, Path],
    output_dir: Union[str, Path],
    quantization: str = "f16",                # f32 | f16 | q8_0 | q4_K_M ...
    convert_script_path: Optional[str] = None,
    llama_cpp_dir: Optional[str] = None,
    keep_intermediate_hf: bool = False,
) -> Path:
    """
    步骤：
    1. 导出 HF 目录到 `output_dir/_hf_tmp/`
    2. 调用 `convert_hf_to_gguf.py` → `output_dir/model.gguf`
    3. 若 quantization != f16/f32 → 调用 `llama-quantize` 量化
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp_hf = out / "_hf_tmp"

    logger.info(f"[to_gguf] step 1: export HF format → {tmp_hf}")
    export_to_hf(
        state_dict_dir=state_dict_dir,
        model_config_path=model_config_path,
        tokenizer_dir=tokenizer_dir,
        output_dir=tmp_hf,
        max_shard_size="10GB",     # gguf 转换工具往往要求单文件，给宽松点
    )

    # 找 convert_hf_to_gguf.py
    convert_script = _locate_convert_script(convert_script_path, llama_cpp_dir)
    if convert_script is None:
        msg = (
            "[to_gguf] cannot find convert_hf_to_gguf.py.\n"
            "Please:\n"
            "  1) git clone https://github.com/ggml-org/llama.cpp\n"
            "  2) pip install -r llama.cpp/requirements.txt\n"
            "  3) Re-run with --llama_cpp_dir /path/to/llama.cpp\n"
            "Or: keep HF format at {tmp} and convert manually.".format(tmp=tmp_hf)
        )
        logger.error(msg)
        return tmp_hf

    raw_gguf = out / "model-f16.gguf"
    logger.info(f"[to_gguf] step 2: running convert script → {raw_gguf}")
    try:
        subprocess.run(
            [
                sys.executable, str(convert_script),
                str(tmp_hf),
                "--outfile", str(raw_gguf),
                "--outtype", "f16",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"[to_gguf] convert failed: {e}")
        return tmp_hf

    if quantization in ("f16", "f32"):
        final = raw_gguf
    else:
        # 调用 llama-quantize
        quantize_bin = _locate_quantize_bin(llama_cpp_dir)
        if quantize_bin is None:
            logger.warning("[to_gguf] llama-quantize not found, kept f16")
            final = raw_gguf
        else:
            final = out / f"model-{quantization}.gguf"
            logger.info(f"[to_gguf] step 3: quantize to {quantization} → {final}")
            try:
                subprocess.run(
                    [str(quantize_bin), str(raw_gguf), str(final), quantization],
                    check=True,
                )
                if raw_gguf.exists() and final.exists():
                    raw_gguf.unlink()
            except subprocess.CalledProcessError as e:
                logger.error(f"[to_gguf] quantize failed: {e}")
                final = raw_gguf

    if not keep_intermediate_hf:
        shutil.rmtree(tmp_hf, ignore_errors=True)

    logger.info(f"[to_gguf] ✅ done: {final}")
    return final


def _locate_convert_script(explicit: Optional[str], llama_cpp_dir: Optional[str]) -> Optional[Path]:
    if explicit and Path(explicit).exists():
        return Path(explicit)
    candidates = []
    if llama_cpp_dir:
        candidates += [
            Path(llama_cpp_dir) / "convert_hf_to_gguf.py",
            Path(llama_cpp_dir) / "convert-hf-to-gguf.py",
        ]
    # 常见安装路径
    candidates += [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/usr/local/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _locate_quantize_bin(llama_cpp_dir: Optional[str]) -> Optional[Path]:
    if llama_cpp_dir:
        for name in ("llama-quantize", "quantize"):
            p = Path(llama_cpp_dir) / "build" / "bin" / name
            if p.exists():
                return p
            p = Path(llama_cpp_dir) / name
            if p.exists():
                return p
    # PATH
    from shutil import which
    for name in ("llama-quantize", "quantize"):
        path = which(name)
        if path:
            return Path(path)
    return None
