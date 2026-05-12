"""
推理引擎抽象。

提供 3 种 backend：
- LocalEngine    本地 DeepseekV4ForCausalLM（用我们自己的 generate）
- OpenAIEngine   兼容 OpenAI Chat API（含 vLLM / 本服务端 / 其它兼容服务）
- VLLMEngine     直接调用 vllm.LLM（如果安装了 vllm）

统一接口：
    engine.generate(prompts: List[str], **kwargs) -> List[str]
    engine.compute_logprobs(prompts, completions) -> List[float]   (可选)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 抽象基类
# ============================================================

class InferenceEngine:
    name: str = "base"

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """对一批 prompt 生成文本。返回的字符串不含原 prompt。"""
        raise NotImplementedError

    def compute_logprobs(
        self,
        prompts: List[str],
        completions: List[str],
    ) -> List[float]:
        """计算 P(completion | prompt) 的对数（用于多选题 ppl scoring）。"""
        raise NotImplementedError

    def close(self):
        pass


# ============================================================
# Local Engine
# ============================================================

class LocalEngine(InferenceEngine):
    """
    使用本地 DeepseekV4ForCausalLM + 我们自己的 generate。

    支持 multiple_choice 的 logprob scoring（避免生成纯文本再 parse）。
    """
    name = "local"

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 4096,
        yarn_factor: Optional[float] = None,    # 推理时动态切换 YaRN factor
    ):
        from deepseek_v4.modeling.model import DeepseekV4Config, DeepseekV4ForCausalLM
        from deepseek_v4.tokenizer.tokenizer import DeepseekV4Tokenizer

        self.tokenizer = DeepseekV4Tokenizer.from_pretrained(tokenizer_path)
        # 加载 config
        cfg_file = Path(model_path) / "config.json"
        if cfg_file.exists():
            import json
            with open(cfg_file, "r", encoding="utf-8") as f:
                self.config = DeepseekV4Config.from_dict(json.load(f))
        else:
            raise FileNotFoundError(f"config.json not found in {model_path}")

        # 在加载权重之前注入 YaRN scaling
        if yarn_factor is not None and yarn_factor > 1.0:
            self._apply_yarn(self.config, yarn_factor, max_seq_len)

        self.model = DeepseekV4ForCausalLM(self.config)
        self._load_state(model_path)
        self.device = torch.device(device)
        self.model.to(self.device, dtype=dtype)
        self.model.eval()
        self.max_seq_len = max_seq_len
        logger.info(
            f"[LocalEngine] loaded from {model_path}, "
            f"device={self.device}, dtype={dtype}, max_seq_len={max_seq_len}, "
            f"yarn_factor={yarn_factor}"
        )

    @staticmethod
    def _apply_yarn(config, factor: float, target_max_position: int) -> None:
        """
        修改 config 中的 rope_scaling 以启用 YaRN。
        """
        original = (
            config.rope_scaling.get("original_max_position_embeddings", 65536)
            if config.rope_scaling
            else 65536
        )
        config.rope_scaling = {
            "type": "yarn",
            "factor": factor,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": original,
        }
        config.max_position_embeddings = max(
            config.max_position_embeddings, int(original * factor), target_max_position
        )
        # 重新建 rope_parameters（DeepseekV4Config.__post_init__ 一致性）
        partial = config.qk_rope_head_dim / config.head_dim
        rope_extra = {k: v for k, v in config.rope_scaling.items() if k != "type"}
        config.rope_parameters = {
            "main": {
                "rope_type": "yarn", "rope_theta": config.rope_theta,
                "partial_rotary_factor": partial, **rope_extra,
            },
            "compress": {
                "rope_type": "yarn", "rope_theta": config.compress_rope_theta,
                "partial_rotary_factor": partial, **rope_extra,
            },
        }
        logger.info(f"[LocalEngine] YaRN injected: factor={factor}, new_max_pos={config.max_position_embeddings}")

    def _load_state(self, model_path: str) -> None:
        from safetensors.torch import load_file
        p = Path(model_path)
        if p.is_dir():
            # 优先分片 index，否则单文件
            idx = p / "model.safetensors.index.json"
            if idx.exists():
                import json
                with open(idx, "r", encoding="utf-8") as f:
                    weight_map = json.load(f)["weight_map"]
                files = set(weight_map.values())
                state_dict = {}
                for f in files:
                    state_dict.update(load_file(str(p / f)))
            else:
                st = p / "model.safetensors"
                bin_ = p / "pytorch_model.bin"
                if st.exists():
                    state_dict = load_file(str(st))
                elif bin_.exists():
                    state_dict = torch.load(str(bin_), map_location="cpu")
                else:
                    raise FileNotFoundError(f"no model file in {p}")
        else:
            state_dict = load_file(str(p)) if str(p).endswith(".safetensors") \
                else torch.load(str(p), map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"[LocalEngine] missing={len(missing)} unexpected={len(unexpected)}")

    # --------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        from deepseek_v4.inference.generation import GenerationConfig, generate

        # 左 padding 编码
        encoded = [self.tokenizer.encode(p) for p in prompts]
        # 截断防越界
        encoded = [e[-self.max_seq_len:] for e in encoded]
        max_p = max(len(e) for e in encoded)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full((len(prompts), max_p), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(input_ids)
        for i, e in enumerate(encoded):
            input_ids[i, -len(e):] = torch.tensor(e, dtype=torch.long, device=self.device)
            attention_mask[i, -len(e):] = 1

        stop_ids = []
        if stop:
            for s in stop:
                ids = self.tokenizer.encode(s)
                if len(ids) == 1:
                    stop_ids.append(ids[0])

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            top_k=kwargs.get("top_k", 0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            stop_token_ids=stop_ids,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        out = generate(self.model, input_ids, attention_mask=attention_mask, config=gen_cfg)
        # 仅 decode 生成部分
        responses = out["responses"]
        response_mask = out["response_mask"]
        texts: List[str] = []
        for i in range(responses.shape[0]):
            valid_len = int(response_mask[i].sum().item())
            ids = responses[i, :valid_len].tolist()
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            # 进一步剪掉自定义 stop
            if stop:
                for s in stop:
                    if s in text:
                        text = text[: text.index(s)]
                        break
            texts.append(text)
        return texts

    # --------------------------------------------------------------

    @torch.no_grad()
    def compute_logprobs(
        self,
        prompts: List[str],
        completions: List[str],
    ) -> List[float]:
        """
        计算 P(completion | prompt) 的 log。

        实现：把 prompt + completion 拼接成完整序列，模型一次 forward，
        累加 completion 区域的 per-token log-prob。
        """
        import torch.nn.functional as F
        assert len(prompts) == len(completions)
        out: List[float] = []
        pad_id = self.tokenizer.pad_token_id

        for p, c in zip(prompts, completions):
            p_ids = self.tokenizer.encode(p)
            c_ids = self.tokenizer.encode(c)
            full = p_ids + c_ids
            if len(full) > self.max_seq_len:
                # 截断 prompt 头
                cut = len(full) - self.max_seq_len
                p_ids = p_ids[cut:]
                full = p_ids + c_ids
            input_ids = torch.tensor([full], dtype=torch.long, device=self.device)
            attn = torch.ones_like(input_ids)
            outputs = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            # shift
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            per = log_probs.gather(-1, shift_labels[:, :, None]).squeeze(-1)
            # 只累加 completion 区域：起点 = len(p_ids) - 1（shift 后）
            start = max(len(p_ids) - 1, 0)
            score = float(per[0, start:].sum().item())
            out.append(score)
        return out


# ============================================================
# OpenAI / OpenAI-compatible Engine
# ============================================================

class OpenAIEngine(InferenceEngine):
    """通过 OpenAI 兼容 API 推理（含 vLLM 服务、本仓库 server、其他兼容服务）。"""
    name = "openai"

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "deepseek-v4-mini",
        max_workers: int = 8,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        try:
            import openai
            self._openai = openai
        except ImportError:
            raise ImportError("请 pip install openai>=1.0")
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.max_workers = max_workers
        self.max_retries = max_retries

    def _one(self, prompt: str, **kwargs) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                # 这里默认走 completions 风格（评测通常 prompt 已渲染完毕）
                resp = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=kwargs.get("max_new_tokens", 512),
                    temperature=kwargs.get("temperature", 0.0),
                    top_p=kwargs.get("top_p", 1.0),
                    stop=kwargs.get("stop"),
                )
                return resp.choices[0].text
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"OpenAIEngine failed after {self.max_retries} retries: {last_err}")

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda p: self._one(p, **kwargs), prompts))


# ============================================================
# vLLM Engine
# ============================================================

class VLLMEngine(InferenceEngine):
    """直接调用 vllm.LLM（推理快，仅作为离线 benchmark）。"""
    name = "vllm"

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("请 pip install vllm")
        self._SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path or model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    def generate(
        self, prompts: List[str], max_new_tokens=512, temperature=0.0, top_p=1.0, stop=None, **kwargs,
    ) -> List[str]:
        params = self._SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            top_k=kwargs.get("top_k", -1),
        )
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]


# ============================================================
# Engine 工厂
# ============================================================

def build_engine(
    backend: str = "local",
    **kwargs,
) -> InferenceEngine:
    """
    工厂方法。

    backend in {"local", "openai", "vllm"}。
    """
    backend = backend.lower()
    if backend == "local":
        return LocalEngine(**kwargs)
    if backend == "openai":
        return OpenAIEngine(**kwargs)
    if backend == "vllm":
        return VLLMEngine(**kwargs)
    raise ValueError(f"Unknown backend: {backend}")
