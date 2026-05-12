"""
服务端推理引擎（含 continuous batching 简化版）。

设计：
- 单个 worker 线程持有 model；外部用 asyncio.Queue 提交请求
- worker 每个调度循环：
    1. 从队列收集所有 ready 请求（最长等 batch_wait_ms）
    2. 对每个 request 准备 prompt → 左 padding 成 batch
    3. prefill 一次（建立 cache）
    4. 同步 decode 循环：每步从队列检查是否要新加入请求（简化版：当前 batch 全部跑完再纳新）
- 每个 token 一旦产生立刻 push 到对应请求的 streaming queue

注意：
- 这是「同步 batch + 增量 generate」的简化 continuous batching：吞吐已较朴素 sequential 大幅提高
- 完整 PagedAttention / token-level scheduling 不在 demo 范围内（生产推荐 vLLM）
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from deepseek_v4.inference.generation import (
    GenerationConfig,
    prepare_logits_warper,
    sample_token,
)
from deepseek_v4.inference.server.config import ServerConfig
from deepseek_v4.inference.yarn import YarnConfig, apply_yarn_to_model
from deepseek_v4.modeling.model import DeepseekV4Config, DeepseekV4ForCausalLM
from deepseek_v4.tokenizer.tokenizer import DeepseekV4Tokenizer
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Request / Response 数据结构
# ============================================================

@dataclass
class GenerationRequest:
    """单个生成请求。"""
    request_id: str
    prompt_ids: List[int]                          # 已 tokenize 的 prompt
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop_token_ids: List[int] = field(default_factory=list)
    stop_strings: List[str] = field(default_factory=list)
    eos_token_id: int = 1
    pad_token_id: int = 0
    bos_token_id: int = 0
    stream: bool = True

    # 状态
    output_queue: "asyncio.Queue[Dict[str, Any]]" = None
    cancelled: bool = False

    # 调度填充
    created_at: float = field(default_factory=time.time)


# ============================================================
# ServerEngine
# ============================================================

class ServerEngine:
    """
    简化 continuous batching server engine。

    用法：
        engine = ServerEngine(config)
        engine.start()
        await engine.submit(request)
        ... 从 request.output_queue 读 token ...
        engine.stop()
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._pending: "Queue[GenerationRequest]" = Queue(
            maxsize=config.max_concurrent_requests,
        )

        # 模型 / tokenizer
        self.tokenizer: Optional[DeepseekV4Tokenizer] = None
        self.model: Optional[DeepseekV4ForCausalLM] = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self.model_config: Optional[DeepseekV4Config] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------------

    def load(self) -> None:
        cfg = self.config
        logger.info(f"[Engine] Loading tokenizer: {cfg.tokenizer_path}")
        self.tokenizer = DeepseekV4Tokenizer.from_pretrained(cfg.tokenizer_path)

        logger.info(f"[Engine] Loading model: {cfg.model_path}")
        cfg_path = Path(cfg.model_path) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in {cfg.model_path}")
        import json
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.model_config = DeepseekV4Config.from_dict(json.load(f))

        self.model = DeepseekV4ForCausalLM(self.model_config)
        # YaRN 注入（在加载权重之前修改 config + 重算 inv_freq）
        if cfg.yarn_factor and cfg.yarn_factor > 1.0:
            yarn = YarnConfig(
                method="yarn", factor=cfg.yarn_factor,
                original_max_position=self.model_config.rope_scaling.get(
                    "original_max_position_embeddings",
                    self.model_config.max_position_embeddings,
                ),
                target_max_position=cfg.max_model_len,
            )
            apply_yarn_to_model(self.model, yarn)

        self._load_state(cfg.model_path)

        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        # 预热
        try:
            with torch.no_grad():
                dummy = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
                _ = self.model(input_ids=dummy, use_cache=True)
            logger.info("[Engine] Warm-up forward done")
        except Exception as e:
            logger.warning(f"[Engine] warm-up failed: {e}")

    def _load_state(self, model_path: str) -> None:
        from safetensors.torch import load_file
        p = Path(model_path)
        if p.is_dir():
            idx = p / "model.safetensors.index.json"
            if idx.exists():
                import json as _json
                with open(idx, "r", encoding="utf-8") as f:
                    weight_map = _json.load(f)["weight_map"]
                files = set(weight_map.values())
                state_dict = {}
                for fn in files:
                    state_dict.update(load_file(str(p / fn)))
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
            state_dict = (
                load_file(str(p)) if str(p).endswith(".safetensors")
                else torch.load(str(p), map_location="cpu")
            )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"[Engine] missing={len(missing)} unexpected={len(unexpected)}")

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.model is None:
            self.load()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="ds4-engine-worker", daemon=True,
        )
        self._worker_thread.start()
        logger.info("[Engine] worker started")

    def stop(self, timeout: float = 30.0) -> None:
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
            logger.info("[Engine] worker stopped")

    # ------------------------------------------------------------------
    # 提交请求
    # ------------------------------------------------------------------

    async def submit(self, req: GenerationRequest) -> None:
        if req.output_queue is None:
            req.output_queue = asyncio.Queue()
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        try:
            self._pending.put(req, block=False)
        except Exception:
            # 满了：放入一个 ERROR 事件
            await req.output_queue.put({"type": "error", "message": "server overloaded"})

    # ------------------------------------------------------------------
    # Worker：核心调度
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        cfg = self.config
        try:
            while not self._stop_event.is_set():
                batch = self._collect_batch()
                if not batch:
                    time.sleep(0.001)
                    continue
                try:
                    self._run_batch(batch)
                except Exception as e:
                    logger.exception(f"[Engine] batch failed: {e}")
                    for r in batch:
                        self._push(r, {"type": "error", "message": str(e)})
        except Exception:
            logger.exception("[Engine] worker crashed")

    def _collect_batch(self) -> List[GenerationRequest]:
        """收集一批 ready 的请求。"""
        deadline = time.time() + self.config.batch_wait_ms / 1000.0
        batch: List[GenerationRequest] = []
        while len(batch) < self.config.max_batch_size:
            timeout = max(deadline - time.time(), 0.0)
            try:
                req = self._pending.get(timeout=timeout)
                if not req.cancelled:
                    batch.append(req)
            except Empty:
                break
        return batch

    # ------------------------------------------------------------------
    # 执行一个 batch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_batch(self, batch: List[GenerationRequest]) -> None:
        cfg = self.config
        B = len(batch)
        pad_id = self.tokenizer.pad_token_id

        # 左 padding
        max_plen = max(len(r.prompt_ids) for r in batch)
        max_plen = min(max_plen, cfg.max_model_len - 1)
        input_ids = torch.full(
            (B, max_plen), pad_id, dtype=torch.long, device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for i, r in enumerate(batch):
            ids = r.prompt_ids[-max_plen:]
            input_ids[i, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=self.device)
            attention_mask[i, -len(ids):] = 1

        # Prefill
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
        )
        past_kv = out["past_key_values"]
        last_logits = out["logits"][:, -1, :]

        # 每条 request 维护：prev_ids (用于 repetition penalty), 是否结束, 已生成 token 数
        prev_ids = input_ids.clone()
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        per_req_tokens = [[] for _ in range(B)]
        per_req_text_emitted = ["" for _ in range(B)]   # 已 push 到 client 的字符串

        # 各 request 的 stop sets
        stop_ids_sets = [
            set(r.stop_token_ids) | {r.eos_token_id} for r in batch
        ]
        max_new_per_req = [r.max_new_tokens for r in batch]
        max_steps = max(max_new_per_req)

        # 第一个 token sample
        warpers = [
            prepare_logits_warper(GenerationConfig(
                temperature=r.temperature, top_p=r.top_p, top_k=r.top_k,
                repetition_penalty=r.repetition_penalty,
                do_sample=(r.temperature > 0),
            )) for r in batch
        ]

        def _step_sample(logits: torch.Tensor) -> torch.Tensor:
            """对 batch 中每个 row 各自 warp + sample。"""
            tokens = torch.empty(B, dtype=torch.long, device=self.device)
            for i in range(B):
                if finished[i]:
                    tokens[i] = pad_id
                    continue
                lg = warpers[i](logits[i:i + 1].float(), prev_ids[i:i + 1])
                tk, _ = sample_token(lg, do_sample=batch[i].temperature > 0)
                tokens[i] = tk
            return tokens

        next_token = _step_sample(last_logits)
        for step in range(max_steps):
            # 写出 token
            cur_attn = torch.cat(
                [attention_mask, torch.ones((B, 1), dtype=attention_mask.dtype, device=self.device)],
                dim=1,
            )
            attention_mask = cur_attn

            for i in range(B):
                if finished[i] or step >= max_new_per_req[i]:
                    continue
                tok = int(next_token[i].item())
                per_req_tokens[i].append(tok)
                # 检查停止 token
                if tok in stop_ids_sets[i]:
                    finished[i] = True
                # 流式 decode：每个 token 增量 decode；为正确处理 BPE 的字节连续性，
                # 重新 decode 整段 then 取差
                text_so_far = self.tokenizer.decode(per_req_tokens[i], skip_special_tokens=False)
                # 检查 stop strings
                for s in batch[i].stop_strings:
                    if s in text_so_far[len(per_req_text_emitted[i]):]:
                        idx = text_so_far.find(s, len(per_req_text_emitted[i]))
                        text_so_far = text_so_far[:idx]
                        finished[i] = True
                        break
                delta = text_so_far[len(per_req_text_emitted[i]):]
                per_req_text_emitted[i] = text_so_far
                if delta:
                    self._push(batch[i], {"type": "delta", "text": delta})
                if step + 1 >= max_new_per_req[i]:
                    finished[i] = True
                    # 留 finish_reason=length
                    self._push(batch[i], {"type": "finish", "reason": "length"})
                elif finished[i]:
                    self._push(batch[i], {"type": "finish", "reason": "stop"})

            if finished.all():
                break

            # 准备下一个 step 输入
            inp = next_token[:, None]
            prev_ids = torch.cat([prev_ids, inp], dim=1)

            out = self.model(
                input_ids=inp, attention_mask=attention_mask,
                past_key_values=past_kv, use_cache=True,
            )
            past_kv = out["past_key_values"]
            last_logits = out["logits"][:, -1, :]
            next_token = _step_sample(last_logits)

        # 兜底：尚未发 finish 的 request
        for i in range(B):
            if not _request_finished_signal_sent(batch[i]):
                # 没发就追加 length
                self._push(batch[i], {"type": "finish", "reason": "length"})

    # ------------------------------------------------------------------
    # 把消息 push 到 async queue（跨线程）
    # ------------------------------------------------------------------

    def _push(self, req: GenerationRequest, payload: Dict[str, Any]) -> None:
        if req.cancelled or self.loop is None or req.output_queue is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                req.output_queue.put(payload), self.loop,
            )
            if payload.get("type") in ("finish", "error"):
                req._finished = True
        except Exception:
            pass


def _request_finished_signal_sent(req: GenerationRequest) -> bool:
    return getattr(req, "_finished", False)
