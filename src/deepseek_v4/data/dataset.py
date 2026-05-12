"""
Dataset 实现：

1. PretrainDataset：纯文本，做语言建模（next-token prediction）。
2. PackedDataset：把短文档拼成 max_seq_len 长度，提升 GPU 利用率。
3. SFTDataset：对话数据，仅在 assistant 部分计算 loss（loss mask）。
4. DPODataset / PreferenceDataset：包含 chosen / rejected 偏好对。

所有 Dataset 都支持：
- 流式 / 内存模式
- 分布式 shard 切分
- Token 缓存（避免每个 epoch 重新 tokenize）
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset

from deepseek_v4.tokenizer.encoding import encode_messages
from deepseek_v4.utils.io import read_jsonl
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 基础：Token 缓存
# ============================================================

def _cache_key(paths: List[str], extra: str = "") -> str:
    """根据输入文件路径 + mtime 生成稳定缓存 key。"""
    sig = []
    for p in paths:
        p = Path(p)
        if p.exists():
            sig.append(f"{p}:{p.stat().st_size}:{int(p.stat().st_mtime)}")
        else:
            sig.append(str(p))
    sig.append(extra)
    return hashlib.md5("||".join(sig).encode()).hexdigest()[:16]


def _save_cache(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_cache(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Pretrain Dataset
# ============================================================

class PretrainDataset(Dataset):
    """
    纯文本预训练数据集。每个样本即一段 token 序列。

    每个文档独立 tokenize，超过 max_seq_len 会被截断。
    若需高 GPU 利用率，请使用 PackedDataset。
    """

    def __init__(
        self,
        paths: List[Union[str, Path]],
        tokenizer,
        max_seq_len: int = 4096,
        text_field: str = "text",
        cache_dir: Optional[Union[str, Path]] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.paths = [str(p) for p in paths]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_field = text_field
        self.add_bos = add_bos
        self.add_eos = add_eos

        # token 缓存
        self.token_ids: List[List[int]] = []
        cache_dir = Path(cache_dir) if cache_dir else None
        key = _cache_key(self.paths, f"pt_{max_seq_len}_{text_field}_{add_bos}_{add_eos}")
        cache_path = cache_dir / f"pretrain_{key}.pkl" if cache_dir else None

        if cache_path and cache_path.exists():
            logger.info(f"[PretrainDataset] loading cache: {cache_path}")
            self.token_ids = _load_cache(cache_path)
        else:
            self._build()
            if cache_path:
                logger.info(f"[PretrainDataset] saving cache: {cache_path}")
                _save_cache(cache_path, self.token_ids)
        logger.info(f"[PretrainDataset] {len(self.token_ids)} samples")

    def _build(self) -> None:
        for p in self.paths:
            for row in read_jsonl(p):
                text = row.get(self.text_field, "")
                if not isinstance(text, str) or not text.strip():
                    continue
                ids = self.tokenizer.encode(text)
                if self.add_bos and ids and ids[0] != self.tokenizer.bos_token_id:
                    ids = [self.tokenizer.bos_token_id] + ids
                if self.add_eos:
                    ids.append(self.tokenizer.eos_token_id)
                if len(ids) < 2:
                    continue
                # 截断
                if len(ids) > self.max_seq_len:
                    ids = ids[: self.max_seq_len]
                self.token_ids.append(ids)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.token_ids[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),
        }


# ============================================================
# Packed Pretrain Dataset
# ============================================================

class PackedDataset(Dataset):
    """
    Packing：把多个短文档拼接成 max_seq_len 长度。

    用 EOS 分隔，避免跨文档 attention（attention_mask 透过分隔实现）。
    每条 packed sample 都恰好是 max_seq_len 长度，无 padding 浪费。
    """

    def __init__(
        self,
        paths: List[Union[str, Path]],
        tokenizer,
        max_seq_len: int = 4096,
        text_field: str = "text",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.paths = [str(p) for p in paths]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_field = text_field

        self.packed: List[np.ndarray] = []
        cache_dir = Path(cache_dir) if cache_dir else None
        key = _cache_key(self.paths, f"pack_{max_seq_len}_{text_field}")
        cache_path = cache_dir / f"packed_{key}.pkl" if cache_dir else None

        if cache_path and cache_path.exists():
            logger.info(f"[PackedDataset] loading cache: {cache_path}")
            self.packed = _load_cache(cache_path)
        else:
            self._build()
            if cache_path:
                logger.info(f"[PackedDataset] saving cache: {cache_path}")
                _save_cache(cache_path, self.packed)
        logger.info(f"[PackedDataset] {len(self.packed)} packs of {max_seq_len} tokens")

    def _build(self) -> None:
        eos = self.tokenizer.eos_token_id
        bos = self.tokenizer.bos_token_id
        buf: List[int] = []

        def flush():
            if len(buf) >= self.max_seq_len:
                # 切多段
                while len(buf) >= self.max_seq_len:
                    self.packed.append(np.asarray(buf[: self.max_seq_len], dtype=np.int32))
                    del buf[: self.max_seq_len]

        for p in self.paths:
            for row in read_jsonl(p):
                text = row.get(self.text_field, "")
                if not isinstance(text, str) or not text.strip():
                    continue
                ids = self.tokenizer.encode(text)
                if not ids:
                    continue
                if not buf:
                    buf.append(bos)
                buf.extend(ids)
                buf.append(eos)
                flush()
        # 丢掉残余短包（< max_seq_len），避免不一致 batch
        # 想保留可改为 padding

    def __len__(self) -> int:
        return len(self.packed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        arr = self.packed[idx]
        ids = torch.from_numpy(arr.astype(np.int64))
        return {"input_ids": ids, "labels": ids.clone()}


# ============================================================
# SFT Dataset
# ============================================================

class SFTDataset(Dataset):
    """
    SFT 数据集：会话格式，仅 assistant 内容计 loss。

    输入 jsonl 行格式（任一）：
    1) {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    2) {"messages": [...], "thinking": True}   → 走 thinking_mode
    3) {"messages": [...], "tools": [...]}     → 工具调用 SFT
    """

    def __init__(
        self,
        paths: List[Union[str, Path]],
        tokenizer,
        max_seq_len: int = 4096,
        thinking_mode_default: str = "chat",
        mask_user: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        ignore_index: int = -100,
    ):
        self.paths = [str(p) for p in paths]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.thinking_mode_default = thinking_mode_default
        self.mask_user = mask_user
        self.ignore_index = ignore_index

        self.examples: List[Tuple[List[int], List[int]]] = []
        cache_dir = Path(cache_dir) if cache_dir else None
        key = _cache_key(self.paths, f"sft_{max_seq_len}_{thinking_mode_default}_{mask_user}")
        cache_path = cache_dir / f"sft_{key}.pkl" if cache_dir else None

        if cache_path and cache_path.exists():
            logger.info(f"[SFTDataset] loading cache: {cache_path}")
            self.examples = _load_cache(cache_path)
        else:
            self._build()
            if cache_path:
                logger.info(f"[SFTDataset] saving cache: {cache_path}")
                _save_cache(cache_path, self.examples)
        logger.info(f"[SFTDataset] {len(self.examples)} examples")

    def _build(self) -> None:
        """
        编码策略：
        1. 一次性 encode 整段对话 → prompt_text
        2. 逐条 assistant 消息单独 encode → 找到它在 prompt_text 中的偏移
        3. 将 prompt_text tokenize 后，根据偏移把对应 token 标记为可学习

        为了准确，采用"渐进式拼接"：依次 encode messages[:i+1] 与 messages[:i] 比较增量。
        """
        for p in self.paths:
            for row in read_jsonl(p):
                msgs = row.get("messages")
                if not msgs:
                    continue
                # 兼容不同结构
                msgs = self._normalize_messages(msgs)
                if not msgs:
                    continue

                thinking_mode = "thinking" if row.get("thinking") else self.thinking_mode_default
                tools = row.get("tools")
                if tools:
                    # 把 tools 挂到 system / 首条
                    for m in msgs:
                        if m.get("role") == "system":
                            m["tools"] = tools
                            break
                    else:
                        msgs = [{"role": "system", "content": "", "tools": tools}] + msgs

                input_ids, label_ids = self._encode_with_mask(msgs, thinking_mode)
                if not input_ids:
                    continue
                if len(input_ids) > self.max_seq_len:
                    input_ids = input_ids[: self.max_seq_len]
                    label_ids = label_ids[: self.max_seq_len]
                # 确保至少有一个有效 label
                if not any(l != self.ignore_index for l in label_ids):
                    continue
                self.examples.append((input_ids, label_ids))

    def _normalize_messages(self, msgs: Any) -> List[Dict[str, Any]]:
        """统一为 [{"role": ..., "content": ...}, ...] 格式。"""
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except Exception:
                return []
        if not isinstance(msgs, list):
            return []
        norm: List[Dict[str, Any]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            # 兼容 ShareGPT 等格式
            if "from" in m and "value" in m:
                role_map = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}
                role = role_map.get(m["from"], m["from"])
                norm.append({"role": role, "content": m["value"]})
            elif "role" in m:
                norm.append({k: v for k, v in m.items()})
            else:
                continue
        return norm

    def _encode_with_mask(
        self,
        msgs: List[Dict[str, Any]],
        thinking_mode: str,
    ) -> Tuple[List[int], List[int]]:
        """
        渐进式编码：对每个 assistant 消息，找到它在完整 prompt 中的 token 范围，作为可学习区域。

        策略：
            for i, m in enumerate(msgs):
                if m['role'] == 'assistant':
                    prefix_text = encode_messages(msgs[:i] + [user_stop])
                    full_text   = encode_messages(msgs[:i+1])
                    delta_text  = full_text[len(prefix_text):]
                    mask 该段为可学习
        """
        tok = self.tokenizer
        full_ids: List[int] = []
        label_ids: List[int] = []

        prev_text = ""
        prev_len = 0
        for i in range(len(msgs)):
            partial = msgs[: i + 1]
            try:
                cur_text = encode_messages(
                    partial,
                    thinking_mode=thinking_mode,
                    drop_thinking=False,
                    add_default_bos_token=(i == 0),
                )
            except Exception as e:
                logger.debug(f"encode_messages failed: {e}")
                return [], []

            cur_ids = tok.encode(cur_text)

            # 计算增量
            new_part = cur_ids[prev_len:]

            role = msgs[i].get("role")
            if role == "assistant":
                # 可学习
                full_ids.extend(new_part)
                label_ids.extend(new_part)
            else:
                full_ids.extend(new_part)
                label_ids.extend([self.ignore_index] * len(new_part))

            prev_text = cur_text
            prev_len = len(cur_ids)

        return full_ids, label_ids

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self.examples[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels, dtype=torch.long),
        }


# ============================================================
# DPO / Preference Dataset
# ============================================================

class DPODataset(Dataset):
    """
    DPO 偏好数据。

    每行 jsonl：
        { "prompt": str | messages,
          "chosen":  str (assistant content),
          "rejected": str }
        或
        { "chosen":   messages[],
          "rejected": messages[] }   # 完整对话各一份

    输出每条样本：
        {"prompt_ids": ..., "chosen_ids": ..., "rejected_ids": ...,
         "chosen_labels": ..., "rejected_labels": ...}
    """

    def __init__(
        self,
        paths: List[Union[str, Path]],
        tokenizer,
        max_prompt_len: int = 1024,
        max_seq_len: int = 2048,
        cache_dir: Optional[Union[str, Path]] = None,
        ignore_index: int = -100,
    ):
        self.paths = [str(p) for p in paths]
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

        self.examples: List[Dict[str, List[int]]] = []
        cache_dir = Path(cache_dir) if cache_dir else None
        key = _cache_key(self.paths, f"dpo_{max_prompt_len}_{max_seq_len}")
        cache_path = cache_dir / f"dpo_{key}.pkl" if cache_dir else None

        if cache_path and cache_path.exists():
            self.examples = _load_cache(cache_path)
            logger.info(f"[DPODataset] loaded cache: {len(self.examples)} examples")
        else:
            self._build()
            if cache_path:
                _save_cache(cache_path, self.examples)
        logger.info(f"[DPODataset] {len(self.examples)} examples")

    def _encode_response(self, prompt_msgs: List[Dict[str, Any]], response: str) -> Tuple[List[int], List[int]]:
        """编码 prompt + response，返回 (full_ids, response_ids_only)。"""
        prompt_text = encode_messages(prompt_msgs, thinking_mode="chat", drop_thinking=True)
        full_msgs = prompt_msgs + [{"role": "assistant", "content": response}]
        full_text = encode_messages(full_msgs, thinking_mode="chat", drop_thinking=True)
        prompt_ids = self.tokenizer.encode(prompt_text)
        full_ids = self.tokenizer.encode(full_text)
        response_ids = full_ids[len(prompt_ids):]
        return full_ids, response_ids

    def _normalize(self, row: Dict[str, Any]) -> Optional[Tuple[List[Dict], str, str]]:
        """规整为 (prompt_msgs, chosen_content, rejected_content)。"""
        if "prompt" in row and "chosen" in row and "rejected" in row:
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]
            if isinstance(prompt, str):
                prompt_msgs = [{"role": "user", "content": prompt}]
            else:
                prompt_msgs = prompt
            if isinstance(chosen, list):
                # chosen 为 messages
                chosen_content = next((m["content"] for m in chosen if m.get("role") == "assistant"), "")
            else:
                chosen_content = str(chosen)
            if isinstance(rejected, list):
                rejected_content = next((m["content"] for m in rejected if m.get("role") == "assistant"), "")
            else:
                rejected_content = str(rejected)
            return prompt_msgs, chosen_content, rejected_content
        if "chosen" in row and "rejected" in row and isinstance(row["chosen"], list):
            # 完整对话各一份
            chosen_msgs = row["chosen"]
            rejected_msgs = row["rejected"]
            # 找最后一个 assistant 作为目标
            last_assistant_c = next((m for m in reversed(chosen_msgs) if m.get("role") == "assistant"), None)
            last_assistant_r = next((m for m in reversed(rejected_msgs) if m.get("role") == "assistant"), None)
            if not last_assistant_c or not last_assistant_r:
                return None
            prompt_msgs = [m for m in chosen_msgs if m is not last_assistant_c]
            return prompt_msgs, last_assistant_c["content"], last_assistant_r["content"]
        return None

    def _build(self) -> None:
        for p in self.paths:
            for row in read_jsonl(p):
                norm = self._normalize(row)
                if norm is None:
                    continue
                prompt_msgs, chosen, rejected = norm
                if not chosen or not rejected:
                    continue

                try:
                    chosen_ids, _ = self._encode_response(prompt_msgs, chosen)
                    rejected_ids, _ = self._encode_response(prompt_msgs, rejected)
                except Exception:
                    continue

                # prompt ids
                prompt_text = encode_messages(prompt_msgs, thinking_mode="chat", drop_thinking=True)
                prompt_ids = self.tokenizer.encode(prompt_text)
                if len(prompt_ids) > self.max_prompt_len:
                    # 左截断保留最近上下文
                    prompt_ids = prompt_ids[-self.max_prompt_len:]
                    # 重新拼合 chosen / rejected
                    chosen_ids = prompt_ids + chosen_ids[len(prompt_ids):]
                    rejected_ids = prompt_ids + rejected_ids[len(prompt_ids):]
                if len(chosen_ids) > self.max_seq_len:
                    chosen_ids = chosen_ids[: self.max_seq_len]
                if len(rejected_ids) > self.max_seq_len:
                    rejected_ids = rejected_ids[: self.max_seq_len]

                p_len = len(prompt_ids)
                # labels：prompt 部分 mask 掉
                chosen_labels = [self.ignore_index] * p_len + chosen_ids[p_len:]
                rejected_labels = [self.ignore_index] * p_len + rejected_ids[p_len:]

                self.examples.append({
                    "prompt_ids":      prompt_ids,
                    "chosen_ids":      chosen_ids,
                    "rejected_ids":    rejected_ids,
                    "chosen_labels":   chosen_labels,
                    "rejected_labels": rejected_labels,
                })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {k: torch.tensor(v, dtype=torch.long) for k, v in ex.items()}


# 别名（用于 reward model 等）
PreferenceDataset = DPODataset


# ============================================================
# DataLoader 构造工具
# ============================================================

def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Callable,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
    seed: int = 42,
) -> DataLoader:
    """根据是否分布式自动选 sampler。"""
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)
        shuffle = False  # sampler 已经处理
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )
