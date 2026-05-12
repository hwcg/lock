"""
DataLoader collator：负责 padding 与对齐。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class PadCollator:
    """通用 padding collator。"""
    pad_token_id: int
    ignore_index: int = -100
    padding_side: str = "right"
    max_length: int = None  # None 表示动态 padding 到 batch 最大长度
    pad_to_multiple_of: int = None

    def _pad_one(self, seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(s.shape[0] for s in seqs)
        if self.max_length:
            max_len = min(max_len, self.max_length)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            n = min(s.shape[0], max_len)
            if self.padding_side == "right":
                out[i, :n] = s[:n]
            else:
                out[i, -n:] = s[:n]
        return out

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = batch[0].keys()
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            seqs = [b[k] for b in batch]
            pad_val = self.ignore_index if "label" in k else self.pad_token_id
            out[k] = self._pad_one(seqs, pad_val)
        # attention_mask
        if "attention_mask" not in out and "input_ids" in out:
            out["attention_mask"] = (out["input_ids"] != self.pad_token_id).long()
        return out


@dataclass
class PretrainCollator(PadCollator):
    """预训练专用（无差别）。"""
    pass


@dataclass
class SFTCollator(PadCollator):
    """SFT 专用：把 labels 中 pad 部分置为 ignore_index。"""
    pass


@dataclass
class DPOCollator:
    """DPO 专用 collator：把 chosen / rejected 分别 pad。"""
    pad_token_id: int
    ignore_index: int = -100
    pad_to_multiple_of: int = None

    def _pad(self, seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(s.shape[0] for s in seqs)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            out[i, :s.shape[0]] = s
        return out

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key in ("chosen_ids", "rejected_ids"):
            seqs = [b[key] for b in batch]
            out[key] = self._pad(seqs, self.pad_token_id)
            out[key.replace("_ids", "_attention_mask")] = (out[key] != self.pad_token_id).long()
        for key in ("chosen_labels", "rejected_labels"):
            seqs = [b[key] for b in batch]
            out[key] = self._pad(seqs, self.ignore_index)
        return out
