"""
DeepSeek-V4 主 Tokenizer 类。

封装 BPE + chat encoding，提供 HuggingFace 兼容接口：
- save_pretrained / from_pretrained
- __call__ / encode / decode / batch_encode_plus
- apply_chat_template
- add_special_tokens

可被 transformers.AutoTokenizer.from_pretrained 加载（通过 trust_remote_code）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from deepseek_v4.tokenizer.bpe import BPETokenizer
from deepseek_v4.tokenizer.encoding import encode_messages, parse_message_from_completion_text
from deepseek_v4.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    SpecialTokens,
)

TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.txt"
SPECIAL_TOKENS_FILE = "special_tokens_map.json"


class DeepseekV4Tokenizer:
    """
    DeepSeek-V4 完整 Tokenizer。
    
    主要方法：
        __call__(text)              → {input_ids, attention_mask, ...}
        encode(text)                → List[int]
        decode(ids)                 → str
        apply_chat_template(msgs)   → str | List[int]
        parse_assistant_output(s)   → assistant message dict
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        bpe_tokenizer: BPETokenizer,
        special_tokens: Optional[SpecialTokens] = None,
        model_max_length: int = 4096,
        padding_side: str = "left",
        truncation_side: str = "right",
    ):
        self.bpe = bpe_tokenizer
        self.st = special_tokens or SpecialTokens.default()
        self.model_max_length = model_max_length
        self.padding_side = padding_side
        self.truncation_side = truncation_side

        # 资源 ID
        self.bos_token = self.st.bos
        self.eos_token = self.st.eos
        self.pad_token = self.st.pad
        self.unk_token = self.st.unk

        self.bos_token_id = self.bpe.token_to_id(self.bos_token)
        self.eos_token_id = self.bpe.token_to_id(self.eos_token)
        self.pad_token_id = self.bpe.token_to_id(self.pad_token)
        self.unk_token_id = self.bpe.token_to_id(self.unk_token)

        for tok in ALL_SPECIAL_TOKENS:
            if self.bpe.token_to_id(tok) is None:
                raise ValueError(f"Special token {tok!r} 未在 vocab 中找到，请重新训练 tokenizer")

        # 把所有特殊 token 注册到 BPE 中（如果不在）
        for tok in ALL_SPECIAL_TOKENS:
            self.bpe.special_tokens.add(tok)
        # 重新编译特殊 token 正则
        import re
        escaped = sorted((re.escape(t) for t in self.bpe.special_tokens), key=len, reverse=True)
        self.bpe._special_re = re.compile("(" + "|".join(escaped) + ")")

    # ------------------------------------------------------------------
    # 基础编码
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], torch.Tensor]:
        ids = self.bpe.encode(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        if truncation and max_length and len(ids) > max_length:
            if self.truncation_side == "right":
                ids = ids[:max_length]
            else:
                ids = ids[-max_length:]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return [self.decode(x, skip_special_tokens=skip_special_tokens) for x in ids]
        return self.bpe.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        sequences: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = False,
    ) -> List[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.detach().cpu().tolist()
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]

    # ------------------------------------------------------------------
    # HuggingFace-style __call__
    # ------------------------------------------------------------------

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = False,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
    ) -> Dict[str, Any]:
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = list(text)
            single = False

        all_ids: List[List[int]] = []
        for t in texts:
            ids = self.bpe.encode(t)
            if add_special_tokens:
                ids = [self.bos_token_id] + ids
            if truncation and max_length and len(ids) > max_length:
                if self.truncation_side == "right":
                    ids = ids[:max_length]
                else:
                    ids = ids[-max_length:]
            all_ids.append(ids)

        # Padding
        if padding:
            target_len = max_length if (padding == "max_length" and max_length) else max(len(x) for x in all_ids)
            attention_masks: List[List[int]] = []
            padded: List[List[int]] = []
            for ids in all_ids:
                pad_len = target_len - len(ids)
                if pad_len <= 0:
                    padded.append(ids[:target_len])
                    attention_masks.append([1] * target_len)
                    continue
                if self.padding_side == "right":
                    padded.append(ids + [self.pad_token_id] * pad_len)
                    attention_masks.append([1] * len(ids) + [0] * pad_len)
                else:
                    padded.append([self.pad_token_id] * pad_len + ids)
                    attention_masks.append([0] * pad_len + [1] * len(ids))
        else:
            padded = all_ids
            attention_masks = [[1] * len(ids) for ids in all_ids]

        result: Dict[str, Any] = {"input_ids": padded}
        if return_attention_mask:
            result["attention_mask"] = attention_masks
        if return_token_type_ids:
            result["token_type_ids"] = [[0] * len(ids) for ids in padded]

        if return_tensors == "pt":
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}

        if single and return_tensors != "pt":
            result = {k: v[0] for k, v in result.items()}
        return result

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        thinking_mode: str = "chat",
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        return_tensors: Optional[str] = None,
        drop_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: bool = False,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, List[int], torch.Tensor]:
        """应用 V4 chat 模板。"""
        # 如果不要生成提示，强制最后一条非 user / developer
        if not add_generation_prompt:
            # 直接 encode，不追加 Assistant 标记
            # 通过 trick：附加一条假的 assistant 在末尾，然后从原 prompt 截断
            # 这里采用直接复用 encode_messages 但不让它走"追加 Assistant"逻辑
            # —— 只要 messages 末尾不是 user/developer 即可
            pass

        text = encode_messages(
            messages,
            thinking_mode=thinking_mode,
            context=context,
            drop_thinking=drop_thinking,
            add_default_bos_token=True,
            reasoning_effort=reasoning_effort,
        )
        if not tokenize:
            return text
        ids = self.bpe.encode(text)
        if truncation and max_length and len(ids) > max_length:
            ids = ids[-max_length:]      # chat 模板：通常左截断保留最近上下文
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def parse_assistant_output(self, text: str, thinking_mode: str = "chat") -> Dict[str, Any]:
        """解析模型 generate 出的 assistant 文本为结构化 dict。"""
        return parse_message_from_completion_text(text, thinking_mode=thinking_mode)

    # ------------------------------------------------------------------
    # Vocab utilities
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self.bpe.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.bpe.vocab)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self.bpe.token_to_id(tokens) or self.unk_token_id
        return [self.bpe.token_to_id(t) or self.unk_token_id for t in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.bpe.id_to_token_str(ids) or self.unk_token
        return [self.bpe.id_to_token_str(i) or self.unk_token for i in ids]

    # ------------------------------------------------------------------
    # 持久化（HF 兼容）
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str) -> None:
        d = Path(save_directory)
        d.mkdir(parents=True, exist_ok=True)

        # vocab + merges
        with open(d / VOCAB_FILE, "w", encoding="utf-8") as f:
            json.dump(self.bpe.vocab, f, ensure_ascii=False, indent=2)
        with open(d / MERGES_FILE, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for a, b in self.bpe.merges:
                f.write(f"{a} {b}\n")

        # special tokens
        st_map = {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "additional_special_tokens": [
                t for t in ALL_SPECIAL_TOKENS
                if t not in {self.bos_token, self.eos_token, self.pad_token, self.unk_token}
            ],
        }
        with open(d / SPECIAL_TOKENS_FILE, "w", encoding="utf-8") as f:
            json.dump(st_map, f, ensure_ascii=False, indent=2)

        # tokenizer config
        cfg = {
            "tokenizer_class": "DeepseekV4Tokenizer",
            "model_max_length": self.model_max_length,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "add_bos_token": False,
            "add_eos_token": False,
            "clean_up_tokenization_spaces": False,
            "auto_map": {"AutoTokenizer": ["tokenization_deepseek_v4.DeepseekV4Tokenizer", None]},
        }
        with open(d / TOKENIZER_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs) -> "DeepseekV4Tokenizer":
        d = Path(pretrained_path)
        # 读取配置
        cfg_path = d / TOKENIZER_CONFIG_FILE
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {}
        bpe = BPETokenizer.from_files(
            vocab_file=str(d / VOCAB_FILE),
            merges_file=str(d / MERGES_FILE),
            special_tokens=ALL_SPECIAL_TOKENS,
        )
        return cls(
            bpe_tokenizer=bpe,
            model_max_length=kwargs.get("model_max_length", cfg.get("model_max_length", 4096)),
            padding_side=kwargs.get("padding_side", cfg.get("padding_side", "left")),
            truncation_side=kwargs.get("truncation_side", cfg.get("truncation_side", "right")),
        )

    # ------------------------------------------------------------------
    # 实用工具
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"DeepseekV4Tokenizer(vocab_size={self.vocab_size}, "
            f"model_max_length={self.model_max_length}, "
            f"padding_side={self.padding_side})"
        )
