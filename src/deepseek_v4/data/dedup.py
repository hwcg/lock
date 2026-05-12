"""
去重模块（从 0 实现）：

1. **ExactDeduper**：基于 hash 的精确去重
2. **MinHashDeduper**：MinHash + LSH 的近似去重
3. **SimHashDeduper**：SimHash + 海明距离的近似去重

适用场景：
- ExactDeduper：完全相同的文档（最快）
- MinHashDeduper：内容近似（如轻微改写、模板化）
- SimHashDeduper：长文档高效近似匹配

实现要点：
- 全部基于 Python + numpy 实现，无第三方依赖
- 支持流式去重，内存占用与已见文档数线性
- LSH bucket 索引加速 MinHash 查询
"""
from __future__ import annotations

import hashlib
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

from deepseek_v4.utils.io import read_jsonl, write_jsonl
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Exact Dedup
# ============================================================

class ExactDeduper:
    """SHA-256 精确去重。"""

    def __init__(self):
        self.seen: Set[str] = set()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        h = self._hash(text)
        if h in self.seen:
            return True
        self.seen.add(h)
        return False

    def __len__(self) -> int:
        return len(self.seen)


# ============================================================
# Shingling（n-gram）
# ============================================================

# 中英分词 fallback：英文按空格，中文按字符
_WORD_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)


def tokenize_for_shingles(text: str) -> List[str]:
    """简单分词：英文按 word，中文按字。"""
    tokens: List[str] = []
    for chunk in _WORD_RE.findall(text):
        # CJK：按字
        is_cjk = any("一" <= c <= "鿿" for c in chunk)
        if is_cjk:
            tokens.extend(list(chunk))
        else:
            tokens.append(chunk.lower())
    return tokens


def get_shingles(text: str, n: int = 5) -> Set[str]:
    """生成 n-shingle 集合。"""
    toks = tokenize_for_shingles(text)
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


# ============================================================
# MinHash + LSH
# ============================================================

# 一个 64-bit 素数（用于 universal hashing）
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _mh_hash(s: str) -> int:
    """字符串 → 32-bit hash。"""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


class MinHash:
    """单文档的 MinHash 签名。"""

    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        rng = np.random.RandomState(seed)
        # 生成 num_perm 个 (a, b) 用于 universal hashing
        self.a = rng.randint(1, _MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
        self.b = rng.randint(0, _MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
        self.hashvalues = np.full(num_perm, _MAX_HASH, dtype=np.uint64)

    def update(self, shingles: Set[str]) -> None:
        """加入一组 shingles，更新签名。"""
        if not shingles:
            return
        hs = np.array([_mh_hash(s) for s in shingles], dtype=np.uint64)
        # 对每个 shingle，对每个 perm 算 universal hash
        # hs[:, None]: (S, 1), self.a[None, :]: (1, P)
        permuted = ((hs[:, None] * self.a[None, :] + self.b[None, :]) % _MERSENNE_PRIME) & _MAX_HASH
        new_min = permuted.min(axis=0)
        self.hashvalues = np.minimum(self.hashvalues, new_min)

    def jaccard(self, other: "MinHash") -> float:
        if self.num_perm != other.num_perm:
            raise ValueError("num_perm mismatch")
        return float((self.hashvalues == other.hashvalues).mean())


class LSHIndex:
    """
    Locality-Sensitive Hashing index（band 法）。

    把 num_perm 切成 num_bands 段，每段 r 个 hash。
    两个文档某一 band 段相同 → 加入候选集，最后用真实 Jaccard 比较。
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.85, num_bands: Optional[int] = None):
        self.num_perm = num_perm
        self.threshold = threshold
        # 选定 band/r 使得 P(match) 在 threshold 附近最敏感
        if num_bands is None:
            # 经验公式：b 越大 r 越小，召回更高但精度降；
            # 反之亦然。这里取使 (1/b)**(1/r) ≈ threshold 的 (b, r)
            best_b, best_r = self._pick_bands(num_perm, threshold)
        else:
            best_b = num_bands
            best_r = num_perm // num_bands
        self.num_bands = best_b
        self.rows_per_band = best_r
        # band_idx → { band_hash: [doc_id, ...] }
        self.buckets: List[Dict[bytes, List[int]]] = [defaultdict(list) for _ in range(self.num_bands)]
        self._next_id = 0
        self.signatures: Dict[int, np.ndarray] = {}

    @staticmethod
    def _pick_bands(num_perm: int, threshold: float) -> Tuple[int, int]:
        """选 (b, r) 使 b*r = num_perm 且 (1/b)^(1/r) 接近 threshold。"""
        best = (None, None, float("inf"))
        for b in range(1, num_perm + 1):
            if num_perm % b != 0:
                continue
            r = num_perm // b
            t_approx = (1 / b) ** (1 / r)
            diff = abs(t_approx - threshold)
            if diff < best[2]:
                best = (b, r, diff)
        return best[0], best[1]

    def add(self, sig: np.ndarray) -> int:
        """加入签名，返回 doc id。"""
        doc_id = self._next_id
        self._next_id += 1
        self.signatures[doc_id] = sig
        for band in range(self.num_bands):
            start = band * self.rows_per_band
            band_sig = sig[start:start + self.rows_per_band].tobytes()
            self.buckets[band][band_sig].append(doc_id)
        return doc_id

    def query(self, sig: np.ndarray, return_jaccard: bool = False) -> List[int]:
        """查询所有候选近邻 doc id。"""
        candidates: Set[int] = set()
        for band in range(self.num_bands):
            start = band * self.rows_per_band
            band_sig = sig[start:start + self.rows_per_band].tobytes()
            candidates.update(self.buckets[band].get(band_sig, []))
        if not return_jaccard:
            return list(candidates)
        # 真实 Jaccard
        result = []
        for cid in candidates:
            j = float((self.signatures[cid] == sig).mean())
            if j >= self.threshold:
                result.append((cid, j))
        return result


class MinHashDeduper:
    """MinHash + LSH 流式近似去重。"""

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.85,
        ngram_size: int = 5,
        seed: int = 42,
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.seed = seed
        self.index = LSHIndex(num_perm=num_perm, threshold=threshold)

    def signature(self, text: str) -> np.ndarray:
        mh = MinHash(num_perm=self.num_perm, seed=self.seed)
        mh.update(get_shingles(text, n=self.ngram_size))
        return mh.hashvalues

    def is_duplicate(self, text: str) -> bool:
        sig = self.signature(text)
        # 查询候选
        candidates = self.index.query(sig)
        # 真实 Jaccard 验证
        for cid in candidates:
            j = float((self.index.signatures[cid] == sig).mean())
            if j >= self.threshold:
                return True
        # 否则插入
        self.index.add(sig)
        return False

    def __len__(self) -> int:
        return self.index._next_id


# ============================================================
# SimHash
# ============================================================

class SimHashDeduper:
    """
    SimHash 64-bit 签名 + 海明距离阈值去重。

    适合长文档；速度比 MinHash 快但召回略低。
    """

    def __init__(self, hamming_threshold: int = 3, ngram_size: int = 5):
        self.threshold = hamming_threshold
        self.ngram_size = ngram_size
        # 简单全量比较；大规模可改 bucketing
        self.fingerprints: List[int] = []

    @staticmethod
    def _hash64(s: str) -> int:
        h = hashlib.md5(s.encode("utf-8")).digest()
        return struct.unpack("<Q", h[:8])[0]

    def fingerprint(self, text: str) -> int:
        shingles = get_shingles(text, n=self.ngram_size)
        if not shingles:
            return 0
        v = [0] * 64
        for s in shingles:
            h = self._hash64(s)
            for i in range(64):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        fp = 0
        for i in range(64):
            if v[i] > 0:
                fp |= (1 << i)
        return fp

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    def is_duplicate(self, text: str) -> bool:
        fp = self.fingerprint(text)
        for old in self.fingerprints:
            if self._hamming(old, fp) <= self.threshold:
                return True
        self.fingerprints.append(fp)
        return False

    def __len__(self) -> int:
        return len(self.fingerprints)


# ============================================================
# 去重 pipeline
# ============================================================

def dedup_pipeline(
    input_paths: Union[List[str], str, Path],
    output_path: Union[str, Path],
    text_field: str = "text",
    method: str = "minhash",     # exact | minhash | simhash
    threshold: float = 0.85,
    num_perm: int = 128,
    ngram_size: int = 5,
    hamming_threshold: int = 3,
    show_progress: bool = True,
) -> Dict[str, int]:
    """
    跨多个 jsonl 文件流式去重。

    Args:
        input_paths: 单个或多个输入 jsonl
        output_path: 去重后的输出 jsonl
        text_field: 用于计算 hash 的字段（messages 类型会拼接为字符串）
    Returns:
        统计字典
    """
    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]

    if method == "exact":
        deduper = ExactDeduper()
    elif method == "minhash":
        deduper = MinHashDeduper(
            num_perm=num_perm, threshold=threshold, ngram_size=ngram_size,
        )
    elif method == "simhash":
        deduper = SimHashDeduper(hamming_threshold=hamming_threshold, ngram_size=ngram_size)
    else:
        raise ValueError(f"Unknown dedup method: {method}")

    stats = {"total": 0, "kept": 0, "dropped": 0}

    def _extract_text(row: Dict[str, Any]) -> str:
        v = row.get(text_field)
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            # messages 风格：拼接所有 content
            return " ".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in v
            )
        return str(v) if v is not None else ""

    def _iter():
        for p in input_paths:
            iterable = read_jsonl(p)
            if show_progress:
                iterable = tqdm(iterable, desc=f"dedup {Path(p).name}", unit="row")
            for row in iterable:
                stats["total"] += 1
                text = _extract_text(row)
                if not text:
                    stats["dropped"] += 1
                    continue
                if deduper.is_duplicate(text):
                    stats["dropped"] += 1
                    continue
                stats["kept"] += 1
                yield row

    write_jsonl(output_path, _iter())
    logger.info(
        f"[Dedup-{method}] kept={stats['kept']:,}/{stats['total']:,} "
        f"({stats['kept']/max(stats['total'],1)*100:.1f}%)"
    )
    return stats
