"""
数据清洗流水线：

清洗规则（按 GPT-3 / Chinchilla / Falcon 论文综合）：
1. 基本清洗：去 HTML / 多空格 / 控制字符 / 不可见字符
2. 语言过滤：langdetect 或简单的字符比例启发式
3. 质量过滤：
   - 长度过短 / 过长
   - 重复 line / n-gram
   - 标点字母比例
   - 平均 token / word 长度
   - 大写比例
4. 安全过滤（可选）：脏话表 + 敏感词
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

from tqdm import tqdm

from deepseek_v4.utils.io import read_jsonl, write_jsonl
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 基础清洗
# ============================================================

# 删除控制字符（保留 \n \t \r）
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 删除 HTML 标签
_HTML_RE = re.compile(r"<[^>]+>")
# 删除多空格
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
# 删除多换行（>2）
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# URL 模式
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Email 模式
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


@dataclass
class TextCleaner:
    """通用文本清洗。"""
    normalize_unicode: bool = True
    remove_html: bool = True
    remove_control_chars: bool = True
    normalize_whitespace: bool = True
    strip_urls: bool = False
    strip_emails: bool = False
    lowercase: bool = False
    max_consecutive_newlines: int = 2

    def __call__(self, text: str) -> str:
        if not text:
            return ""
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        if self.remove_html:
            text = _HTML_RE.sub("", text)
        if self.remove_control_chars:
            text = _CTRL_RE.sub("", text)
        if self.strip_urls:
            text = _URL_RE.sub("", text)
        if self.strip_emails:
            text = _EMAIL_RE.sub("", text)
        if self.normalize_whitespace:
            text = _MULTI_SPACE_RE.sub(" ", text)
            text = _MULTI_NEWLINE_RE.sub("\n" * self.max_consecutive_newlines, text)
        if self.lowercase:
            text = text.lower()
        return text.strip()


# ============================================================
# 语言过滤
# ============================================================

class LanguageFilter:
    """
    基于字符比例的轻量语言过滤（不需要第三方库）。

    若安装了 langdetect / fasttext，会自动使用更准确的分类器。
    """

    CJK_RANGES = [
        (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),  # CJK
        (0x3040, 0x309F), (0x30A0, 0x30FF),  # 日文
        (0xAC00, 0xD7AF),  # 韩文
    ]

    def __init__(self, allowed_languages: Optional[List[str]] = None, threshold: float = 0.5):
        """
        Args:
            allowed_languages: ['zh', 'en'] 或 None 表示不过滤。
            threshold: 主导语言的最低占比。
        """
        self.allowed = set(allowed_languages) if allowed_languages else None
        self.threshold = threshold
        # 尝试加载 langdetect
        self._langdetect = None
        try:
            from langdetect import detect_langs, DetectorFactory
            DetectorFactory.seed = 0
            self._langdetect = detect_langs
        except ImportError:
            pass

    @staticmethod
    def _is_cjk(cp: int) -> bool:
        for lo, hi in LanguageFilter.CJK_RANGES:
            if lo <= cp <= hi:
                return True
        return False

    def detect(self, text: str) -> str:
        """快速检测主导语言。"""
        if not text:
            return "unknown"
        if self._langdetect:
            try:
                langs = self._langdetect(text[:1000])
                if langs:
                    return langs[0].lang
            except Exception:
                pass
        # 启发式：CJK vs Latin
        cjk = 0
        latin = 0
        for ch in text:
            cp = ord(ch)
            if self._is_cjk(cp):
                cjk += 1
            elif ch.isalpha():
                latin += 1
        total = cjk + latin
        if total == 0:
            return "unknown"
        if cjk / total > 0.3:
            return "zh"
        return "en"

    def __call__(self, text: str) -> bool:
        if self.allowed is None:
            return True
        return self.detect(text) in self.allowed


# ============================================================
# 质量过滤
# ============================================================

@dataclass
class QualityFilter:
    """
    基于启发式的质量过滤。

    返回 True 表示通过，False 表示丢弃。
    """
    min_length: int = 30          # 字符数
    max_length: int = 1_000_000
    min_avg_word_length: float = 1.0
    max_avg_word_length: float = 25.0
    max_repeated_line_ratio: float = 0.3
    max_top_ngram_ratio: float = 0.2
    max_symbol_to_word_ratio: float = 0.1
    max_uppercase_ratio: float = 0.4
    min_alpha_ratio: float = 0.4

    def __call__(self, text: str, return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
        def _result(ok: bool, reason: str = "") -> Union[bool, Tuple[bool, str]]:
            return (ok, reason) if return_reason else ok

        if not text:
            return _result(False, "empty")

        # 长度
        if len(text) < self.min_length:
            return _result(False, "too_short")
        if len(text) > self.max_length:
            return _result(False, "too_long")

        words = text.split()
        if not words:
            return _result(False, "no_words")

        # 平均词长
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len < self.min_avg_word_length:
            return _result(False, "avg_word_too_short")
        if avg_len > self.max_avg_word_length:
            return _result(False, "avg_word_too_long")

        # 字母比例
        alpha = sum(1 for c in text if c.isalpha())
        if alpha / max(len(text), 1) < self.min_alpha_ratio:
            return _result(False, "low_alpha_ratio")

        # 大写比例
        latin_alpha = [c for c in text if "a" <= c.lower() <= "z"]
        if latin_alpha:
            upper_ratio = sum(1 for c in latin_alpha if c.isupper()) / len(latin_alpha)
            if upper_ratio > self.max_uppercase_ratio:
                return _result(False, "too_uppercase")

        # 重复行
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) > 3:
            line_counts = Counter(lines)
            duplicated = sum(c for c in line_counts.values() if c > 1)
            if duplicated / len(lines) > self.max_repeated_line_ratio:
                return _result(False, "too_many_repeated_lines")

        # 最常见 trigram 占比
        if len(words) >= 30:
            trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
            top = Counter(trigrams).most_common(1)[0][1]
            if top / len(trigrams) > self.max_top_ngram_ratio:
                return _result(False, "too_repetitive_ngram")

        # 符号占比
        symbols = sum(1 for c in text if c in string.punctuation)
        if symbols / max(len(words), 1) > self.max_symbol_to_word_ratio * 10:
            return _result(False, "too_many_symbols")

        return _result(True, "ok")


# ============================================================
# 安全过滤
# ============================================================

class SafetyFilter:
    """
    简单的脏话 / 敏感词过滤。

    用 trie 实现高效多模匹配。可由用户传入词表。
    """

    DEFAULT_BLOCKLIST: Set[str] = set()  # 默认空，避免误伤

    def __init__(self, blocklist: Optional[Set[str]] = None, max_hits: int = 3):
        self.blocklist = (blocklist or self.DEFAULT_BLOCKLIST)
        self.max_hits = max_hits

    def __call__(self, text: str) -> bool:
        if not self.blocklist:
            return True
        lower = text.lower()
        hits = sum(1 for w in self.blocklist if w in lower)
        return hits < self.max_hits


# ============================================================
# 流水线
# ============================================================

def clean_pipeline(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    text_field: str = "text",
    cleaner: Optional[TextCleaner] = None,
    quality_filter: Optional[QualityFilter] = None,
    language_filter: Optional[LanguageFilter] = None,
    safety_filter: Optional[SafetyFilter] = None,
    keep_failed_to: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, int]:
    """
    完整清洗流水线：jsonl → cleaned jsonl。

    Returns:
        统计字典 {kept, dropped, drop_reasons: {...}}
    """
    cleaner = cleaner or TextCleaner()

    stats = {"total": 0, "kept": 0, "dropped": 0, "drop_reasons": {}}
    failed_fp = None
    if keep_failed_to:
        Path(keep_failed_to).parent.mkdir(parents=True, exist_ok=True)
        failed_fp = open(keep_failed_to, "w", encoding="utf-8")

    def _iter():
        for row in tqdm(read_jsonl(input_path), desc=f"cleaning {Path(input_path).name}",
                        disable=not show_progress):
            stats["total"] += 1
            text = row.get(text_field, "")
            if not isinstance(text, str):
                # 嵌套结构（如 messages）跳过文本清洗（保留原样）
                yield row
                stats["kept"] += 1
                continue

            cleaned = cleaner(text)
            row[text_field] = cleaned

            # 质量过滤
            if quality_filter is not None:
                ok, reason = quality_filter(cleaned, return_reason=True)
                if not ok:
                    _drop(stats, reason, row, failed_fp)
                    continue

            # 语言过滤
            if language_filter is not None and not language_filter(cleaned):
                _drop(stats, "language", row, failed_fp)
                continue

            # 安全过滤
            if safety_filter is not None and not safety_filter(cleaned):
                _drop(stats, "safety", row, failed_fp)
                continue

            stats["kept"] += 1
            yield row

    n = write_jsonl(output_path, _iter())
    if failed_fp:
        failed_fp.close()

    stats["written"] = n
    logger.info(
        f"[Clean] {Path(input_path).name}: kept={stats['kept']:,}/{stats['total']:,} "
        f"({stats['kept']/max(stats['total'],1)*100:.1f}%), "
        f"reasons={dict(sorted(stats['drop_reasons'].items(), key=lambda x: -x[1]))}"
    )
    return stats


def _drop(stats: Dict[str, Any], reason: str, row: Dict[str, Any], failed_fp) -> None:
    stats["dropped"] += 1
    stats["drop_reasons"][reason] = stats["drop_reasons"].get(reason, 0) + 1
    if failed_fp:
        import json
        failed_fp.write(json.dumps({"reason": reason, "row": row}, ensure_ascii=False) + "\n")
